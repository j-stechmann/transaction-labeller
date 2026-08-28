use crate::config::Config;
use crate::model::Transaction;
use crate::prompt;
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;
use thiserror::Error;
use tracing::{debug, warn};

/// One LLM call = one micro-batch of transactions.
pub struct LlmRequest {
    pub transactions: Vec<Transaction>,
    pub language: String,
}

/// Raw parsed model output: slot `k` of the parsed vector corresponds to the
/// k-th input transaction (positional association); the model-echoed index is
/// not used for mapping.
#[derive(Debug, Clone)]
#[allow(dead_code)] // `index` kept for debugging/logging of raw output
pub struct RawLabel {
    pub index: usize,
    pub label: String,
}

#[derive(Error, Debug, Clone)]
pub enum LlmError {
    #[error("LLM backend unreachable: {0}")]
    Unreachable(String),
    #[error("LLM backend error (HTTP {status}): {body}")]
    Http { status: u16, body: String },
    #[error("LLM response was not valid JSON: {0}")]
    BadResponse(String),
    #[error("LLM request timed out after {0}s")]
    Timeout(u64),
}

pub struct OllamaClient {
    http: reqwest::Client,
    base_url: String,
    model: String,
    num_ctx: u32,
    request_timeout: Duration,
    max_retries: u32,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    message: Option<ChatMessage>,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    content: Option<String>,
}

#[derive(Debug, Serialize)]
struct ChatRequestBody {
    model: String,
    messages: Vec<Value>,
    stream: bool,
    format: Value,
    keep_alive: String,
    /// `false` disables the model's thinking mode: thinking models would burn
    /// the entire `num_predict` budget on reasoning and never emit content.
    think: bool,
    options: Value,
}

impl OllamaClient {
    pub fn new(cfg: &Config) -> Self {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(3))
            .build()
            .expect("reqwest client builds with static config");
        Self {
            http,
            base_url: cfg.ollama_url.clone(),
            model: cfg.model.clone(),
            num_ctx: cfg.num_ctx,
            request_timeout: Duration::from_secs(cfg.request_timeout_secs),
            max_retries: cfg.max_retries,
        }
    }

    /// Liveness/model reachability probe.
    pub async fn health(&self) -> Result<(), LlmError> {
        let url = format!("{}/api/tags", self.base_url);
        let res = self
            .http
            .get(&url)
            .timeout(Duration::from_secs(5))
            .send()
            .await
            .map_err(|e| LlmError::Unreachable(e.to_string()))?;
        if res.status().is_success() {
            Ok(())
        } else {
            Err(LlmError::Http {
                status: res.status().as_u16(),
                body: "health check failed".into(),
            })
        }
    }

    /// Model size in bytes from `/api/tags` (used for the VRAM budget check).
    pub async fn model_size_bytes(&self) -> Result<Option<u64>, LlmError> {
        let url = format!("{}/api/tags", self.base_url);
        let res = self
            .http
            .get(&url)
            .timeout(Duration::from_secs(5))
            .send()
            .await
            .map_err(|e| LlmError::Unreachable(e.to_string()))?;
        let body: Value = res
            .json()
            .await
            .map_err(|e| LlmError::BadResponse(e.to_string()))?;
        let tags = body
            .get("models")
            .and_then(|m| m.as_array())
            .cloned()
            .unwrap_or_default();
        // Prefer an exact tag match; fall back to base-name match only if no
        // exact match exists (avoids `4b` matching `4b-instruct`'s size).
        let want_base = self.model.split(':').next().unwrap_or(&self.model);
        let mut fallback_size = None;
        for tag in tags {
            let name = tag.get("name").and_then(|n| n.as_str()).unwrap_or("");
            if name == self.model {
                return Ok(tag.get("size").and_then(|s| s.as_u64()));
            }
            let base = name.split(':').next().unwrap_or(name);
            if base == want_base && fallback_size.is_none() {
                fallback_size = tag.get("size").and_then(|s| s.as_u64());
            }
        }
        Ok(fallback_size)
    }

    /// Classify one micro-batch. Retries transient failures with exponential
    /// backoff + jitter; association with transactions is positional (the
    /// model-echoed `index` is ignored for mapping).
    pub async fn classify_batch(
        &self,
        req: &LlmRequest,
    ) -> Result<Vec<Option<RawLabel>>, LlmError> {
        let system = prompt::system_prompt(&req.language);
        let user = prompt::user_prompt(&req.transactions);
        let schema = prompt::response_schema();

        let body = ChatRequestBody {
            model: self.model.clone(),
            messages: vec![
                json!({"role": "system", "content": system}),
                json!({"role": "user", "content": user}),
            ],
            stream: false,
            format: schema,
            keep_alive: "10m".to_string(),
            think: false,
            options: json!({
                "num_ctx": self.num_ctx,
                "temperature": 0.0,
                "num_predict": 768,
            }),
        };

        let mut attempt: u32 = 0;
        loop {
            attempt += 1;
            match self.attempt(&body).await {
                Ok(content) => {
                    let parsed = parse_model_output(&content, req.transactions.len());
                    return Ok(parsed);
                }
                Err(e @ LlmError::Timeout(_)) => {
                    // Timeout exhausted → per-item fallback upstream, not a 503.
                    warn!(attempt, error = %e, "LLM call timed out; degrading item-wise");
                    return Ok(vec![None; req.transactions.len()]);
                }
                Err(e) => {
                    let transient = matches!(
                        e,
                        LlmError::Unreachable(_)
                            | LlmError::Http {
                                status: 429 | 500..=599,
                                ..
                            }
                    );
                    if !transient || attempt > self.max_retries {
                        return Err(e);
                    }
                    let backoff = backoff_delay(attempt);
                    warn!(
                        attempt,
                        backoff_ms = backoff.as_millis() as u64,
                        error = %e,
                        "LLM call failed, retrying"
                    );
                    tokio::time::sleep(backoff).await;
                }
            }
        }
    }

    async fn attempt(&self, body: &ChatRequestBody) -> Result<String, LlmError> {
        let url = format!("{}/api/chat", self.base_url);
        let res = self
            .http
            .post(&url)
            .json(body)
            .timeout(self.request_timeout)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    LlmError::Timeout(self.request_timeout.as_secs())
                } else {
                    LlmError::Unreachable(e.to_string())
                }
            })?;

        let status = res.status();
        let text = res
            .text()
            .await
            .map_err(|e| LlmError::BadResponse(e.to_string()))?;
        if !status.is_success() {
            return Err(LlmError::Http {
                status: status.as_u16(),
                body: text.chars().take(500).collect(),
            });
        }

        let parsed: ChatResponse =
            serde_json::from_str(&text).map_err(|e| LlmError::BadResponse(e.to_string()))?;
        parsed
            .message
            .and_then(|m| m.content)
            .ok_or_else(|| LlmError::BadResponse("missing message.content".into()))
    }
}

fn backoff_delay(attempt: u32) -> Duration {
    let base_ms = 200u64.saturating_mul(4u64.saturating_pow(attempt.saturating_sub(1)));
    let jitter = rand::thread_rng().gen_range(0..=base_ms / 4);
    Duration::from_millis(base_ms + jitter)
}

/// Extracts a JSON object from model output, tolerating markdown fences and
/// surrounding prose. Tries successive `{` positions so prose containing a
/// brace does not poison the parse. Returns labels mapped positionally:
/// slot `i` corresponds to input transaction `i`; missing/invalid entries are
/// `None`. The model-echoed `index` is ignored for association.
pub fn parse_model_output(content: &str, expected_len: usize) -> Vec<Option<RawLabel>> {
    let mut out: Vec<Option<RawLabel>> = vec![None; expected_len];
    let Some(json_str) = extract_json(content) else {
        debug!(content = %content, "no JSON found in model output");
        return out;
    };

    let value: Value = match serde_json::from_str(&json_str) {
        Ok(v) => v,
        Err(e) => {
            debug!(error = %e, "model output is not valid JSON");
            return out;
        }
    };

    let Some(items) = value.get("results").and_then(|r| r.as_array()) else {
        return out;
    };

    for (pos, item) in items.iter().enumerate() {
        if pos >= expected_len {
            break;
        }
        let label = match item.get("label").and_then(|c| c.as_str()) {
            Some(c) => sanitize_label(c),
            None => continue,
        };
        if label.is_empty() || out[pos].is_some() {
            continue;
        }
        out[pos] = Some(RawLabel { index: pos, label });
    }
    out
}

/// Labels are free-form model output: trim, collapse whitespace, bound length.
fn sanitize_label(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut prev_space = false;
    for c in raw.trim().chars() {
        if c.is_whitespace() {
            if !prev_space {
                out.push(' ');
            }
            prev_space = true;
        } else if !c.is_control() {
            out.push(c);
            prev_space = false;
        }
    }
    out.chars().take(64).collect()
}

/// Finds the first parseable JSON object in a string (skips markdown fences
/// and prose; tries successive `{` candidates).
fn extract_json(content: &str) -> Option<String> {
    let mut search_from = 0usize;
    while let Some(rel) = content[search_from..].find('{') {
        let start = search_from + rel;
        if let Some(candidate) = balanced_object(&content[start..]) {
            if serde_json::from_str::<Value>(&candidate).is_ok() {
                return Some(candidate);
            }
        }
        search_from = start + 1;
    }
    None
}

/// Returns the outermost balanced `{...}` at the start of `s`, respecting
/// string literals and escapes.
fn balanced_object(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    if bytes.first() != Some(&b'{') {
        return None;
    }
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate() {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(s[..=i].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_plain_json() {
        let out = parse_model_output(
            r#"{"results":[{"index":0,"label":"Lebensmittel"},{"index":1,"label":"Einkommen"}]}"#,
            2,
        );
        assert_eq!(out[0].as_ref().unwrap().label, "Lebensmittel");
        assert_eq!(out[1].as_ref().unwrap().label, "Einkommen");
    }

    #[test]
    fn parse_markdown_fenced_json() {
        let content =
            "Here you go:\n```json\n{\"results\":[{\"index\":0,\"label\":\"Restaurant\"}]}\n```";
        let out = parse_model_output(content, 2);
        assert_eq!(out[0].as_ref().unwrap().label, "Restaurant");
        assert!(out[1].is_none());
    }

    #[test]
    fn parse_json_with_surrounding_prose() {
        let content = "Sure! {\"results\":[{\"index\":1,\"label\":\"Miete\"}]} hope that helps";
        let out = parse_model_output(content, 3);
        // Positional: first result slot maps to first transaction regardless
        // of the echoed index.
        assert_eq!(out[0].as_ref().unwrap().label, "Miete");
        assert!(out[1].is_none());
        assert!(out[2].is_none());
    }

    #[test]
    fn echoed_indices_are_ignored() {
        let content = r#"{"results":[{"index":5,"label":"A"},{"label":"B"},{"index":0}]}"#;
        let out = parse_model_output(content, 3);
        assert_eq!(out[0].as_ref().unwrap().label, "A");
        assert_eq!(out[1].as_ref().unwrap().label, "B");
        // third entry has no label → stays unmapped
        assert!(out[2].is_none());
    }

    #[test]
    fn extra_results_beyond_input_are_dropped() {
        let content = r#"{"results":[{"label":"A"},{"label":"B"}]}"#;
        let out = parse_model_output(content, 1);
        assert_eq!(out[0].as_ref().unwrap().label, "A");
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn results_map_positionally_in_order() {
        let content = r#"{"results":[{"label":"A"},{"label":"B"}]}"#;
        let out = parse_model_output(content, 2);
        assert_eq!(out[0].as_ref().unwrap().label, "A");
        assert_eq!(out[1].as_ref().unwrap().label, "B");
    }

    #[test]
    fn braces_inside_strings_are_tolerated() {
        let content = r#"{"results":[{"label":"Zinsen & Gebühren {Sparen}"}]}"#;
        let out = parse_model_output(content, 1);
        assert_eq!(out[0].as_ref().unwrap().label, "Zinsen & Gebühren {Sparen}");
    }

    #[test]
    fn labels_are_sanitized() {
        // Actual control characters and newlines embedded via concat! of
        // escaped and raw pieces (format! escaping of JSON braces is unwieldy).
        let nl = "\n";
        let bell = "\u{0007}";
        let content = "{\"results\":[{\"label\":\"  Lebensmittel ".to_string()
            + &nl.replace('\n', "\\n")
            + "\"},{\"label\":\"a  b\"},{\"label\":\" x"
            + &bell.replace('\u{0007}', "\\u0007")
            + "y\"},{\"label\":\"\"}]}";
        let out = parse_model_output(&content, 4);
        assert_eq!(out[0].as_ref().unwrap().label, "Lebensmittel");
        assert_eq!(out[1].as_ref().unwrap().label, "a b");
        assert_eq!(out[2].as_ref().unwrap().label, "xy");
        assert!(out[3].is_none(), "empty label is invalid");
    }

    #[test]
    fn labels_are_length_capped() {
        let long = "x".repeat(200);
        let content = format!(r#"{{"results":[{{"label":"{long}"}}]}}"#);
        let out = parse_model_output(&content, 1);
        assert_eq!(out[0].as_ref().unwrap().label.len(), 64);
    }

    #[test]
    fn garbage_yields_all_none() {
        assert!(parse_model_output("no json at all", 2)
            .iter()
            .all(|o| o.is_none()));
        assert!(parse_model_output("{\"results\": 5}", 2)
            .iter()
            .all(|o| o.is_none()));
    }
}
