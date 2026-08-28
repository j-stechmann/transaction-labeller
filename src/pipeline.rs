use crate::config::Config;
use crate::llm::{LlmError, LlmRequest, OllamaClient, RawLabel};
use crate::model::{BatchResponse, LabelResult, Transaction};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Semaphore;
use tracing::debug;

/// Context for one validation pass: shared settings for all items in it.
struct RenderCtx {
    include_rationale: bool,
}

/// Orchestrates: chunking → semaphore-bounded concurrent LLM calls →
/// label cleanup → per-item retry → fallback. Association with input
/// transactions is strictly positional.
pub struct LabelService {
    client: Arc<OllamaClient>,
    semaphore: Arc<Semaphore>,
    micro_batch: usize,
    model: String,
    /// Server-default label language (used when request omits `options.language`).
    pub default_language: String,
}

#[derive(Debug, Clone)]
pub enum LabelFailure {
    /// Backend down/overloaded: whole request fails with 503 + Retry-After.
    Backend(LlmError),
}

impl LabelService {
    pub fn new(cfg: &Config) -> Self {
        Self {
            client: Arc::new(OllamaClient::new(cfg)),
            semaphore: Arc::new(Semaphore::new(cfg.concurrency)),
            micro_batch: cfg.micro_batch,
            model: cfg.model.clone(),
            default_language: cfg.language.clone(),
        }
    }

    pub fn client(&self) -> &OllamaClient {
        &self.client
    }

    /// Labels a batch of transactions in parallel (micro-batched prompts).
    ///
    /// Primary calls run under the semaphore; per-item retries acquire their
    /// own permit so concurrency stays bounded. Timeouts degrade item-wise
    /// (fallback) instead of failing the whole request; only unreachable-
    /// backend errors abort with 503.
    pub async fn label(
        &self,
        transactions: Vec<Transaction>,
        language: String,
        include_rationale: bool,
    ) -> Result<BatchResponse, LabelFailure> {
        let started = Instant::now();
        let chunks: Vec<Vec<Transaction>> = transactions
            .chunks(self.micro_batch)
            .map(|c| c.to_vec())
            .collect();
        let chunk_count = chunks.len();

        let mut handles = Vec::with_capacity(chunk_count);
        for chunk in chunks.into_iter() {
            let client = Arc::clone(&self.client);
            let sem = Arc::clone(&self.semaphore);
            let lang = language.clone();
            handles.push(tokio::spawn(async move {
                let _permit = sem
                    .acquire()
                    .await
                    .map_err(|_| LlmError::Unreachable("shutdown".into()))?;
                let req = LlmRequest {
                    transactions: chunk.clone(),
                    language: lang,
                    include_rationale,
                };
                let res = client.classify_batch(&req).await;
                drop(_permit);
                // Timeout → all-None here, where the chunk is still owned:
                // validation will send these items through per-item retry →
                // fallback, keeping the results[i] ↔ transactions[i] contract.
                let n = chunk.len();
                match res {
                    Ok(raw) => Ok((chunk, raw)),
                    Err(LlmError::Timeout(_)) => Ok((chunk, vec![None; n])),
                    Err(e) => Err(e),
                }
            }));
        }

        let ctx = RenderCtx { include_rationale };
        let mut slots: Vec<Vec<LabelResult>> = Vec::with_capacity(chunk_count);
        for handle in handles {
            let (chunk, raw) = match handle.await {
                Ok(Ok(pair)) => pair,
                Ok(Err(e)) => return Err(LabelFailure::Backend(e)),
                Err(join) => {
                    return Err(LabelFailure::Backend(LlmError::Unreachable(format!(
                        "internal task failure: {join}"
                    ))))
                }
            };
            if chunk.is_empty() {
                continue;
            }
            let validated = self.validate_chunk(&chunk, raw, &language, &ctx).await;
            slots.push(validated);
        }

        let mut results: Vec<LabelResult> = Vec::with_capacity(transactions.len());
        for chunk_results in slots {
            results.extend(chunk_results);
        }

        Ok(BatchResponse {
            results,
            batch_ms: started.elapsed().as_millis() as u64,
        })
    }

    /// Validates positional raw labels. Order is preserved: slot `i` of the
    /// output corresponds to `chunk[i]`. Empty/missing labels get one
    /// individual retry, then a generic fallback label.
    async fn validate_chunk(
        &self,
        chunk: &[Transaction],
        raw: Vec<Option<RawLabel>>,
        language: &str,
        ctx: &RenderCtx,
    ) -> Vec<LabelResult> {
        let mut out: Vec<Option<LabelResult>> = vec![None; chunk.len()];
        let mut needs_retry: Vec<(usize, Transaction)> = Vec::new();

        for (i, tx) in chunk.iter().enumerate() {
            match raw.get(i).and_then(|r| r.as_ref()) {
                Some(rl) => {
                    out[i] = Some(self.make_result(tx, &rl.label, rl.rationale.clone(), ctx));
                }
                None => needs_retry.push((i, tx.clone())),
            }
        }

        // Individual retries for invalid/missing items (semaphore-bounded).
        for (i, tx) in needs_retry {
            let outcome = self
                .classify_single(&tx, language, ctx.include_rationale)
                .await;
            match outcome {
                Some((label, rationale)) => {
                    out[i] = Some(self.make_result(&tx, &label, rationale, ctx));
                }
                None => {
                    debug!(id = %tx.id, "falling back after failed labelling");
                    let fallback = if tx.amount < 0.0 {
                        default_expense_label(language)
                    } else {
                        default_income_label(language)
                    };
                    out[i] = Some(self.make_result(&tx, fallback, None, ctx));
                }
            }
        }

        out.into_iter()
            .map(|o| o.expect("every slot is filled"))
            .collect()
    }

    async fn classify_single(
        &self,
        tx: &Transaction,
        language: &str,
        include_rationale: bool,
    ) -> Option<(String, Option<String>)> {
        let req = LlmRequest {
            transactions: vec![tx.clone()],
            language: language.to_string(),
            include_rationale,
        };
        let _permit = self.semaphore.acquire().await.ok()?;
        let res = self.client.classify_batch(&req).await;
        drop(_permit);
        match res {
            Ok(r) => r
                .first()
                .cloned()
                .flatten()
                .map(|rl| (rl.label, rl.rationale)),
            Err(_) => None,
        }
    }

    fn make_result(
        &self,
        tx: &Transaction,
        label: &str,
        rationale: Option<String>,
        ctx: &RenderCtx,
    ) -> LabelResult {
        LabelResult {
            id: tx.id.clone(),
            label: label.to_string(),
            rationale: rationale.filter(|_| ctx.include_rationale),
            model: self.model.clone(),
        }
    }
}

/// Generic fallback labels when the LLM produced nothing usable. Language
/// follows the request language for the shipped defaults.
const DEFAULT_EXPENSE_LABEL_EN: &str = "Other expenses";
const DEFAULT_INCOME_LABEL_EN: &str = "Other income";
const DEFAULT_EXPENSE_LABEL_DE: &str = "Sonstige Ausgaben";
const DEFAULT_INCOME_LABEL_DE: &str = "Sonstige Einnahmen";

fn default_expense_label(language: &str) -> &'static str {
    if language == "de" {
        DEFAULT_EXPENSE_LABEL_DE
    } else {
        DEFAULT_EXPENSE_LABEL_EN
    }
}

fn default_income_label(language: &str) -> &'static str {
    if language == "de" {
        DEFAULT_INCOME_LABEL_DE
    } else {
        DEFAULT_INCOME_LABEL_EN
    }
}

/// Retry-After value for 503 responses.
pub const RETRY_AFTER_SECS: u64 = 5;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn cfg() -> Config {
        Config {
            micro_batch: 2,
            concurrency: 2,
            ..Config::default()
        }
    }

    fn tx(id: &str, amount: f64, cp: &str, purpose: &str) -> Transaction {
        serde_json::from_value(serde_json::json!({
            "id": id, "amount": amount, "counterparty": cp, "purpose": purpose
        }))
        .unwrap()
    }

    #[test]
    fn fallback_labels_by_direction_and_language() {
        assert_eq!(default_expense_label("de"), "Sonstige Ausgaben");
        assert_eq!(default_expense_label("en"), "Other expenses");
        assert_eq!(default_income_label("de"), "Sonstige Einnahmen");
        assert_eq!(default_income_label("en"), "Other income");
        assert_eq!(
            default_expense_label("fr"),
            "Other expenses",
            "non-de falls back to en"
        );
    }

    #[tokio::test]
    async fn label_with_unreachable_backend_fails_backend() {
        let mut cfg = cfg();
        cfg.ollama_url = "http://127.0.0.1:1".into(); // nothing listens
        cfg.request_timeout_secs = 1;
        cfg.max_retries = 0;
        let svc = LabelService::new(&cfg);
        let res = svc
            .label(vec![tx("a", -5.0, "X", "Y")], "de".into(), false)
            .await;
        assert!(matches!(res, Err(LabelFailure::Backend(_))));
    }
}
