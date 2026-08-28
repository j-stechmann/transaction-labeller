use crate::config::Config;
use crate::library::LabelLibrary;
use crate::llm::{LlmError, LlmRequest, OllamaClient, RawLabel};
use crate::model::{BatchLabelResponse, LabeledTransaction, Transaction};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Semaphore;
use tracing::debug;

/// Orchestrates: chunking → semaphore-bounded concurrent LLM calls →
/// label cleanup → per-item retry → fallback. Association with input
/// transactions is strictly positional.
pub struct LabelService {
    client: Arc<OllamaClient>,
    semaphore: Arc<Semaphore>,
    micro_batch: usize,
    /// Server-default label language (used when request omits `options.language`).
    pub default_language: String,
    /// Existing labels shown to the model (prefer-reuse) and extended with
    /// every label actually returned. `None` = library disabled.
    library: Option<Arc<LabelLibrary>>,
}

#[derive(Debug, Clone)]
pub enum LabelFailure {
    /// Backend down/overloaded: whole request fails with 503 + Retry-After.
    Backend(LlmError),
}

/// Outcome of validating one chunk: the positional results (fallbacks
/// included — every slot must stay filled) plus the subset of labels the
/// model actually produced, which is what the library records.
struct ValidatedChunk {
    results: Vec<LabeledTransaction>,
    model_labels: Vec<String>,
}

/// True when `label` is one of the generic fallbacks for `language`. The
/// shipped defaults are static strings, so this is exact. If a model happens
/// to return the fallback wording on its own it is also skipped from
/// recording — harmless (a generic label is never worth learning anyway).
fn is_fallback_label(language: &str, label: &str) -> bool {
    label == default_expense_label(language) || label == default_income_label(language)
}

impl LabelService {
    pub fn new(cfg: &Config) -> Self {
        let library = if cfg.label_library.is_empty() {
            None
        } else {
            Some(Arc::new(LabelLibrary::open(
                std::path::PathBuf::from(&cfg.label_library),
                cfg.library_prompt_max,
            )))
        };
        Self {
            client: Arc::new(OllamaClient::new(cfg)),
            semaphore: Arc::new(Semaphore::new(cfg.concurrency)),
            micro_batch: cfg.micro_batch,
            default_language: cfg.language.clone(),
            library,
        }
    }

    pub fn client(&self) -> &OllamaClient {
        &self.client
    }

    /// Library access for the API (e.g. `GET /v1/labels`).
    pub fn library(&self) -> Option<&Arc<LabelLibrary>> {
        self.library.as_ref()
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
    ) -> Result<BatchLabelResponse, LabelFailure> {
        let started = Instant::now();
        let library_labels = self.library_labels(&language);
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
            let library_labels = library_labels.clone();
            handles.push(tokio::spawn(async move {
                let _permit = sem
                    .acquire()
                    .await
                    .map_err(|_| LlmError::Unreachable("shutdown".into()))?;
                let req = LlmRequest {
                    transactions: chunk.clone(),
                    language: lang,
                    library_labels,
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

        let mut slots: Vec<Vec<LabeledTransaction>> = Vec::with_capacity(chunk_count);
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
            let validated = self.validate_chunk(&chunk, raw, &language).await;
            // Learn: only labels the model actually produced grow the library.
            // Fallback labels are excluded so a flaky-backend burst cannot
            // inflate the generic label into the top-ranked prompt entry.
            if let Some(lib) = &self.library {
                lib.record(&language, &validated.model_labels);
            }
            slots.push(validated.results);
        }

        let mut results: Vec<LabeledTransaction> = Vec::with_capacity(transactions.len());
        for chunk_results in slots {
            results.extend(chunk_results);
        }

        debug!(
            items = results.len(),
            ms = started.elapsed().as_millis() as u64,
            "batch labelled"
        );

        Ok(BatchLabelResponse { results })
    }

    /// Library labels for `language` (empty when disabled).
    fn library_labels(&self, language: &str) -> Vec<String> {
        self.library
            .as_ref()
            .map(|lib| lib.labels_for(language))
            .unwrap_or_default()
    }

    /// Validates positional raw labels. Order is preserved: slot `i` of the
    /// output corresponds to `chunk[i]`. Empty/missing labels get one
    /// individual retry, then a generic fallback label. `model_labels` holds
    /// only the labels the model produced (retries included), so callers can
    /// record them without polluting the library with fallbacks.
    async fn validate_chunk(
        &self,
        chunk: &[Transaction],
        raw: Vec<Option<RawLabel>>,
        language: &str,
    ) -> ValidatedChunk {
        let mut out: Vec<Option<LabeledTransaction>> = vec![None; chunk.len()];
        let mut needs_retry: Vec<(usize, Transaction)> = Vec::new();

        for (i, tx) in chunk.iter().enumerate() {
            match raw.get(i).and_then(|r| r.as_ref()) {
                Some(rl) => {
                    out[i] = Some(LabeledTransaction {
                        id: tx.id.clone(),
                        label: rl.label.clone(),
                    });
                }
                None => needs_retry.push((i, tx.clone())),
            }
        }

        // Individual retries for invalid/missing items (semaphore-bounded).
        for (i, tx) in needs_retry {
            let outcome = self.classify_single(&tx, language).await;
            match outcome {
                Some(label) => {
                    out[i] = Some(LabeledTransaction {
                        id: tx.id.clone(),
                        label,
                    });
                }
                None => {
                    debug!(id = %tx.id, "falling back after failed labelling");
                    let fallback = if tx.amount < 0.0 {
                        default_expense_label(language)
                    } else {
                        default_income_label(language)
                    };
                    out[i] = Some(LabeledTransaction {
                        id: tx.id.clone(),
                        label: fallback.to_string(),
                    });
                }
            }
        }

        let mut results = Vec::with_capacity(out.len());
        let mut model_labels = Vec::new();
        for o in out.into_iter().flatten() {
            if !is_fallback_label(language, &o.label) {
                model_labels.push(o.label.clone());
            }
            results.push(o);
        }
        ValidatedChunk {
            results,
            model_labels,
        }
    }

    async fn classify_single(&self, tx: &Transaction, language: &str) -> Option<String> {
        let req = LlmRequest {
            transactions: vec![tx.clone()],
            language: language.to_string(),
            library_labels: self.library_labels(language),
        };
        let _permit = self.semaphore.acquire().await.ok()?;
        let res = self.client.classify_batch(&req).await;
        drop(_permit);
        res.ok()
            .and_then(|r| r.first().cloned().flatten().map(|rl| rl.label))
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
        cfg.label_library = String::new(); // hermetic: no file I/O
        let svc = LabelService::new(&cfg);
        let res = svc.label(vec![tx("a", -5.0, "X", "Y")], "de".into()).await;
        assert!(matches!(res, Err(LabelFailure::Backend(_))));
    }

    #[test]
    fn is_fallback_label_detects_shipped_defaults_only() {
        assert!(is_fallback_label("de", "Sonstige Ausgaben"));
        assert!(is_fallback_label("de", "Sonstige Einnahmen"));
        assert!(is_fallback_label("en", "Other expenses"));
        assert!(is_fallback_label("en", "Other income"));
        assert!(!is_fallback_label("de", "Lebensmittel"));
        assert!(
            !is_fallback_label("de", "Sonstige Ausgaben "),
            "no trimming"
        );
        assert!(!is_fallback_label("fr", "Sonstige Ausgaben"));
    }
}
