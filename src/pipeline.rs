use crate::config::Config;
use crate::llm::{LlmError, LlmRequest, OllamaClient, RawClassification};
use crate::model::{
    ApiError, BatchResponse, Direction, ItemStatus, LabelResult, Transaction,
};
use crate::taxonomy::Taxonomy;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tracing::debug;

/// Orchestrates: chunking → semaphore-bounded concurrent LLM calls →
/// validation → per-item retry → `unknown` fallback. Association with input
/// transactions is strictly positional.
pub struct LabelService {
    client: Arc<OllamaClient>,
    taxonomy: Arc<Taxonomy>,
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
    pub fn new(cfg: &Config, taxonomy: Taxonomy) -> Self {
        let slugs = taxonomy.slugs();
        Self {
            client: Arc::new(OllamaClient::new(cfg, slugs)),
            taxonomy: Arc::new(taxonomy),
            semaphore: Arc::new(Semaphore::new(cfg.concurrency)),
            micro_batch: cfg.micro_batch,
            model: cfg.model.clone(),
            default_language: cfg.language.clone(),
        }
    }

    pub fn taxonomy(&self) -> &Taxonomy {
        &self.taxonomy
    }

    pub fn client(&self) -> &OllamaClient {
        &self.client
    }

    /// Labels a batch of transactions in parallel (micro-batched prompts).
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

        let mut handles = Vec::with_capacity(chunks.len());
        for chunk in chunks {
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
                res.map(|raw| (chunk, raw))
            }));
        }

        let mut results: Vec<LabelResult> = Vec::with_capacity(transactions.len());
        // Preserve input order: chunks are processed in order; join all.
        for handle in handles {
            let (chunk, raw) = handle
                .await
                .map_err(|e| LabelFailure::Backend(LlmError::Unreachable(e.to_string())))?
                .map_err(LabelFailure::Backend)?;

            let validated = self.validate_chunk(&chunk, raw, &language, include_rationale).await;
            results.extend(validated);
        }

        Ok(BatchResponse {
            results,
            batch_ms: started.elapsed().as_millis() as u64,
        })
    }

    /// Validates positional raw classifications against the taxonomy.
    /// Invalid/missing labels get one individual retry, then `unknown`.
    async fn validate_chunk(
        &self,
        chunk: &[Transaction],
        raw: Vec<Option<RawClassification>>,
        language: &str,
        include_rationale: bool,
    ) -> Vec<LabelResult> {
        let mut out = Vec::with_capacity(chunk.len());
        let mut needs_retry: Vec<(usize, Transaction)> = Vec::new();

        for (i, tx) in chunk.iter().enumerate() {
            let direction = Direction::from_amount(tx.amount, false);
            match raw.get(i).and_then(|r| r.as_ref()) {
                Some(rc) => match self.taxonomy.lookup_ci(&rc.category) {
                    Some(cat) => out.push(self.make_result(tx, &cat.slug, language, rc.rationale.clone(), ItemStatus::Ok, direction)),
                    None => needs_retry.push((i, tx.clone())),
                },
                None => needs_retry.push((i, tx.clone())),
            }
        }

        // Individual retries for invalid/missing items.
        for (_, tx) in needs_retry {
            let retry_req = LlmRequest {
                transactions: vec![tx.clone()],
                language: language.to_string(),
                include_rationale,
            };
            let direction = Direction::from_amount(tx.amount, false);
            let outcome = match self.client.classify_batch(&retry_req).await {
                Ok(r) => r
                    .first()
                    .cloned()
                    .flatten()
                    .and_then(|rc| self.taxonomy.lookup_ci(&rc.category).map(|c| (c.slug.clone(), rc.rationale))),
                Err(_) => None,
            };
            match outcome {
                Some((slug, rationale)) => out.push(self.make_result(&tx, &slug, language, rationale, ItemStatus::Ok, direction)),
                None => {
                    debug!(id = %tx.id, "falling back to unknown");
                    let unknown_slug = if direction == Direction::Income {
                        "other_income"
                    } else {
                        "other_expense"
                    };
                    out.push(self.make_result(&tx, unknown_slug, language, None, ItemStatus::FallbackUnknown, direction));
                }
            }
        }

        out
    }

    fn make_result(
        &self,
        tx: &Transaction,
        slug: &str,
        language: &str,
        rationale: Option<String>,
        status: ItemStatus,
        direction: Direction,
    ) -> LabelResult {
        let cat = self
            .taxonomy
            .lookup(slug)
            .expect("slug validated against taxonomy");
        LabelResult {
            id: tx.id.clone(),
            category: slug.to_string(),
            category_label: cat.display_name(language).to_string(),
            direction,
            rationale: rationale.filter(|_| status == ItemStatus::Ok),
            status,
            model: self.model.clone(),
        }
    }
}

/// Convenience: duration formatting for Retry-After.
pub fn retry_after() -> Duration {
    Duration::from_secs(5)
}

impl From<LabelFailure> for ApiError {
    fn from(f: LabelFailure) -> Self {
        match f {
            LabelFailure::Backend(e) => ApiError::backend_unavailable(format!("LLM backend failed: {e}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::taxonomy::builtin;

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
    fn fallback_slug_matches_direction() {
        let t = builtin();
        assert!(t.lookup("other_income").is_some());
        assert!(t.lookup("other_expense").is_some());
    }
}