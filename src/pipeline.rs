use crate::config::Config;
use crate::llm::{LlmError, LlmRequest, OllamaClient, RawClassification};
use crate::model::{
    ApiError, BatchResponse, Direction, ItemStatus, LabelResult, Transaction,
};
use crate::taxonomy::Taxonomy;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Semaphore;
use tracing::debug;

/// Context for one validation pass: shared settings for all items in it.
struct RenderCtx<'a> {
    language: &'a str,
    include_rationale: bool,
}

/// Orchestrates: chunking → semaphore-bounded concurrent LLM calls →
/// validation → per-item retry → fallback. Association with input
/// transactions is strictly positional.
pub struct LabelService {
    client: Arc<OllamaClient>,
    taxonomy: Arc<Taxonomy>,
    semaphore: Arc<Semaphore>,
    micro_batch: usize,
    model: String,
    /// Server-default label language (used when request omits `options.language`).
    pub default_language: String,
    /// Fallback slugs validated against the taxonomy at construction.
    fallback_income: String,
    fallback_expense: String,
}

#[derive(Debug, Clone)]
pub enum LabelFailure {
    /// Backend down/overloaded: whole request fails with 503 + Retry-After.
    Backend(LlmError),
}

impl LabelService {
    pub fn new(cfg: &Config, taxonomy: Taxonomy) -> Self {
        // Fallback slugs must exist in the effective taxonomy (builtin or
        // custom); otherwise a per-item fallback would panic later.
        let fallback_income = pick_fallback(&taxonomy, Direction::Income)
            .unwrap_or_else(|| {
                panic!(
                    "taxonomy must contain a generic income category (looked for `other_income`)"
                )
            });
        let fallback_expense = pick_fallback(&taxonomy, Direction::Expense).unwrap_or_else(|| {
            panic!("taxonomy must contain a generic expense category (looked for `other_expense`)")
        });
        let slugs = taxonomy.slugs();
        Self {
            client: Arc::new(OllamaClient::new(cfg, taxonomy.clone(), slugs)),
            taxonomy: Arc::new(taxonomy),
            semaphore: Arc::new(Semaphore::new(cfg.concurrency)),
            micro_batch: cfg.micro_batch,
            model: cfg.model.clone(),
            default_language: cfg.language.clone(),
            fallback_income,
            fallback_expense,
        }
    }

    pub fn taxonomy(&self) -> &Taxonomy {
        &self.taxonomy
    }

    pub fn client(&self) -> &OllamaClient {
        &self.client
    }

    /// Labels a batch of transactions in parallel (micro-batched prompts).
    ///
    /// Primary calls run under the semaphore; per-item retries acquire their
    /// own permit so concurrency stays bounded. Primary-call timeouts degrade
    /// item-wise (fallback) instead of failing the whole request; only
    /// unreachable-backend errors abort with 503.
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
                res.map(|raw| (chunk, raw))
            }));
        }

        let mut slots: Vec<Vec<LabelResult>> = Vec::with_capacity(chunk_count);
        for handle in handles {
            let (chunk, raw) = match handle.await {
                Ok(Ok(pair)) => pair,
                Ok(Err(e)) => {
                    // Defensive: timeouts are converted to all-None upstream;
                    // remaining backend errors fail wholesale with 503.
                    if matches!(e, LlmError::Timeout(_)) {
                        continue;
                    } else {
                        return Err(LabelFailure::Backend(e));
                    }
                }
                Err(join) => {
                    return Err(LabelFailure::Backend(LlmError::Unreachable(format!(
                        "internal task failure: {join}"
                    ))))
                }
            };
            if chunk.is_empty() {
                continue;
            }
            let validated = self
                .validate_chunk(&chunk, raw, &language, include_rationale)
                .await;
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

    /// Validates positional raw classifications against the taxonomy.
    /// Order is preserved: slot `i` of `out` corresponds to `chunk[i]`.
    /// Invalid/missing labels get one individual retry, then fallback.
    async fn validate_chunk(
        &self,
        chunk: &[Transaction],
        raw: Vec<Option<RawClassification>>,
        language: &str,
        include_rationale: bool,
    ) -> Vec<LabelResult> {
        let ctx = RenderCtx { language, include_rationale };
        let mut out: Vec<Option<LabelResult>> = vec![None; chunk.len()];
        let mut needs_retry: Vec<(usize, Transaction)> = Vec::new();

        for (i, tx) in chunk.iter().enumerate() {
            let direction = Direction::from_amount(tx.amount, false);
            match raw.get(i).and_then(|r| r.as_ref()) {
                Some(rc) => match self.resolve_slug(&rc.category, direction, tx.amount) {
                    Some(slug) => {
                        out[i] = Some(self.make_result(tx, &slug, &ctx, rc.rationale.clone(), ItemStatus::Ok, direction));
                    }
                    None => needs_retry.push((i, tx.clone())),
                },
                None => needs_retry.push((i, tx.clone())),
            }
        }

        // Individual retries for invalid/missing items (semaphore-bounded).
        for (i, tx) in needs_retry {
            let direction = Direction::from_amount(tx.amount, false);
            let outcome = self.classify_single(&tx, ctx.language, ctx.include_rationale).await;
            let resolved = outcome.and_then(|(cat, rat)| self.resolve_slug(&cat, direction, tx.amount).map(|s| (s, rat)));
            match resolved {
                Some((slug, rationale)) => {
                    out[i] = Some(self.make_result(&tx, &slug, &ctx, rationale, ItemStatus::Ok, direction));
                }
                None => {
                    debug!(id = %tx.id, "falling back after failed validation");
                    let fallback = if direction == Direction::Income {
                        &self.fallback_income
                    } else {
                        &self.fallback_expense
                    };
                    out[i] = Some(self.make_result(&tx, fallback, &ctx, None, ItemStatus::FallbackUnknown, direction));
                }
            }
        }

        out.into_iter()
            .map(|o| o.expect("every slot is filled"))
            .collect()
    }

    /// Slug is accepted only if it exists in the taxonomy AND is consistent
    /// with the transaction's direction. `amount == 0` transactions may take
    /// either direction's slug (the sign carries no evidence, design.md §API).
    fn resolve_slug(&self, raw_category: &str, direction: Direction, amount: f64) -> Option<String> {
        let cat = self.taxonomy.lookup_ci(raw_category)?;
        let slug_is_income = is_income_slug(&cat.slug);
        if amount == 0.0 || slug_is_income == (direction == Direction::Income) {
            Some(cat.slug.clone())
        } else {
            None
        }
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
            Ok(r) => r.first().cloned().flatten().map(|rc| (rc.category, rc.rationale)),
            Err(_) => None,
        }
    }

    fn make_result(
        &self,
        tx: &Transaction,
        slug: &str,
        ctx: &RenderCtx<'_>,
        rationale: Option<String>,
        status: ItemStatus,
        direction: Direction,
    ) -> LabelResult {
        let cat = self
            .taxonomy
            .lookup(slug)
            .unwrap_or_else(|| panic!("slug `{slug}` must exist in taxonomy"));
        LabelResult {
            id: tx.id.clone(),
            category: slug.to_string(),
            category_label: cat.display_name(ctx.language).to_string(),
            direction,
            rationale: rationale
                .filter(|_| ctx.include_rationale && status == ItemStatus::Ok),
            status,
            model: self.model.clone(),
        }
    }
}

/// Prefers `other_income`/`other_expense`; falls back to any slug whose name
/// suggests a generic category.
fn pick_fallback(tax: &Taxonomy, dir: Direction) -> Option<String> {
    let want = if dir == Direction::Income {
        "other_income"
    } else {
        "other_expense"
    };
    if tax.lookup(want).is_some() {
        return Some(want.to_string());
    }
    // Custom taxonomies: first slug containing "other" that matches direction.
    tax.iter()
        .find(|c| c.slug.contains("other") && is_income_slug(&c.slug) == (dir == Direction::Income))
        .map(|c| c.slug.clone())
}

/// Direction of a slug by naming convention: income slugs end in `_income`,
/// expense slugs end in `_expense`. The builtin taxonomy follows this; custom
/// taxonomies must too (validated at startup via fallback resolution).
fn is_income_slug(slug: &str) -> bool {
    slug.ends_with("_income")
}

impl From<LabelFailure> for ApiError {
    fn from(f: LabelFailure) -> Self {
        match f {
            LabelFailure::Backend(e) => {
                ApiError::backend_unavailable(format!("LLM backend failed: {e}"))
            }
        }
    }
}

/// Retry-After value for 503 responses.
pub const RETRY_AFTER_SECS: u64 = 5;

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
    fn fallback_slugs_exist_in_builtin() {
        let t = builtin();
        assert!(t.lookup("other_income").is_some());
        assert!(t.lookup("other_expense").is_some());
        assert_eq!(
            pick_fallback(&t, Direction::Income).as_deref(),
            Some("other_income")
        );
        assert_eq!(
            pick_fallback(&t, Direction::Expense).as_deref(),
            Some("other_expense")
        );
    }

    #[test]
    fn custom_taxonomy_needs_other_fallback() {
        let json = r#"{"categories":[
            {"slug":"salary_income","names":{"de":"Gehalt"}},
            {"slug":"my_other_expense","names":{"de":"Rest"}}
        ]}"#;
        let t = Taxonomy::from_str(json).unwrap();
        assert_eq!(pick_fallback(&t, Direction::Expense).as_deref(), Some("my_other_expense"));
        assert!(pick_fallback(&t, Direction::Income).is_none(),
            "no generic income category → startup must fail");
    }

    #[test]
    fn direction_mismatch_is_rejected() {
        let svc = LabelService::new(&cfg(), builtin());
        assert_eq!(
            svc.resolve_slug("salary_income", Direction::Income, -1.0).as_deref(),
            Some("salary_income")
        );
        assert_eq!(
            svc.resolve_slug("SALARY_INCOME", Direction::Income, 100.0).as_deref(),
            Some("salary_income")
        );
        // income slug on expense tx → model error
        assert!(svc.resolve_slug("salary_income", Direction::Expense, -50.0).is_none());
        assert!(svc.resolve_slug("nope", Direction::Expense, -1.0).is_none());
        // amount == 0: either direction allowed
        assert_eq!(
            svc.resolve_slug("salary_income", Direction::Expense, 0.0).as_deref(),
            Some("salary_income")
        );
        assert_eq!(
            svc.resolve_slug("groceries", Direction::Income, 0.0).as_deref(),
            Some("groceries")
        );
    }

    #[test]
    fn is_income_convention() {
        assert!(is_income_slug("salary_income"));
        assert!(!is_income_slug("groceries"));
    }

    #[tokio::test]
    async fn label_with_unreachable_backend_fails_backend() {
        let mut cfg = cfg();
        cfg.ollama_url = "http://127.0.0.1:1".into(); // nothing listens
        cfg.request_timeout_secs = 1;
        cfg.max_retries = 0;
        let svc = LabelService::new(&cfg, builtin());
        let res = svc
            .label(vec![tx("a", -5.0, "X", "Y")], "de".into(), false)
            .await;
        assert!(matches!(res, Err(LabelFailure::Backend(_))));
    }
}