use crate::model::{
    ApiError, ItemStatus, LabelOptions, LabelResult, Transaction,
};
use crate::pipeline::{LabelFailure, LabelService, RETRY_AFTER_SECS};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use std::sync::Arc;
use tracing::{info, warn};

#[derive(Debug, Deserialize)]
pub struct LabelSingleRequest {
    pub transaction: Transaction,
    #[serde(default)]
    pub options: LabelOptions,
}

#[derive(Debug, Deserialize)]
pub struct LabelBatchRequest {
    pub transactions: Vec<Transaction>,
    #[serde(default)]
    pub options: LabelOptions,
}

pub struct ApiState {
    pub service: Arc<LabelService>,
    pub max_batch: usize,
    #[allow(dead_code)]
    pub max_field_len: usize,
}

/// Request-level validation shared by single + batch endpoints.
fn validate_language(opts: &LabelOptions, state: &ApiState) -> Result<(), ApiError> {
    if let Some(lang) = &opts.language {
        let l = lang.trim();
        if l.len() != 2 || !l.chars().all(|c| c.is_ascii_alphabetic()) {
            return Err(ApiError::invalid_request(format!(
                "language must be a 2-letter ISO 639-1 code, got: {lang:?}"
            )));
        }
        let _ = state; // reserved for future taxonomy language coverage checks
    }
    Ok(())
}

fn validate_ids(txs: &[Transaction]) -> Result<(), ApiError> {
    let mut seen = std::collections::HashSet::with_capacity(txs.len());
    for (i, tx) in txs.iter().enumerate() {
        if !seen.insert(tx.id.as_str()) {
            return Err(ApiError::invalid_request(format!(
                "duplicate transaction id {:?} (index {i}); ids must be unique",
                tx.id
            )));
        }
    }
    Ok(())
}

fn api_err(err: ApiError, status: StatusCode, headers: Option<(&str, &str)>) -> Response {
    let mut res = (status, Json(err)).into_response();
    if let Some((name, value)) = headers {
        if let Ok(v) = value.parse() {
            res.headers_mut().insert(
                axum::http::header::HeaderName::from_bytes(name.as_bytes())
                    .expect("static header name"),
                v,
            );
        }
    }
    res
}

/// POST /v1/label — single transaction.
pub async fn label_single(
    State(state): State<Arc<ApiState>>,
    Json(body): Json<LabelSingleRequest>,
) -> Response {
    if let Err(e) = validate_language(&body.options, &state) {
        return api_err(e, StatusCode::BAD_REQUEST, None);
    }
    if let Err(e) = validate_ids(std::slice::from_ref(&body.transaction)) {
        return api_err(e, StatusCode::BAD_REQUEST, None);
    }

    let language = body.options.effective_language(&state.service.default_language);
    match state
        .service
        .label(vec![body.transaction], language, body.options.include_rationale)
        .await
    {
        Ok(mut batch) => match batch.results.pop() {
            Some(result) => {
                let resp = SingleResponse {
                    result,
                    latency_ms: batch.batch_ms,
                };
                (StatusCode::OK, Json(resp)).into_response()
            }
            None => api_err(
                ApiError::new("internal", "no result produced"),
                StatusCode::INTERNAL_SERVER_ERROR,
                None,
            ),
        },
        Err(failure) => failure_to_response(failure),
    }
}

#[derive(Debug, serde::Serialize)]
pub struct SingleResponse {
    #[serde(flatten)]
    pub result: LabelResult,
    pub latency_ms: u64,
}

/// POST /v1/label:batch — parallel labelling of up to `max_batch` transactions.
pub async fn label_batch(
    State(state): State<Arc<ApiState>>,
    Json(body): Json<LabelBatchRequest>,
) -> Response {
    if body.transactions.is_empty() {
        return api_err(
            ApiError::invalid_request("transactions must not be empty"),
            StatusCode::BAD_REQUEST,
            None,
        );
    }
    if body.transactions.len() > state.max_batch {
        return api_err(
            ApiError::invalid_request(format!(
                "batch too large: {} transactions (max {}); chunk client-side",
                body.transactions.len(),
                state.max_batch
            )),
            StatusCode::PAYLOAD_TOO_LARGE,
            None,
        );
    }
    if let Err(e) = validate_language(&body.options, &state) {
        return api_err(e, StatusCode::BAD_REQUEST, None);
    }
    if let Err(e) = validate_ids(&body.transactions) {
        return api_err(e, StatusCode::BAD_REQUEST, None);
    }

    let language = body.options.effective_language(&state.service.default_language);
    let started = std::time::Instant::now();
    match state
        .service
        .label(body.transactions, language, body.options.include_rationale)
        .await
    {
        Ok(batch) => {
            let n = batch.results.len();
            let fallback = batch
                .results
                .iter()
                .filter(|r| r.status == ItemStatus::FallbackUnknown)
                .count();
            info!(
                items = n,
                fallbacks = fallback,
                ms = started.elapsed().as_millis() as u64,
                "batch labelled"
            );
            (StatusCode::OK, Json(batch)).into_response()
        }
        Err(failure) => failure_to_response(failure),
    }
}

fn failure_to_response(failure: LabelFailure) -> Response {
    match failure {
        LabelFailure::Backend(e) => {
            warn!(error = %e, "LLM backend failure");
            api_err(
                ApiError::backend_unavailable(format!("LLM backend failed: {e}")),
                StatusCode::SERVICE_UNAVAILABLE,
                Some(("Retry-After", &RETRY_AFTER_SECS.to_string())),
            )
        }
    }
}

/// GET /v1/health — liveness + backend reachability.
pub async fn health(State(state): State<Arc<ApiState>>) -> Response {
    match state.service.client().health().await {
        Ok(()) => (
            StatusCode::OK,
            Json(serde_json::json!({"status": "ok", "backend": "reachable"})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"status": "degraded", "backend": format!("{e}")})),
        )
            .into_response(),
    }
}

/// GET /v1/taxonomy — effective taxonomy with per-language display names.
pub async fn taxonomy(State(state): State<Arc<ApiState>>) -> Response {
    let cats: Vec<serde_json::Value> = state
        .service
        .taxonomy()
        .iter()
        .map(|c| serde_json::json!({ "slug": c.slug }))
        .collect();
    (StatusCode::OK, Json(serde_json::json!({ "categories": cats }))).into_response()
}

/// VRAM advisory check: warn (or fail in strict mode) if model weights exceed
/// 80% of the budget.
pub async fn vram_check_service(
    service: &LabelService,
    budget_mb: u64,
    strict: bool,
) -> Result<(), String> {
    let limit = (budget_mb as f64 * 0.8) as u64;
    match service.client().model_size_bytes().await {
        Ok(Some(bytes)) => {
            let size_mb = bytes / (1024 * 1024);
            if size_mb > limit {
                let msg = format!(
                    "model size {size_mb} MB exceeds 80% of VRAM budget {budget_mb} MB ({limit} MB allowed)"
                );
                if strict {
                    Err(msg)
                } else {
                    tracing::warn!("{msg}");
                    Ok(())
                }
            } else {
                info!(size_mb, budget_mb, "VRAM budget check passed");
                Ok(())
            }
        }
        Ok(None) => {
            tracing::warn!("could not determine model size for VRAM check");
            Ok(())
        }
        Err(e) => {
            tracing::warn!("VRAM check skipped: {e}");
            Ok(())
        }
    }
}