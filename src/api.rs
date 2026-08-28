use crate::model::{
    ApiError, BatchResponse, HealthResponse, LabelBatchRequest, LabelOptions, LabelSingleRequest,
    Transaction,
};
use crate::pipeline::{LabelFailure, LabelService, RETRY_AFTER_SECS};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use std::sync::Arc;
use tracing::{info, warn};
use utoipa::OpenApi;

pub struct ApiState {
    pub service: Arc<LabelService>,
    pub max_batch: usize,
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
        let _ = state; // reserved for future checks
    }
    Ok(())
}

/// Caps string field lengths so a single transaction cannot blow the prompt
/// context (design.md: prompts must survive `TL_NUM_CTX`).
fn validate_field_lengths(txs: &[Transaction], max_len: usize) -> Result<(), ApiError> {
    for (i, tx) in txs.iter().enumerate() {
        let check = |name: &str, v: &str| {
            if v.len() > max_len {
                Err(ApiError::invalid_request(format!(
                    "transaction {i} (id {:?}): field `{name}` exceeds {max_len} bytes",
                    tx.id
                )))
            } else {
                Ok(())
            }
        };
        check("id", &tx.id)?;
        check("counterparty", &tx.counterparty)?;
        check("purpose", &tx.purpose)?;
        check("date", &tx.date)?;
        if tx.currency.len() > 8 {
            return Err(ApiError::invalid_request(format!(
                "transaction {i} (id {:?}): field `currency` exceeds 8 bytes",
                tx.id
            )));
        }
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
#[utoipa::path(
    post,
    path = "/v1/label",
    request_body = LabelSingleRequest,
    responses(
        (status = 200, description = "Transaction labelled with an LLM-generated category name", body = BatchResponse),
        (status = 400, description = "Invalid input (bad language, over-long field)", body = ApiError),
        (status = 422, description = "Body fails validation (e.g. non-finite amount)", body = ApiError),
        (status = 503, description = "LLM backend unreachable/overloaded; Retry-After: 5", body = ApiError)
    ),
    tag = "labelling"
)]
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
    if let Err(e) =
        validate_field_lengths(std::slice::from_ref(&body.transaction), state.max_field_len)
    {
        return api_err(e, StatusCode::BAD_REQUEST, None);
    }

    let language = body
        .options
        .effective_language(&state.service.default_language);
    match state
        .service
        .label(
            vec![body.transaction],
            language,
            body.options.include_rationale,
        )
        .await
    {
        Ok(batch) => (StatusCode::OK, Json(batch)).into_response(),
        Err(failure) => failure_to_response(failure),
    }
}

/// POST /v1/label:batch — parallel labelling of up to `max_batch` transactions.
#[utoipa::path(
    post,
    path = "/v1/label:batch",
    request_body = LabelBatchRequest,
    responses(
        (status = 200, description = "Transactions labelled in parallel; input order preserved; items never fail wholesale (generic fallback label instead)", body = BatchResponse),
        (status = 400, description = "Empty batch, duplicate ids, bad language, over-long fields", body = ApiError),
        (status = 413, description = "Batch exceeds TL_MAX_BATCH (default 100)", body = ApiError),
        (status = 422, description = "Body fails validation (e.g. non-finite amount)", body = ApiError),
        (status = 503, description = "LLM backend unreachable/overloaded; Retry-After: 5", body = ApiError)
    ),
    tag = "labelling"
)]
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
    if let Err(e) = validate_field_lengths(&body.transactions, state.max_field_len) {
        return api_err(e, StatusCode::BAD_REQUEST, None);
    }

    let language = body
        .options
        .effective_language(&state.service.default_language);
    let started = std::time::Instant::now();
    match state
        .service
        .label(body.transactions, language, body.options.include_rationale)
        .await
    {
        Ok(batch) => {
            info!(
                items = batch.results.len(),
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
#[utoipa::path(
    get,
    path = "/v1/health",
    responses(
        (status = 200, description = "Service alive and Ollama reachable", body = HealthResponse),
        (status = 503, description = "Service alive but backend degraded", body = HealthResponse)
    ),
    tag = "service"
)]
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

/// OpenAPI specification root: aggregates all schemas and operations.
#[derive(OpenApi)]
#[openapi(
    info(
        title = "transaction-labeller",
        version = "0.2.0",
        description = "Labels bank transactions with category names generated dynamically by a local LLM (Ollama). There is no fixed taxonomy: the model invents short, consistent labels in the requested language. Direction is implied by the amount sign; the client receives only the label.",
        license(name = "MIT")
    ),
    paths(
        crate::api::label_single,
        crate::api::label_batch,
        crate::api::health
    ),
    components(
        schemas(
            crate::model::Transaction,
            crate::model::LabelOptions,
            crate::model::LabelSingleRequest,
            crate::model::LabelBatchRequest,
            crate::model::LabelResult,
            crate::model::BatchResponse,
            crate::model::ApiError,
            crate::model::ErrorBody,
            crate::model::HealthResponse
        )
    ),
    tags(
        (name = "labelling", description = "Transaction labelling"),
        (name = "service", description = "Service metadata: health")
    )
)]
pub struct ApiDoc;
