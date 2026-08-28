use crate::model::{
    ApiError, BatchLabelResponse, HealthResponse, LabelBatchRequest, LabelListResponse,
    LabelOptions, LabelSingleRequest, SingleLabelResponse, Transaction,
};
use crate::pipeline::{LabelFailure, LabelService, RETRY_AFTER_SECS};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use std::sync::Arc;
use tracing::{info, warn};
use utoipa::OpenApi;

pub struct ApiState {
    pub service: Arc<LabelService>,
    pub max_batch: usize,
    pub max_field_len: usize,
}

/// Shared language-code check: trims, then requires a 2-letter ASCII code.
/// Returns the normalized (trimmed) code; the caller lowercases when needed.
fn valid_language_code(raw: &str) -> Result<&str, ApiError> {
    let l = raw.trim();
    if l.len() != 2 || !l.chars().all(|c| c.is_ascii_alphabetic()) {
        return Err(ApiError::invalid_request(format!(
            "language must be a 2-letter ISO 639-1 code, got: {raw:?}"
        )));
    }
    Ok(l)
}

/// Request-level validation shared by single + batch endpoints.
fn validate_language(opts: &LabelOptions, state: &ApiState) -> Result<(), ApiError> {
    if let Some(lang) = &opts.language {
        valid_language_code(lang)?;
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
        (status = 200, description = "The transaction id paired with its LLM-generated category name", body = SingleLabelResponse),
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
    match state.service.label(vec![body.transaction], language).await {
        Ok(batch) => match batch.results.into_iter().next() {
            Some(r) => (
                StatusCode::OK,
                Json(SingleLabelResponse {
                    id: r.id,
                    label: r.label,
                }),
            )
                .into_response(),
            None => api_err(
                ApiError::new("internal", "no label produced"),
                StatusCode::INTERNAL_SERVER_ERROR,
                None,
            ),
        },
        Err(failure) => failure_to_response(failure),
    }
}

/// POST /v1/label:batch — parallel labelling of up to `max_batch` transactions.
#[utoipa::path(
    post,
    path = "/v1/label:batch",
    request_body = LabelBatchRequest,
    responses(
        (status = 200, description = "One {id, label} per input transaction, in input order; items never fail wholesale (generic fallback label instead)", body = BatchLabelResponse),
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
    match state.service.label(body.transactions, language).await {
        Ok(resp) => {
            info!(
                items = resp.results.len(),
                ms = started.elapsed().as_millis() as u64,
                "batch labelled"
            );
            (StatusCode::OK, Json(resp)).into_response()
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

/// GET /v1/labels — the label library: labels already known for a language.
/// These are the labels the model is instructed to reuse verbatim.
#[utoipa::path(
    get,
    path = "/v1/labels",
    params(
        ("language" = Option<String>, Query, description = "ISO 639-1 code; defaults to the server language")
    ),
    responses(
        (status = 200, description = "Known labels, most-used first", body = LabelListResponse),
        (status = 400, description = "Bad language code", body = ApiError)
    ),
    tag = "labelling"
)]
pub async fn list_labels(
    State(state): State<Arc<ApiState>>,
    axum::extract::Query(query): axum::extract::Query<LabelListQuery>,
) -> Response {
    let lang = match query_language(&state, &query) {
        Ok(lang) => lang,
        Err(e) => return api_err(e, StatusCode::BAD_REQUEST, None),
    };
    let labels = match state.service.library() {
        Some(lib) => lib.labels_for(&lang),
        None => Vec::new(),
    };
    (StatusCode::OK, Json(LabelListResponse { labels })).into_response()
}

#[derive(Debug, Deserialize)]
pub struct LabelListQuery {
    pub language: Option<String>,
}

fn query_language(state: &ApiState, query: &LabelListQuery) -> Result<String, ApiError> {
    match &query.language {
        Some(l) => Ok(valid_language_code(l)?.to_lowercase()),
        None => Ok(state.service.default_language.clone()),
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
        version = "0.3.0",
        description = "Labels bank transactions with category names generated dynamically by a local LLM (Ollama). There is no fixed taxonomy: the model invents short, consistent labels in the requested language. Direction is implied by the amount sign; the client receives only the label.",
        license(name = "MIT")
    ),
    paths(
        crate::api::label_single,
        crate::api::label_batch,
        crate::api::list_labels,
        crate::api::health
    ),
    components(
        schemas(
            crate::model::Transaction,
            crate::model::LabelOptions,
            crate::model::LabelSingleRequest,
            crate::model::LabelBatchRequest,
            crate::model::SingleLabelResponse,
            crate::model::BatchLabelResponse,
            crate::model::LabeledTransaction,
            crate::model::LabelListResponse,
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
