use crate::api::{health, label_batch, label_single, ApiDoc, ApiState};
use crate::pipeline::LabelService;
use axum::routing::{get, post};
use axum::Router;
use std::sync::Arc;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

pub fn max_field_len() -> usize {
    512
}

pub fn build_router(service: Arc<LabelService>, max_batch: usize) -> Router {
    let state = Arc::new(ApiState {
        service,
        max_batch,
        max_field_len: max_field_len(),
    });
    Router::new()
        .route("/v1/label", post(label_single))
        .route("/v1/label:batch", post(label_batch))
        .route("/v1/health", get(health))
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .with_state(state)
}
