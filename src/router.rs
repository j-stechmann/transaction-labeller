use crate::api::{health, label_batch, label_single, taxonomy, ApiState};
use crate::pipeline::LabelService;
use axum::routing::{get, post};
use axum::Router;
use std::sync::Arc;

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
        .route("/v1/taxonomy", get(taxonomy))
        .with_state(state)
}