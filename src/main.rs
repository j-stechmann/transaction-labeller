use std::sync::Arc;
use tracing::{error, info, warn};

mod api;
mod config;
mod llm;
mod model;
mod pipeline;
mod prompt;
mod router;
mod taxonomy;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "transaction_labeller=info,tower_http=warn".into()),
        )
        .init();

    let cfg = match config::Config::from_env() {
        Ok(c) => c,
        Err(e) => {
            error!("{e}");
            std::process::exit(2);
        }
    };

    let taxonomy = match taxonomy::Taxonomy::load(cfg.taxonomy_path.as_deref()) {
        Ok(t) => t,
        Err(e) => {
            error!("{e}");
            std::process::exit(2);
        }
    };
    info!(
        categories = taxonomy.len(),
        language = %cfg.language,
        model = %cfg.model,
        ollama = %cfg.ollama_url,
        concurrency = cfg.concurrency,
        micro_batch = cfg.micro_batch,
        num_ctx = cfg.num_ctx,
        "starting transaction-labeller"
    );

    if cfg.bind_addr.parse::<std::net::SocketAddr>().map(|a| !a.ip().is_loopback()).unwrap_or(false) {
        warn!("binding to a non-loopback address: this service has no authentication");
    }

    let service = Arc::new(pipeline::LabelService::new(&cfg, taxonomy));
    let app = router::build_router(Arc::clone(&service), cfg.max_batch);

    // Advisory VRAM budget check against the running Ollama instance.
    let strict = std::env::var("TL_STRICT_VRAM").map(|v| v == "1" || v == "true").unwrap_or(false);
    if let Err(msg) = api::vram_check_service(&service, cfg.vram_budget_mb, false).await {
        error!("{msg}");
        if strict {
            std::process::exit(3);
        }
    }

    let listener = tokio::net::TcpListener::bind(&cfg.bind_addr)
        .await
        .unwrap_or_else(|e| panic!("failed to bind {}: {e}", cfg.bind_addr));
    info!(addr = %cfg.bind_addr, "listening");
    axum::serve(listener, app).await.expect("server runs");
}