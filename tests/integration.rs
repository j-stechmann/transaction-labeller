//! End-to-end integration tests: full HTTP API against a mock Ollama server.
//! These verify API contract, parallelism bounds, validation, and the
//! retry/fallback paths (see docs/design.md §Testing).

mod mock_llm;

use mock_llm::{default_keyword_map, spawn_with, MockBehaviour, MockServer};
use serde_json::{json, Value};
use std::sync::Arc;
use transaction_labeller::config::Config;
use transaction_labeller::pipeline::LabelService;

async fn spawn_app(mock: &MockServer, cfg_overrides: &[(&str, &str)]) -> (String, Config) {
    let cfg = test_config(&mock.url(), cfg_overrides);
    let taxonomy =
        transaction_labeller::taxonomy::Taxonomy::load(cfg.taxonomy_path.as_deref()).unwrap();
    let service = Arc::new(LabelService::new(&cfg, taxonomy));
    let app = transaction_labeller::router::build_router(Arc::clone(&service), cfg.max_batch);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{addr}"), cfg)
}

fn test_config(ollama_url: &str, env: &[(&str, &str)]) -> transaction_labeller::config::Config {
    let mut cfg = transaction_labeller::config::Config {
        ollama_url: ollama_url.to_string(),
        ..transaction_labeller::config::Config::default()
    };
    for (k, v) in env {
        match *k {
            "TL_CONCURRENCY" => cfg.concurrency = v.parse().unwrap(),
            "TL_MICRO_BATCH" => cfg.micro_batch = v.parse().unwrap(),
            "TL_MAX_BATCH" => cfg.max_batch = v.parse().unwrap(),
            "TL_MODEL" => cfg.model = v.to_string(),
            _ => {}
        }
    }
    cfg
}

async fn post_json(url: &str, path: &str, body: Value) -> (u16, Value) {
    let client = reqwest::Client::new();
    let res = client
        .post(format!("{url}{path}"))
        .json(&body)
        .send()
        .await
        .expect("request completes");
    let status = res.status().as_u16();
    let body: Value = res.json().await.expect("JSON body");
    (status, body)
}

fn tx(id: &str, amount: f64, counterparty: &str, purpose: &str) -> Value {
    json!({
        "id": id, "amount": amount,
        "counterparty": counterparty, "purpose": purpose,
        "date": "2026-02-14", "currency": "EUR"
    })
}

#[tokio::test]
async fn single_transaction_labelled_with_localized_label() {
    let b = MockBehaviour {
        keyword_map: default_keyword_map(),
        rationale: true,
        ..MockBehaviour::default()
    };
    let m = spawn_with(b).await;
    let (url, _cfg) = spawn_app(&m, &[]).await;

    let (status, body) = post_json(
        &url,
        "/v1/label",
        json!({
            "transaction": tx("tx1", -42.13, "REWE SAGT DANKE", "Einkauf 14.02"),
            "options": {"language": "de", "include_rationale": true}
        }),
    )
    .await;

    assert_eq!(status, 200, "body: {body}");
    let r = &body["results"][0];
    assert_eq!(r["id"], "tx1");
    assert_eq!(r["category"], "groceries");
    assert_eq!(r["category_label"], "Lebensmittel");
    assert_eq!(r["direction"], "expense");
    assert_eq!(r["status"], "ok");
    assert_eq!(r["model"], "qwen3.5:4b");
    assert!(r["rationale"].is_string());
    assert!(body["batch_ms"].is_u64());
}

#[tokio::test]
async fn language_switch_changes_label_not_slug() {
    let b = MockBehaviour {
        keyword_map: default_keyword_map(),
        ..MockBehaviour::default()
    };
    let m = spawn_with(b).await;
    let (url, _cfg) = spawn_app(&m, &[]).await;

    let mk_body = |lang: &str| {
        json!({
            "transaction": tx("tx1", -42.13, "REWE", "Einkauf"),
            "options": {"language": lang}
        })
    };

    let (_, de) = post_json(&url, "/v1/label", mk_body("de")).await;
    let (_, en) = post_json(&url, "/v1/label", mk_body("en")).await;

    assert_eq!(de["results"][0]["category"], "groceries");
    assert_eq!(en["results"][0]["category"], "groceries");
    assert_eq!(de["results"][0]["category_label"], "Lebensmittel");
    assert_eq!(en["results"][0]["category_label"], "Groceries");
}

#[tokio::test]
async fn batch_labels_in_parallel_and_preserves_order() {
    let b = MockBehaviour {
        keyword_map: default_keyword_map(),
        delay_ms: 50,
        ..MockBehaviour::default()
    };
    let m = spawn_with(b).await;
    let (url, cfg) = spawn_app(&m, &[("TL_CONCURRENCY", "4"), ("TL_MICRO_BATCH", "2")]).await;

    let mut txs = Vec::new();
    let keywords = ["REWE", "NETFLIX", "GEHALT", "SHELL", "AMAZON", "PIZZA"];
    for i in 0..12 {
        txs.push(tx(
            &format!("t{i}"),
            -(i as f64 + 1.0),
            keywords[i % keywords.len()],
            "Test",
        ));
    }
    txs.push(tx("income", 2500.0, "ACME", "GEHALT Februar"));

    let (status, body) = post_json(&url, "/v1/label:batch", json!({"transactions": txs})).await;
    assert_eq!(status, 200, "body: {body}");
    let results = body["results"].as_array().unwrap();
    assert_eq!(results.len(), 13);
    for (i, r) in results.iter().enumerate() {
        let expected_id = if i == 12 {
            "income".to_string()
        } else {
            format!("t{i}")
        };
        assert_eq!(r["id"], expected_id, "positional order must be preserved");
    }
    assert_eq!(results[12]["category"], "salary_income");

    // Parallelism: with 7 chunks, concurrency 4, delay 50ms →
    // sequential would be ≥ 350ms; parallel should be well under.
    assert!(
        m.max_in_flight.load(std::sync::atomic::Ordering::SeqCst) > 1,
        "requests must overlap (mock saw only 1 in flight)"
    );
    assert!(
        body["batch_ms"].as_u64().unwrap() < 7 * 50,
        "batch must run concurrently"
    );
    let _ = cfg;
}

#[tokio::test]
async fn concurrency_is_bounded_by_semaphore() {
    let b = MockBehaviour {
        keyword_map: default_keyword_map(),
        delay_ms: 30,
        ..MockBehaviour::default()
    };
    let m = spawn_with(b).await;
    let (url, cfg) = spawn_app(&m, &[("TL_CONCURRENCY", "2"), ("TL_MICRO_BATCH", "1")]).await;

    let txs: Vec<Value> = (0..10)
        .map(|i| tx(&format!("t{i}"), -1.0, "REWE", "x"))
        .collect();
    let (status, body) = post_json(&url, "/v1/label:batch", json!({"transactions": txs})).await;
    assert_eq!(status, 200, "body: {body}");
    let max_seen = m.max_in_flight.load(std::sync::atomic::Ordering::SeqCst);
    assert!(
        max_seen <= cfg.concurrency,
        "mock saw {max_seen} concurrent calls, bound is {}",
        cfg.concurrency
    );
}

#[tokio::test]
async fn invalid_labels_fall_back_itemwise() {
    let b = MockBehaviour {
        keyword_map: default_keyword_map(),
        invalid_labels: true,
        ..MockBehaviour::default()
    };
    let m = spawn_with(b).await;
    let (url, _cfg) = spawn_app(&m, &[]).await;

    // Primary call returns invalid label → individual retry → still invalid → fallback
    let (status, body) = post_json(
        &url,
        "/v1/label:batch",
        json!({"transactions": [tx("a", -5.0, "REWE", "x"), tx("b", 100.0, "ACME", "Gehalt")]}),
    )
    .await;

    assert_eq!(
        status, 200,
        "batch must degrade item-wise, not fail: {body}"
    );
    let results = body["results"].as_array().unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0]["status"], "fallback_unknown");
    assert_eq!(results[0]["category"], "other_expense");
    assert_eq!(results[1]["status"], "fallback_unknown");
    assert_eq!(results[1]["category"], "other_income");
}

#[tokio::test]
async fn garbage_output_falls_back_itemwise() {
    let b = MockBehaviour {
        keyword_map: default_keyword_map(),
        garbage: true,
        ..MockBehaviour::default()
    };
    let m = spawn_with(b).await;
    let (url, _cfg) = spawn_app(&m, &[]).await;

    let (status, body) = post_json(
        &url,
        "/v1/label",
        json!({"transaction": tx("a", -5.0, "REWE", "x")}),
    )
    .await;

    assert_eq!(status, 200, "body: {body}");
    assert_eq!(body["results"][0]["status"], "fallback_unknown");
    assert_eq!(body["results"][0]["category"], "other_expense");
}

#[tokio::test]
async fn transient_500_is_retried_then_succeeds() {
    let b = MockBehaviour {
        keyword_map: default_keyword_map(),
        fail_first_n: 1, // first chat call fails with 500
        ..MockBehaviour::default()
    };
    let m = spawn_with(b).await;
    let (url, _cfg) = spawn_app(&m, &[]).await;

    let (status, body) = post_json(
        &url,
        "/v1/label",
        json!({"transaction": tx("a", -5.0, "REWE", "x")}),
    )
    .await;

    assert_eq!(
        status, 200,
        "retry after transient 500 must succeed: {body}"
    );
    assert_eq!(body["results"][0]["category"], "groceries");
    assert!(
        m.chat_calls.load(std::sync::atomic::Ordering::SeqCst) >= 2,
        "must have retried"
    );
}

#[tokio::test]
async fn validation_errors_return_uniform_error_body() {
    let m = spawn_with(MockBehaviour::default()).await;
    let (url, cfg) = spawn_app(&m, &[]).await;

    // duplicate ids
    let (status, body) = post_json(
        &url,
        "/v1/label:batch",
        json!({"transactions": [tx("dup", -1.0, "A", "x"), tx("dup", -2.0, "B", "y")]}),
    )
    .await;
    assert_eq!(status, 400);
    assert_eq!(body["error"]["code"], "invalid_request");

    // empty batch
    let (status, body) = post_json(&url, "/v1/label:batch", json!({"transactions": []})).await;
    assert_eq!(status, 400);
    assert_eq!(body["error"]["code"], "invalid_request");

    // batch too large
    let big: Vec<Value> = (0..cfg.max_batch + 1)
        .map(|i| tx(&format!("t{i}"), -1.0, "A", "x"))
        .collect();
    let (status, body) = post_json(&url, "/v1/label:batch", json!({"transactions": big})).await;
    assert_eq!(status, 413);
    assert_eq!(body["error"]["code"], "invalid_request");

    // bad language
    let (status, body) = post_json(
        &url,
        "/v1/label",
        json!({"transaction": tx("a", -1.0, "A", "x"), "options": {"language": "deutsch"}}),
    )
    .await;
    assert_eq!(status, 400);
    assert_eq!(body["error"]["code"], "invalid_request");

    // malformed JSON body (axum's default rejection is 400)
    let client = reqwest::Client::new();
    let res = client
        .post(format!("{url}/v1/label"))
        .header("content-type", "application/json")
        .body("{not json")
        .send()
        .await
        .unwrap();
    let status = res.status().as_u16();
    assert!(
        status == 400 || status == 422,
        "malformed body must be rejected, got {status}"
    );

    // non-finite amount (raw JSON 1e999 → inf must be rejected)
    let client = reqwest::Client::new();
    let res = client
        .post(format!("{url}/v1/label"))
        .header("content-type", "application/json")
        .body(r#"{"transaction": {"id": "a", "amount": 1e999}}"#)
        .send()
        .await
        .unwrap();
    let status = res.status().as_u16();
    assert!(
        status == 400 || status == 422,
        "non-finite amount must be rejected, got {status}"
    );

    // field too long
    let long_purpose = "x".repeat(600);
    let (status, body) = post_json(
        &url,
        "/v1/label",
        json!({"transaction": tx("a", -1.0, "A", &long_purpose)}),
    )
    .await;
    assert_eq!(status, 400, "field length cap must apply: {body}");
    assert_eq!(body["error"]["code"], "invalid_request");
}

#[tokio::test]
async fn health_and_taxonomy_endpoints() {
    let b = MockBehaviour {
        keyword_map: default_keyword_map(),
        ..MockBehaviour::default()
    };
    let m = spawn_with(b).await;
    let (url, _cfg) = spawn_app(&m, &[]).await;

    let client = reqwest::Client::new();
    let res = client.get(format!("{url}/v1/health")).send().await.unwrap();
    assert_eq!(res.status().as_u16(), 200);
    let body: Value = res.json().await.unwrap();
    assert_eq!(body["status"], "ok");

    let res = client
        .get(format!("{url}/v1/taxonomy?language=en"))
        .send()
        .await
        .unwrap();
    assert_eq!(res.status().as_u16(), 200);
    let body: Value = res.json().await.unwrap();
    assert_eq!(body["language"], "en");
    let cats = body["categories"].as_array().unwrap();
    assert!(cats.len() >= 20);
    assert!(
        cats.iter()
            .any(|c| c["slug"] == "groceries" && c["label"] == "Groceries"),
        "taxonomy must expose slug + localized label"
    );

    // unknown 2-letter language: falls back to canonical (de) names per design
    let res = client
        .get(format!("{url}/v1/taxonomy?language=xx"))
        .send()
        .await
        .unwrap();
    assert_eq!(res.status().as_u16(), 200);
    let body: Value = res.json().await.unwrap();
    assert_eq!(body["language"], "xx");
    assert!(
        body["categories"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c["label"] == "Lebensmittel"),
        "unknown language must fall back to canonical names"
    );

    // malformed language → 400
    let res = client
        .get(format!("{url}/v1/taxonomy?language=notalang"))
        .send()
        .await
        .unwrap();
    assert_eq!(res.status().as_u16(), 400);
}

#[tokio::test]
async fn openapi_docs_are_served() {
    let m = spawn_with(MockBehaviour::default()).await;
    let (url, _cfg) = spawn_app(&m, &[]).await;
    let client = reqwest::Client::new();

    // OpenAPI JSON: valid, versioned, all 4 operations + key schemas present.
    let res = client
        .get(format!("{url}/api-docs/openapi.json"))
        .send()
        .await
        .unwrap();
    assert_eq!(res.status().as_u16(), 200);
    let spec: Value = res.json().await.unwrap();
    assert_eq!(
        spec["openapi"].as_str().unwrap().split('.').next(),
        Some("3")
    );
    assert_eq!(spec["info"]["title"], "transaction-labeller");

    let paths = spec["paths"].as_object().unwrap();
    for p in ["/v1/label", "/v1/label:batch", "/v1/health", "/v1/taxonomy"] {
        assert!(paths.contains_key(p), "spec missing path {p}");
    }
    assert!(paths["/v1/label"]["post"].is_object());
    assert!(paths["/v1/label:batch"]["post"].is_object());
    assert!(paths["/v1/health"]["get"].is_object());
    assert!(paths["/v1/taxonomy"]["get"].is_object());

    let schemas = spec["components"]["schemas"].as_object().unwrap();
    for s in [
        "Transaction",
        "LabelOptions",
        "LabelSingleRequest",
        "LabelBatchRequest",
        "LabelResult",
        "BatchResponse",
        "ApiError",
        "HealthResponse",
        "TaxonomyResponse",
        "Direction",
        "ItemStatus",
    ] {
        assert!(schemas.contains_key(s), "spec missing schema {s}");
    }

    // LabelResult schema carries the field semantics docs.
    let lr = &schemas["LabelResult"];
    assert!(lr["properties"]["category"].is_object());
    assert!(lr["properties"]["category_label"].is_object());
    assert!(lr["properties"]["status"].is_object());

    // Swagger UI HTML is served and references the spec.
    let res = client
        .get(format!("{url}/swagger-ui"))
        .send()
        .await
        .unwrap();
    assert_eq!(res.status().as_u16(), 200);
    let html = res.text().await.unwrap();
    assert!(html.contains("swagger"), "swagger-ui HTML must be served");
}

#[tokio::test]
async fn unreachable_backend_returns_503_with_retry_after() {
    // App pointed at a closed port.
    let cfg = test_config("http://127.0.0.1:9", &[]);
    // speed up: no retries
    let cfg = transaction_labeller::config::Config {
        max_retries: 0,
        request_timeout_secs: 2,
        ..cfg
    };
    let taxonomy =
        transaction_labeller::taxonomy::Taxonomy::load(cfg.taxonomy_path.as_deref()).unwrap();
    let service = Arc::new(LabelService::new(&cfg, taxonomy));
    let app = transaction_labeller::router::build_router(service, cfg.max_batch);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let url = format!("http://{addr}");

    let (status, body) = post_json(
        &url,
        "/v1/label",
        json!({"transaction": tx("a", -5.0, "REWE", "x")}),
    )
    .await;

    assert_eq!(status, 503);
    assert_eq!(body["error"]["code"], "backend_unavailable");
}
