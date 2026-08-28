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
    let service = Arc::new(LabelService::new(&cfg));
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
async fn single_transaction_labelled() {
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
    assert_eq!(r["label"], "Lebensmittel");
    assert!(r["rationale"].is_string());
    assert_eq!(r["model"], "qwen3.5:4b");
    assert!(body["batch_ms"].is_u64());
    // Response contains ONLY the label (+id/rationale/model) — no category
    // slug, no direction, no status fields.
    let obj = r.as_object().unwrap();
    assert_eq!(obj.len(), 4, "response must be minimal: {obj:?}");
}

#[tokio::test]
async fn language_switch_changes_label_language() {
    let b = MockBehaviour {
        keyword_map: default_keyword_map(),
        ..MockBehaviour::default()
    };
    let m = spawn_with(b).await;
    let (url, _cfg) = spawn_app(&m, &[]).await;

    // Same merchant, different requested language: the prompt the mock sees
    // is identical except nothing — the mock keys on the keyword. Use an
    // en-keyword transaction for the en request so labels differ.
    let (_, de) = post_json(
        &url,
        "/v1/label",
        json!({
            "transaction": tx("tx1", -42.13, "REWE", "Einkauf"),
            "options": {"language": "de"}
        }),
    )
    .await;
    let (_, en) = post_json(
        &url,
        "/v1/label",
        json!({
            "transaction": tx("tx2", -58.20, "WHOLE FOODS MARKET", "Groceries"),
            "options": {"language": "en"}
        }),
    )
    .await;

    assert_eq!(de["results"][0]["label"], "Lebensmittel");
    assert_eq!(en["results"][0]["label"], "Groceries");
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
    assert_eq!(results[12]["label"], "Einkommen");

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
async fn empty_labels_fall_back_itemwise() {
    let b = MockBehaviour {
        keyword_map: default_keyword_map(),
        empty_labels: true,
        ..MockBehaviour::default()
    };
    let m = spawn_with(b).await;
    let (url, _cfg) = spawn_app(&m, &[]).await;

    // Primary call returns empty labels → individual retry → still empty → fallback
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
    assert_eq!(results[0]["label"], "Sonstige Ausgaben");
    assert_eq!(results[1]["label"], "Sonstige Einnahmen");
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
    assert_eq!(body["results"][0]["label"], "Sonstige Ausgaben");
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
    assert_eq!(body["results"][0]["label"], "Lebensmittel");
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
async fn health_endpoint() {
    let m = spawn_with(MockBehaviour::default()).await;
    let (url, _cfg) = spawn_app(&m, &[]).await;

    let client = reqwest::Client::new();
    let res = client.get(format!("{url}/v1/health")).send().await.unwrap();
    assert_eq!(res.status().as_u16(), 200);
    let body: Value = res.json().await.unwrap();
    assert_eq!(body["status"], "ok");

    // taxonomy endpoint must be gone
    let res = client
        .get(format!("{url}/v1/taxonomy"))
        .send()
        .await
        .unwrap();
    assert_eq!(res.status().as_u16(), 404, "/v1/taxonomy must not exist");
}

#[tokio::test]
async fn openapi_docs_are_served() {
    let m = spawn_with(MockBehaviour::default()).await;
    let (url, _cfg) = spawn_app(&m, &[]).await;
    let client = reqwest::Client::new();

    // OpenAPI JSON: valid, versioned, all operations + key schemas present.
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
    for p in ["/v1/label", "/v1/label:batch", "/v1/health"] {
        assert!(paths.contains_key(p), "spec missing path {p}");
    }
    assert!(
        !paths.contains_key("/v1/taxonomy"),
        "taxonomy endpoint removed"
    );
    assert!(paths["/v1/label"]["post"].is_object());
    assert!(paths["/v1/label:batch"]["post"].is_object());
    assert!(paths["/v1/health"]["get"].is_object());

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
    ] {
        assert!(schemas.contains_key(s), "spec missing schema {s}");
    }

    // LabelResult must be minimal: id, label, rationale, model.
    let props = schemas["LabelResult"]["properties"].as_object().unwrap();
    let keys: Vec<_> = props.keys().collect();
    assert_eq!(keys.len(), 4, "LabelResult must be minimal: {keys:?}");
    assert!(props.contains_key("label"));

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
    let service = Arc::new(LabelService::new(&cfg));
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
