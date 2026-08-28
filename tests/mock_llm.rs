//! Mock Ollama server for integration tests: mimics the subset of the
//! `/api/chat` and `/api/tags` APIs the client uses, with configurable
//! behaviour (keyword mapping, delays, failures, malformed output).

use axum::response::IntoResponse;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Default, Clone)]
pub struct MockBehaviour {
    /// Keyword → slug mapping applied to the user prompt content.
    pub keyword_map: HashMap<String, String>,
    /// Artificial delay per chat request (ms).
    pub delay_ms: u64,
    /// Respond with HTTP 500 for the first N chat requests (1-based).
    pub fail_first_n: usize,
    /// Return unparseable output instead of valid JSON.
    pub garbage: bool,
    /// Include rationale entries in the response.
    pub rationale: bool,
    /// Emit labels that are NOT in the taxonomy (forces fallback path).
    pub invalid_labels: bool,
}

pub struct MockServer {
    pub addr: SocketAddr,
    #[allow(dead_code)]
    pub chat_calls: Arc<AtomicUsize>,
    #[allow(dead_code)]
    pub max_in_flight: Arc<AtomicUsize>,
    #[allow(dead_code)]
    pub behaviour: Arc<Mutex<MockBehaviour>>,
}

/// Keyword→slug mapping for realistic test data (German + English).
pub fn default_keyword_map() -> HashMap<String, String> {
    HashMap::from([
        ("REWE", "groceries"),
        ("EDEKA", "groceries"),
        ("ALDI", "groceries"),
        ("LIDL", "groceries"),
        ("NETTO", "groceries"),
        ("MIETE", "housing"),
        ("Gehalt", "salary_income"),
        ("GEHALT", "salary_income"),
        ("LOHN", "salary_income"),
        ("Salary", "salary_income"),
        ("SPARKASSE", "transfer"),
        ("DKB", "transfer"),
        ("AMAZON", "shopping"),
        ("ZALANDO", "shopping"),
        ("Apotheke", "health"),
        ("APOTHEKE", "health"),
        ("DB Vertrieb", "transport"),
        ("SHELL", "transport"),
        ("TANKSTELLE", "transport"),
        ("NETFLIX", "subscriptions"),
        ("SPOTIFY", "subscriptions"),
        ("Finanzamt", "taxes_fees"),
        ("FINANZAMT", "taxes_fees"),
        ("RESTAURANT", "dining"),
        ("PIZZA", "dining"),
        ("STARBUCKS", "dining"),
        ("Bargeld", "cash_withdrawal"),
        ("Geldautomat", "cash_withdrawal"),
        ("Erstattung", "refund"),
        ("STORNO", "refund"),
        ("Erstattung Storno", "refund"),
        ("Rotes Kreuz", "donations"),
        ("Spende", "donations"),
        ("Kursgebühr", "education"),
        ("Volkshochschule", "education"),
        ("Kino", "leisure"),
        ("Cineplex", "leisure"),
        ("Miete", "housing"),
        ("Vonovia", "housing"),
        ("Allianz", "insurance"),
        ("Versicherung", "insurance"),
        ("Finanzamt", "taxes_fees"),
        ("Einkommensteuer", "taxes_fees"),
        ("Visa", "credit_card_settlement"),
        ("Kreditkartenabrechnung", "credit_card_settlement"),
        ("Geldautomat", "cash_withdrawal"),
        ("Tagesgeld", "savings_investing"),
        ("Amazon", "shopping"),
        ("Medikamente", "health"),
        ("Apotheke", "health"),
        ("Kreditkarte", "credit_card_settlement"),
        ("Kreditkartenabrechnung", "credit_card_settlement"),
        ("Kreditkarte", "credit_card_settlement"),
    ])
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}

/// Spawns the mock server on an ephemeral port with default behaviour.
#[allow(dead_code)]
pub async fn spawn() -> MockServer {
    spawn_with(MockBehaviour::default()).await
}

pub async fn spawn_with(behaviour: MockBehaviour) -> MockServer {
    let chat_calls = Arc::new(AtomicUsize::new(0));
    let in_flight = Arc::new(AtomicUsize::new(0));
    let max_in_flight = Arc::new(AtomicUsize::new(0));
    let behaviour = Arc::new(Mutex::new(behaviour));

    let app = {
        let chat_calls = Arc::clone(&chat_calls);
        let in_flight = Arc::clone(&in_flight);
        let max_in_flight = Arc::clone(&max_in_flight);
        let behaviour = Arc::clone(&behaviour);
        axum::Router::new().route(
            "/api/chat",
            axum::routing::post(move |axum::Json(body): axum::Json<Value>| {
                let chat_calls = Arc::clone(&chat_calls);
                let in_flight = Arc::clone(&in_flight);
                let max_in_flight = Arc::clone(&max_in_flight);
                let behaviour = Arc::clone(&behaviour);
                async move {
                    chat_calls.fetch_add(1, Ordering::SeqCst);
                    let cur = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                    max_in_flight.fetch_max(cur, Ordering::SeqCst);

                    let b = behaviour.lock().await.clone();
                    if b.delay_ms > 0 {
                        tokio::time::sleep(std::time::Duration::from_millis(b.delay_ms)).await;
                    }

                    let call_idx = chat_calls.load(Ordering::SeqCst);
                    let resp = if b.fail_first_n >= call_idx {
                        (
                            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                            axum::Json(json!({"error": "mock failure"})),
                        )
                            .into_response()
                    } else if b.garbage {
                        axum::Json(json!({"message": {"content": "I cannot answer in JSON, sorry!"}}))
                            .into_response()
                    } else {
                        let user = body_user_prompt(&body);
                        let results = classify_prompt(&user, &b);
                        axum::Json(json!({
                            "message": {"content": serde_json::to_string(&json!({"results": results})).unwrap()}
                        }))
                            .into_response()
                    };

                    in_flight.fetch_sub(1, Ordering::SeqCst);
                    resp
                }
            }),
        )
        .route(
            "/api/tags",
            axum::routing::get(|| async {
                axum::Json(json!({
                    "models": [
                        {"name": "qwen3.5:4b", "size": 3_400_000_000u64}
                    ]
                }))
            }),
        )
    };

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    MockServer {
        addr,
        chat_calls,
        max_in_flight,
        behaviour,
    }
}

impl MockServer {
    pub fn url(&self) -> String {
        format!("http://{}", self.addr)
    }
}

/// Extracts the user prompt from the chat request body.
fn body_user_prompt(body: &Value) -> String {
    body.get("messages")
        .and_then(|m| m.as_array())
        .and_then(|a| a.last())
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string()
}

/// Deterministic keyword classification of the prompt, mirroring the contract:
/// one result per `[i]` line, in positional order. Longest matching keyword
/// wins (stable across runs regardless of HashMap iteration order).
fn classify_prompt(user_prompt: &str, behaviour: &MockBehaviour) -> Vec<Value> {
    let mut results = Vec::new();
    for line in user_prompt.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix('[') else {
            continue;
        };
        let Some((idx_str, body)) = rest.split_once(']') else {
            continue;
        };
        let Ok(index) = idx_str.trim().parse::<usize>() else {
            continue;
        };
        let category = if behaviour.invalid_labels {
            "made_up_category".to_string()
        } else {
            let upper = body.to_uppercase();
            let mut best: Option<(usize, &String)> = None;
            for (kw, slug) in &behaviour.keyword_map {
                let matched = body.contains(kw.as_str()) || upper.contains(&kw.to_uppercase());
                if matched {
                    let better = match &best {
                        Some((len, _)) => kw.len() > *len,
                        None => true,
                    };
                    if better {
                        best = Some((kw.len(), slug));
                    }
                }
            }
            best.map(|(_, slug)| slug.clone()).unwrap_or_else(|| {
                if body.contains("amount=-") || body.contains("amount=0") {
                    "other_expense".to_string()
                } else {
                    "other_income".to_string()
                }
            })
        };
        let mut item = json!({"index": index, "category": category});
        if behaviour.rationale {
            item["rationale"] = json!("keyword match");
        }
        results.push(item);
    }
    results
}
