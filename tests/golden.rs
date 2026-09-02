//! Golden result tests (correctness of results through the pipeline).
//!
//! The golden set (`tests/golden/cases.json`) contains real-world-style
//! transactions with expected labels (in the requested language). Two modes:
//!
//! 1. **Deterministic mode (default, `cargo test`)**: runs the golden set
//!    through the full pipeline against the keyword-mapping mock LLM. This
//!    verifies the prompt → parsing → mapping path preserves the contract
//!    (positional order, localized labels).
//! 2. **Live eval (`cargo test -- --ignored live_eval`)**: the full set is
//!    labelled in ONE request against a real Ollama model at temperature 0
//!    (single batch is required: with dynamic labels the model keeps wording
//!    consistent within a batch, but wording shifts across batch
//!    compositions). Expected labels are *semantic sets* — any label in the
//!    set counts as correct, because free-form wording varies (e.g.
//!    `Miete` ≈ `Mietzahlung` ≈ `Wohnen`). Requires a running Ollama with the
//!    model pulled; skipped by default so CI needs no GPU.

mod mock_llm;

use mock_llm::{default_keyword_map, spawn_with, MockBehaviour};
use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;
use transaction_labeller::config::Config;
use transaction_labeller::pipeline::LabelService;

#[derive(Debug, Deserialize)]
struct GoldenCase {
    name: String,
    transaction: Value,
    expected: Value,
}

#[derive(Debug, Deserialize)]
struct GoldenExpectation {
    /// Any of these labels (normalized comparison) counts as correct.
    /// First entry is used as the display expectation.
    label: Value,
}

impl GoldenExpectation {
    fn labels(&self) -> Vec<String> {
        match &self.label {
            Value::String(s) => vec![s.clone()],
            Value::Array(a) => a
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect(),
            _ => panic!("expected.label must be a string or array of strings"),
        }
    }
}

fn load_golden() -> Vec<GoldenCase> {
    let raw = include_str!("golden/cases.json");
    serde_json::from_str(raw).expect("golden cases must parse")
}

/// Live-eval model override (`TL_MODEL=qwen3.5:4b` for the fast GPU profile),
/// validated like `Config::from_env` would (trim + non-empty) so an empty or
/// whitespace value fails loudly here instead of reaching Ollama as `""`.
fn model_from_env() -> String {
    match std::env::var("TL_MODEL") {
        Ok(v) => {
            let v = v.trim().to_string();
            assert!(!v.is_empty(), "TL_MODEL must not be empty");
            v
        }
        Err(_) => Config::default().model,
    }
}

async fn run_through_pipeline(
    cases: &[GoldenCase],
    ollama_url: &str,
    micro_batch: usize,
) -> Vec<(String, String, GoldenExpectation)> {
    let cfg = Config {
        ollama_url: ollama_url.to_string(),
        micro_batch,
        concurrency: 1,
        model: model_from_env(),
        // Keep golden runs hermetic: no label-library file is read or written.
        label_library: String::new(),
        ..Config::default()
    };
    let service = Arc::new(LabelService::new(&cfg));

    let transactions: Vec<transaction_labeller::model::Transaction> = cases
        .iter()
        .map(|c| serde_json::from_value(c.transaction.clone()).unwrap())
        .collect();

    let batch = service
        .label(transactions, "de".to_string())
        .await
        .expect("backend works");

    cases
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let expectation: GoldenExpectation =
                serde_json::from_value(c.expected.clone()).unwrap();
            (c.name.clone(), batch.results[i].label.clone(), expectation)
        })
        .collect()
}

/// Deterministic mode: pipeline correctness against the contract mock.
/// The mock's keyword map is the deterministic ground truth for this set;
/// any cross-item mis-mapping, ordering bug, or parsing regression fails here.
#[tokio::test]
async fn golden_pipeline_maps_positions_exactly() {
    let b = MockBehaviour {
        keyword_map: default_keyword_map(),
        ..MockBehaviour::default()
    };
    let m = spawn_with(b).await;
    let _ = m.system_prompts; // captured but not asserted here
    let cases = load_golden();
    let results = run_through_pipeline(&cases, &m.url(), 4).await;

    assert_eq!(results.len(), cases.len(), "one label per case");
    for (name, label, expected) in &results {
        let expected_label = expected.labels()[0].clone();
        assert_eq!(
            label, &expected_label,
            "case {name}: label mismatch (positional association broken?)"
        );
    }
}

/// Live eval against the real model. Run with:
/// `cargo test -- --ignored live_eval --nocapture`
///
/// The whole golden set goes through in one request (`micro_batch = 32`) so
/// the model applies its wording consistently. Each case lists semantically
/// acceptable labels; a produced label is correct if it matches any of them
/// (case/punctuation-insensitive). This checks that labels are *sensible and
/// on-topic*, not verbatim wording — wording is the model's choice.
#[tokio::test]
#[ignore = "requires running Ollama with the model pulled (GPU)"]
async fn live_eval() {
    let ollama_url =
        std::env::var("TL_OLLAMA_URL").unwrap_or_else(|_| "http://127.0.0.1:11434".to_string());
    let cases = load_golden();
    let results = run_through_pipeline(&cases, &ollama_url, 32).await;

    let mut correct = 0usize;
    let mut wrong: Vec<(String, Vec<String>, String)> = Vec::new();

    eprintln!("\n{:<28} {:<32} {:<28}", "case", "acceptable", "actual");
    for (name, label, expected) in &results {
        let acceptable = expected.labels();
        let ok = acceptable.iter().any(|e| labels_match(label, e));
        if ok {
            correct += 1;
        } else {
            wrong.push((name.clone(), acceptable.clone(), label.clone()));
        }
        eprintln!(
            "{:<28} {:<32} {:<28} {}",
            name,
            acceptable.join(" / "),
            label,
            if ok { "ok" } else { "MISMATCH" }
        );
    }

    let n = results.len();
    let accuracy = correct as f64 / n as f64;
    eprintln!("\nlabel accuracy: {correct}/{n} = {accuracy:.2}");

    if !wrong.is_empty() {
        eprintln!("\nmismatches (case, acceptable → actual):");
        for (n, e, a) in &wrong {
            eprintln!("  {n}: {:?} → {:?}", e.first(), a);
        }
    }

    // Quality bar: labels are free-form; the acceptable-set covers reasonable
    // wording. Below this threshold the model is mislabelling, not rewording.
    assert!(
        accuracy >= 0.8,
        "live eval label accuracy {accuracy:.2} below 0.8 threshold"
    );
}

/// Compares a produced label against an acceptable one, tolerating case,
/// trailing punctuation and hyphen/space differences (dynamic labels are
/// free-form; German compound variants are covered by the acceptable set).
fn labels_match(actual: &str, expected: &str) -> bool {
    let norm = |s: &str| {
        s.trim()
            .to_lowercase()
            .trim_end_matches(['.', '!', '?'])
            .replace('-', " ")
    };
    norm(actual) == norm(expected)
}
