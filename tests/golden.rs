//! Golden result tests (correctness of results through the pipeline).
//!
//! The golden set (`tests/golden/cases.json`) contains real-world-style
//! transactions with expected canonical categories. Two modes:
//!
//! 1. **Deterministic mode (default, `cargo test`)**: runs the golden set
//!    through the full pipeline against the keyword-mapping mock LLM. This
//!    verifies the prompt → validation → mapping path preserves the contract
//!    (positional order, direction checks, localized labels).
//! 2. **Live eval (`cargo test -- --ignored live_eval`)**: same set against a
//!    real Ollama model with temperature 0; asserts macro accuracy ≥ 0.8 and
//!    reports per-category recall + confusion pairs. Requires a running
//!    Ollama with the model pulled; skipped by default so CI needs no GPU.

mod mock_llm;

use mock_llm::{default_keyword_map, spawn_with, MockBehaviour};
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
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
    category: String,
    direction: String,
    #[serde(default)]
    #[allow(dead_code)]
    zero_amount: bool,
}

fn load_golden() -> Vec<GoldenCase> {
    let raw = include_str!("golden/cases.json");
    serde_json::from_str(raw).expect("golden cases must parse")
}

async fn run_through_pipeline(
    cases: &[GoldenCase],
    ollama_url: &str,
) -> Vec<(String, Value, GoldenExpectation)> {
    let cfg = Config {
        ollama_url: ollama_url.to_string(),
        micro_batch: 4,
        concurrency: 2,
        ..Config::default()
    };
    let taxonomy = transaction_labeller::taxonomy::Taxonomy::load(None).unwrap();
    let service = Arc::new(LabelService::new(&cfg, taxonomy));

    let transactions: Vec<transaction_labeller::model::Transaction> = cases
        .iter()
        .map(|c| serde_json::from_value(c.transaction.clone()).unwrap())
        .collect();

    let batch = service
        .label(transactions, "de".to_string(), false)
        .await
        .expect("mock backend works");

    cases
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let expectation: GoldenExpectation =
                serde_json::from_value(c.expected.clone()).unwrap();
            let result = serde_json::to_value(&batch.results[i]).unwrap();
            (c.name.clone(), result, expectation)
        })
        .collect()
}

/// Deterministic mode: pipeline correctness against the contract mock.
#[tokio::test]
async fn golden_pipeline_preserves_contract() {
    let b = MockBehaviour {
        keyword_map: default_keyword_map(),
        ..MockBehaviour::default()
    };
    let m = spawn_with(b).await;
    let cases = load_golden();
    let results = run_through_pipeline(&cases, &m.url()).await;
    let taxonomy = transaction_labeller::taxonomy::builtin();

    for (name, result, expected) in &results {
        assert_eq!(
            result["direction"], expected.direction,
            "case {name}: direction mismatch"
        );
        // Exact category match: the mock's keyword map is the deterministic
        // ground truth for this set. Any cross-item mis-mapping, ordering
        // bug, or validation regression fails here.
        assert_eq!(
            result["category"], expected.category,
            "case {name}: category mismatch (positional association broken?)"
        );
        let slug = result["category"].as_str().unwrap();
        let cat = taxonomy
            .lookup(slug)
            .unwrap_or_else(|| panic!("case {name}: slug {slug} must exist in taxonomy"));
        assert_eq!(
            cat.direction.to_string(),
            expected.direction,
            "case {name}: slug {slug} direction inconsistent"
        );
        assert_eq!(result["status"], "ok", "case {name}: must not fall back");
        assert!(
            result["category_label"].is_string(),
            "case {name}: localized label required"
        );
    }
}

/// Live eval against the real model. Run with:
/// `cargo test -- --ignored live_eval --nocapture`
#[tokio::test]
#[ignore = "requires running Ollama with the model pulled (GPU)"]
async fn live_eval() {
    let ollama_url =
        std::env::var("TL_OLLAMA_URL").unwrap_or_else(|_| "http://127.0.0.1:11434".to_string());
    let cases = load_golden();
    let results = run_through_pipeline(&cases, &ollama_url).await;

    let mut correct = 0usize;
    let mut per_category: HashMap<String, (usize, usize)> = HashMap::new(); // (correct, total)
    let mut confusions: Vec<(String, String)> = Vec::new();

    eprintln!("\n{:<28} {:<22} {:<22}", "case", "expected", "actual");
    for (name, result, expected) in &results {
        let actual = result["category"].as_str().unwrap_or("<none>");
        let ok = actual == expected.category;
        if ok {
            correct += 1;
        } else {
            confusions.push((expected.category.clone(), actual.to_string()));
        }
        let entry = per_category
            .entry(expected.category.clone())
            .or_insert((0, 0));
        entry.1 += 1;
        if ok {
            entry.0 += 1;
        }
        eprintln!(
            "{:<28} {:<22} {:<22} {}",
            name,
            expected.category,
            actual,
            if ok { "✓" } else { "✗" }
        );
    }

    let n = results.len();
    let accuracy = correct as f64 / n as f64;
    eprintln!("\nmacro accuracy: {correct}/{n} = {accuracy:.2}");

    eprintln!("\nper-category recall:");
    for (cat, (c, t)) in &per_category {
        eprintln!("  {cat:<28} {c}/{t}");
    }
    if !confusions.is_empty() {
        eprintln!("\nconfusions (expected → actual):");
        for (e, a) in &confusions {
            eprintln!("  {e} → {a}");
        }
    }

    // Minimum quality bar (see docs/design.md §Testing).
    assert!(
        accuracy >= 0.8,
        "live eval macro accuracy {accuracy:.2} below 0.8 threshold"
    );
}
