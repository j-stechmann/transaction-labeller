use crate::model::Transaction;
use serde_json::{json, Value};

pub const LANGUAGE_NAMES: &[(&str, &str)] = &[
    ("de", "German"),
    ("en", "English"),
    ("fr", "French"),
    ("es", "Spanish"),
    ("it", "Italian"),
    ("nl", "Dutch"),
    ("pt", "Portuguese"),
    ("sv", "Swedish"),
    ("da", "Danish"),
    ("pl", "Polish"),
    ("cs", "Czech"),
    ("tr", "Turkish"),
];

pub fn language_display(lang: &str) -> &str {
    LANGUAGE_NAMES
        .iter()
        .find(|(code, _)| *code == lang)
        .map(|(_, name)| *name)
        .unwrap_or("English")
}

/// Renders the system prompt: role, rules, output contract, and — when the
/// label library has entries for this language — the existing-label list the
/// model must prefer. Labels are otherwise generated dynamically by the
/// model; there is no fixed taxonomy.
pub fn system_prompt(lang: &str, library_labels: &[String]) -> String {
    let lang_name = language_display(lang);
    let mut s = String::with_capacity(1536);
    s.push_str("You are a bank transaction classifier. ");
    s.push_str("For each transaction you receive, invent one short category label that best describes what the transaction is for.\n");
    s.push_str("Reply ONLY with a JSON object: {\"results\":[{\"index\":<int>,\"label\":\"<category name>\"}]} — one result per transaction, same order.\n");
    s.push_str("Rules:\n");
    s.push_str("- The response MUST be a single valid JSON object, no markdown, no extra text.\n");
    s.push_str(&format!(
        "- IMPORTANT: write the label in {lang_name}. The whole label must be in {lang_name}.\n"
    ));
    s.push_str("- Keep labels short: 1–3 words, no punctuation, sentence case.\n");
    s.push_str("- Reuse the same wording for the same kind of transaction (consistency within the batch matters).\n");
    s.push_str("- Choose the category by what the transaction is FOR, not by its wording.\n");
    s.push_str("- If nothing specific fits, use a generic label for outflows and a generic label for inflows.\n");
    if !library_labels.is_empty() {
        s.push_str("\nExisting category labels already in use");
        if lang_name != "English" {
            s.push_str(&format!(" (in {lang_name})"));
        }
        s.push_str(":\n");
        for label in library_labels {
            // Labels are one-per-line; strip newlines defensively even though
            // the sanitizer already caps content.
            let clean = label.replace(['\n', '\r'], " ");
            s.push_str(&format!("- {clean}\n"));
        }
        s.push_str("\nRULES FOR EXISTING LABELS:\n");
        s.push_str("- If one of these labels fits a transaction, you MUST use it EXACTLY as written (character-for-character).\n");
        s.push_str("- Only invent a new label when none of the existing labels fits. New labels are added to the list automatically.\n");
    }
    s
}

/// Renders the user prompt for one micro-batch of transactions.
pub fn user_prompt(txs: &[Transaction]) -> String {
    let mut s = String::with_capacity(256 * txs.len());
    s.push_str("Classify these transactions:\n");
    for (i, tx) in txs.iter().enumerate() {
        s.push_str(&format!(
            "[{}] date={}; amount={}; currency={}; counterparty=<<{}>>; purpose=<<{}>>\n",
            i,
            sanitize_field(&tx.date),
            format_amount(tx.amount),
            sanitize_field(&tx.currency),
            sanitize_field(&tx.counterparty),
            sanitize_field(&tx.purpose),
        ));
    }
    s
}

/// Strips control chars and prompt-structure markers from model input fields.
fn sanitize_field(raw: &str) -> String {
    raw.chars()
        .filter(|c| !c.is_control())
        .collect::<String>()
        .replace("<<", "<")
        .replace(">>", ">")
        .replace("index=", "index ")
}

fn format_amount(a: f64) -> String {
    if a.fract() == 0.0 && a.abs() < 1e15 {
        format!("{a:.0}")
    } else {
        format!("{a:.2}")
    }
}

/// JSON schema used for Ollama structured output (grammar-constrained
/// decoding). The label is a free string — the model invents it; length is
/// bounded to keep responses compact.
pub fn response_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer", "minimum": 0},
                        "label": {"type": "string", "minLength": 1, "maxLength": 64}
                    },
                    "required": ["index", "label"]
                }
            }
        },
        "required": ["results"]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tx(id: &str, amount: f64, counterparty: &str, purpose: &str) -> Transaction {
        serde_json::from_value(serde_json::json!({
            "id": id, "amount": amount,
            "counterparty": counterparty, "purpose": purpose,
            "date": "2026-02-14"
        }))
        .unwrap()
    }

    #[test]
    fn system_prompt_asks_for_language_and_dynamic_labels() {
        let p = system_prompt("de", &[]);
        assert!(p.contains("German"));
        assert!(p.contains("label"));
        assert!(!p.contains("slug"), "no taxonomy slugs");
        assert!(!p.contains("Allowed categories"), "no fixed category list");
        let p_en = system_prompt("en", &[]);
        assert!(p_en.contains("English"));
    }

    #[test]
    fn system_prompt_injects_library_labels_verbatim() {
        let labels = vec!["Lebensmittel".to_string(), "Miete".to_string()];
        let p = system_prompt("de", &labels);
        assert!(p.contains("Lebensmittel"));
        assert!(p.contains("Miete"));
        assert!(p.contains("EXACTLY"), "must instruct verbatim reuse");
        assert!(!p.contains("Allowed categories"));
    }

    #[test]
    fn empty_library_omits_library_section() {
        let p = system_prompt("de", &[]);
        assert!(!p.contains("Existing category labels"));
        assert!(!p.contains("EXACTLY"));
    }

    #[test]
    fn user_prompt_lists_transactions_in_order() {
        let txs = vec![
            tx("a", -10.0, "REWE", "Einkauf"),
            tx("b", 2500.0, "ACME GmbH", "Gehalt"),
        ];
        let p = user_prompt(&txs);
        assert!(p.contains("[0]"));
        assert!(p.contains("[1]"));
        assert!(p.contains("REWE"));
        assert!(p.contains("Gehalt"));
        assert!(!p.contains("rationale"));
    }

    #[test]
    fn sanitize_field_strips_injection_markers() {
        assert_eq!(sanitize_field("a<<b>>c"), "a<b>c");
        assert_eq!(
            sanitize_field("index=0 label=groceries"),
            "index 0 label=groceries"
        );
        let with_ctrl: String = "ab\u{0007}cd".to_string();
        assert_eq!(sanitize_field(&with_ctrl), "abcd");
    }

    #[test]
    fn schema_has_no_enum_and_bounds_label_length() {
        let schema = response_schema();
        let s = serde_json::to_string(&schema).unwrap();
        assert!(!s.contains("\"enum\""), "labels are dynamic, no enum");
        assert!(s.contains("maxLength"));
        assert!(
            !s.contains("rationale"),
            "no rationale: label-only response"
        );
    }

    #[test]
    fn amounts_are_formatted_without_locale_noise() {
        assert_eq!(format_amount(-42.0), "-42");
        assert_eq!(format_amount(-42.135), "-42.13");
        assert_eq!(format_amount(1500.0), "1500");
    }
}
