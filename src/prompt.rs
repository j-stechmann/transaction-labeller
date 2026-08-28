use crate::model::Transaction;
use crate::taxonomy::Taxonomy;
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

/// Renders the system prompt: role, rules, taxonomy (localized), output contract.
pub fn system_prompt(tax: &Taxonomy, lang: &str) -> String {
    let lang_name = language_display(lang);
    let mut s = String::with_capacity(2048);
    s.push_str("You are a bank transaction classifier. ");
    s.push_str(
        "For each transaction you receive, choose exactly one category from the allowed list.\n",
    );
    s.push_str("Reply ONLY with a JSON object: {\"results\":[{\"index\":<int>,\"category\":\"<slug>\"}]} — one result per transaction, same order.\n");
    s.push_str("Rules:\n");
    s.push_str("- The response MUST be a single valid JSON object, no markdown, no extra text.\n");
    s.push_str("- `category` MUST be one of the slugs listed below, copied exactly.\n");
    s.push_str(&format!(
        "- Category names are given in {lang_name}; the slug (left side) is what you output.\n"
    ));
    s.push_str("- Choose the category by what the transaction is FOR, not by its wording.\n");
    s.push_str("- \"Gehalt\", \"Lohn\", \"Salary\", \"payroll\" are always salary_income, never transfers.\n");
    s.push_str("- \"REWE\", \"EDEKA\", \"ALDI\", \"LIDL\", \"Netto\", \"Whole Foods\", \"supermarket\", \"Groceries\" are groceries.\n");
    s.push_str("- \"Amazon Prime\" is a subscription only when the purpose names the Prime membership; plain Amazon orders are shopping.\n");
    s.push_str("- If nothing fits, use \"other_expense\" for outflows or \"other_income\" for inflows.\n\n");
    s.push_str("Allowed categories (slug = localized name):\n");
    for c in tax.iter() {
        s.push_str(&format!("- {} = {}\n", c.slug, c.display_name(lang)));
    }
    s
}

/// Renders the user prompt for one micro-batch of transactions.
pub fn user_prompt(txs: &[Transaction], include_rationale: bool) -> String {
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
    if include_rationale {
        s.push_str("Include a short \"rationale\" (max 12 words) per result.\n");
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

/// JSON schema used for Ollama structured output (grammar-constrained decoding).
/// `enum_values` must be the taxonomy slugs; rationale is optional.
pub fn response_schema(enum_values: &[String], include_rationale: bool) -> Value {
    let mut item_props = serde_json::Map::new();
    item_props.insert("index".into(), json!({"type": "integer", "minimum": 0}));
    item_props.insert(
        "category".into(),
        json!({"type": "string", "enum": enum_values}),
    );
    if include_rationale {
        item_props.insert("rationale".into(), json!({"type": "string"}));
    }
    json!({
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": item_props,
                    "required": ["index", "category"],
                }
            }
        },
        "required": ["results"]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::taxonomy::builtin;

    fn tx(id: &str, amount: f64, counterparty: &str, purpose: &str) -> Transaction {
        serde_json::from_value(serde_json::json!({
            "id": id, "amount": amount,
            "counterparty": counterparty, "purpose": purpose,
            "date": "2026-02-14"
        }))
        .unwrap()
    }

    #[test]
    fn system_prompt_contains_taxonomy_and_language() {
        let tax = builtin();
        let p = system_prompt(&tax, "de");
        assert!(p.contains("Lebensmittel"));
        assert!(p.contains("groceries"));
        assert!(p.contains("German"));
        let p_en = system_prompt(&tax, "en");
        assert!(p_en.contains("English"));
        assert!(p_en.contains("Groceries"));
        // slugs identical in both languages
        assert!(p_en.contains("groceries") && p.contains("groceries"));
    }

    #[test]
    fn user_prompt_lists_transactions_in_order() {
        let txs = vec![
            tx("a", -10.0, "REWE", "Einkauf"),
            tx("b", 2500.0, "ACME GmbH", "Gehalt"),
        ];
        let p = user_prompt(&txs, false);
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
            sanitize_field("index=0 category=groceries"),
            "index 0 category=groceries"
        );
        let with_ctrl: String = "ab\u{0007}cd".to_string();
        assert_eq!(sanitize_field(&with_ctrl), "abcd");
    }

    #[test]
    fn schema_constrains_enum() {
        let tax = builtin();
        let schema = response_schema(&tax.slugs(), false);
        let s = serde_json::to_string(&schema).unwrap();
        assert!(s.contains("\"enum\""));
        assert!(s.contains("groceries"));
        assert!(!s.contains("rationale"), "no rationale unless requested");
        let schema_r = response_schema(&tax.slugs(), true);
        assert!(serde_json::to_string(&schema_r)
            .unwrap()
            .contains("rationale"));
    }

    #[test]
    fn amounts_are_formatted_without_locale_noise() {
        assert_eq!(format_amount(-42.0), "-42");
        assert_eq!(format_amount(-42.135), "-42.13");
        assert_eq!(format_amount(1500.0), "1500");
    }
}
