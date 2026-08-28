use serde::{Deserialize, Serialize};

/// Direction derived deterministically from the amount sign.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Direction {
    Income,
    Expense,
}

impl Direction {
    /// `amount == 0` defaults to expense unless overridden by the caller.
    pub fn from_amount(amount: f64, zero_is_income: bool) -> Self {
        if amount > 0.0 || (amount == 0.0 && zero_is_income) {
            Direction::Income
        } else {
            Direction::Expense
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Direction::Income => "income",
            Direction::Expense => "expense",
        }
    }
}

/// A single banking transaction to be labelled. `amount` must be finite
/// (NaN/Infinity rejected at deserialization).
#[derive(Debug, Clone, Deserialize)]
pub struct Transaction {
    pub id: String,
    #[serde(default)]
    pub counterparty: String,
    #[serde(default)]
    pub purpose: String,
    pub amount: f64,
    #[serde(default = "default_currency")]
    pub currency: String,
    #[serde(default)]
    pub date: String,
}

fn default_currency() -> String {
    "EUR".to_string()
}

/// Per-request options; language overrides the server default.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct LabelOptions {
    pub language: Option<String>,
    #[serde(default)]
    pub include_rationale: bool,
}

impl LabelOptions {
    pub fn effective_language(&self, server_default: &str) -> String {
        self.language
            .as_ref()
            .map(|l| l.trim().to_lowercase())
            .filter(|l| !l.is_empty())
            .unwrap_or_else(|| server_default.to_string())
    }
}

/// Per-item result status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ItemStatus {
    Ok,
    FallbackUnknown,
}

#[derive(Debug, Clone, Serialize)]
pub struct LabelResult {
    pub id: String,
    /// Canonical ASCII slug — stable across languages; clients key on this.
    pub category: String,
    /// Localized display name for the requested language.
    pub category_label: String,
    pub direction: Direction,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rationale: Option<String>,
    pub status: ItemStatus,
    pub model: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct BatchResponse {
    pub results: Vec<LabelResult>,
    pub batch_ms: u64,
}

/// Uniform error body: `{ "error": { "code", "message", "details" } }`.
#[derive(Debug, Clone, Serialize)]
pub struct ApiError {
    pub error: ErrorBody,
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorBody {
    pub code: String,
    pub message: String,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub details: Vec<String>,
}

impl ApiError {
    pub fn new(code: &str, message: impl Into<String>) -> Self {
        Self {
            error: ErrorBody {
                code: code.to_string(),
                message: message.into(),
                details: Vec::new(),
            },
        }
    }

    pub fn with_details(mut self, details: Vec<String>) -> Self {
        self.error.details = details;
        self
    }

    pub fn invalid_request(msg: impl Into<String>) -> Self {
        Self::new("invalid_request", msg)
    }

    pub fn backend_unavailable(msg: impl Into<String>) -> Self {
        Self::new("backend_unavailable", msg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direction_from_sign() {
        assert_eq!(Direction::from_amount(-42.13, false), Direction::Expense);
        assert_eq!(Direction::from_amount(1500.0, false), Direction::Income);
        assert_eq!(Direction::from_amount(0.0, false), Direction::Expense);
        assert_eq!(Direction::from_amount(0.0, true), Direction::Income);
        assert_eq!(Direction::from_amount(-0.0001, true), Direction::Expense);
    }

    #[test]
    fn language_precedence_request_over_default() {
        let opts = LabelOptions {
            language: Some("EN".to_string()),
            include_rationale: false,
        };
        assert_eq!(opts.effective_language("de"), "en");

        let opts_none = LabelOptions::default();
        assert_eq!(opts_none.effective_language("de"), "de");

        let opts_blank = LabelOptions {
            language: Some("  ".to_string()),
            include_rationale: false,
        };
        assert_eq!(opts_blank.effective_language("de"), "de");
    }

    #[test]
    fn transaction_rejects_nonfinite_amount() {
        let body = r#"{"id":"a","amount":NaN}"#;
        let res: Result<Transaction, _> = serde_json::from_str(body);
        assert!(res.is_err(), "NaN must be rejected");

        let body = r#"{"id":"a","amount":null}"#;
        let res: Result<Transaction, _> = serde_json::from_str(body);
        assert!(res.is_err());
    }

    #[test]
    fn transaction_defaults() {
        let body = r#"{"id":"a","amount":-5.5}"#;
        let tx: Transaction = serde_json::from_str(body).unwrap();
        assert_eq!(tx.currency, "EUR");
        assert_eq!(tx.counterparty, "");
        assert_eq!(tx.purpose, "");
    }
}