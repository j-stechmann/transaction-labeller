use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// A single banking transaction to be labelled. `amount` must be finite
/// (NaN and ±Infinity rejected at deserialization).
#[derive(Debug, Clone, Deserialize, ToSchema)]
pub struct Transaction {
    /// Unique identifier within the request; results are returned in input
    /// order, so batch clients can associate by position.
    #[schema(example = "tx-1", max_length = 512)]
    pub id: String,
    /// Payee (outgoing) or payer (incoming) name.
    #[serde(default)]
    #[schema(example = "REWE SAGT DANKE", max_length = 512)]
    pub counterparty: String,
    /// Purpose / reference line from the statement.
    #[serde(default)]
    #[schema(example = "Einkauf 14.02", max_length = 512)]
    pub purpose: String,
    /// Signed amount; negative = outflow, positive = inflow.
    /// Must be finite.
    #[serde(deserialize_with = "deserialize_finite")]
    #[schema(example = -42.13)]
    pub amount: f64,
    /// ISO 4217 currency code (advisory; shown to the model).
    #[serde(default = "default_currency")]
    #[schema(example = "EUR", max_length = 8)]
    pub currency: String,
    /// Booking date, ISO 8601 recommended.
    #[serde(default)]
    #[schema(example = "2026-02-14", max_length = 512)]
    pub date: String,
}

fn deserialize_finite<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let v = f64::deserialize(deserializer)?;
    if !v.is_finite() {
        return Err(serde::de::Error::custom("amount must be finite"));
    }
    Ok(v)
}

fn default_currency() -> String {
    "EUR".to_string()
}

/// Per-request options; language overrides the server default.
#[derive(Debug, Clone, Default, Deserialize, ToSchema)]
pub struct LabelOptions {
    /// ISO 639-1 label language (`de`, `en`, …). Overrides `TL_LANGUAGE`.
    #[schema(example = "de")]
    pub language: Option<String>,
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

/// The label, with the id it belongs to.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct SingleLabelResponse {
    /// Echoes the input transaction id.
    #[schema(example = "tx-1")]
    pub id: String,
    /// LLM-generated category name in the requested language.
    #[schema(example = "Lebensmittel")]
    pub label: String,
}

/// One result per input transaction, in input order (`results[i]` ↔
/// request transaction `i`); `id` makes association explicit.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct BatchLabelResponse {
    #[schema(example = json!([{"id": "a", "label": "Lebensmittel"}, {"id": "b", "label": "Miete"}]))]
    pub results: Vec<LabeledTransaction>,
}

/// A transaction id paired with its LLM-generated label.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct LabeledTransaction {
    /// Echoes the input transaction id.
    #[schema(example = "tx-1")]
    pub id: String,
    /// LLM-generated category name in the requested language.
    #[schema(example = "Lebensmittel")]
    pub label: String,
}

/// Uniform error body: `{ "error": { "code", "message", "details" } }`.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ApiError {
    pub error: ErrorBody,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ErrorBody {
    /// Machine-readable error code (`invalid_request`, `backend_unavailable`).
    #[schema(example = "invalid_request")]
    pub code: String,
    /// Human-readable description.
    #[schema(example = "duplicate transaction id \"tx-1\"; ids must be unique")]
    pub message: String,
    /// Optional extra context (e.g. offending indices).
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

    pub fn invalid_request(msg: impl Into<String>) -> Self {
        Self::new("invalid_request", msg)
    }

    pub fn backend_unavailable(msg: impl Into<String>) -> Self {
        Self::new("backend_unavailable", msg)
    }
}

/// `POST /v1/label` request body.
#[derive(Debug, Deserialize, ToSchema)]
pub struct LabelSingleRequest {
    pub transaction: Transaction,
    #[serde(default)]
    pub options: LabelOptions,
}

/// `POST /v1/label:batch` request body.
#[derive(Debug, Deserialize, ToSchema)]
pub struct LabelBatchRequest {
    /// 1..=`TL_MAX_BATCH` (default 100) transactions; ids must be unique.
    pub transactions: Vec<Transaction>,
    #[serde(default)]
    pub options: LabelOptions,
}

/// `GET /v1/labels` response — the label library for one language.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct LabelListResponse {
    /// Known labels, most-used first. The model is instructed to reuse these
    /// verbatim; new labels are appended automatically as they are used.
    #[schema(example = json!(["Lebensmittel", "Miete", "Einkommen"]))]
    pub labels: Vec<String>,
}

/// `GET /v1/health` response.
#[derive(Debug, Serialize, ToSchema)]
pub struct HealthResponse {
    /// `ok` or `degraded`.
    #[schema(example = "ok")]
    pub status: String,
    /// Backend reachability description.
    #[schema(example = "reachable")]
    pub backend: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn language_precedence_request_over_default() {
        let opts = LabelOptions {
            language: Some("EN".to_string()),
        };
        assert_eq!(opts.effective_language("de"), "en");

        let opts_none = LabelOptions::default();
        assert_eq!(opts_none.effective_language("de"), "de");

        let opts_blank = LabelOptions {
            language: Some("  ".to_string()),
        };
        assert_eq!(opts_blank.effective_language("de"), "de");
    }

    #[test]
    fn transaction_rejects_nonfinite_amount() {
        let body = r#"{"id":"a","amount":NaN}"#;
        let res: Result<Transaction, _> = serde_json::from_str(body);
        assert!(res.is_err(), "NaN must be rejected");

        // 1e999 overflows to inf in serde_json's arbitrary precision path
        let body = r#"{"id":"a","amount":1e999}"#;
        let res: Result<Transaction, _> = serde_json::from_str(body);
        assert!(res.is_err(), "Infinity must be rejected");

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
