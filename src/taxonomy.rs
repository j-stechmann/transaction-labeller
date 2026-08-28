use crate::model::Direction;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

/// Canonical ASCII slug; stable across languages. Used as the model-facing
/// enum value and the API `category` identifier.
pub type Slug = String;

#[derive(Debug, Clone, Deserialize)]
struct RawCategory {
    slug: String,
    direction: Option<String>,
    /// Localized display names keyed by ISO 639-1 code.
    names: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct Category {
    pub slug: Slug,
    /// Taxonomic direction this category belongs to.
    pub direction: Direction,
    names: HashMap<String, String>,
}

impl Category {
    /// Display name for `lang`; falls back to the canonical (first) name and
    /// finally the slug itself.
    pub fn display_name(&self, lang: &str) -> &str {
        self.names
            .get(lang)
            .or_else(|| self.names.get("de"))
            .or_else(|| self.names.values().next())
            .map(|s| s.as_str())
            .unwrap_or(&self.slug)
    }
}

#[derive(Debug, Clone)]
pub struct Taxonomy {
    categories: Vec<Category>,
    by_slug: HashMap<Slug, usize>,
}

#[derive(Debug, Deserialize)]
struct RawTaxonomy {
    categories: Vec<RawCategory>,
}

#[derive(Debug)]
pub enum TaxonomyError {
    Io(std::io::Error),
    Parse(serde_json::Error),
    Invalid(String),
}

impl std::fmt::Display for TaxonomyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaxonomyError::Io(e) => write!(f, "taxonomy file read error: {e}"),
            TaxonomyError::Parse(e) => write!(f, "taxonomy JSON parse error: {e}"),
            TaxonomyError::Invalid(m) => write!(f, "invalid taxonomy: {m}"),
        }
    }
}

impl std::error::Error for TaxonomyError {}

/// Built-in default taxonomy (canonical display names in German, `en` provided).
/// Every category carries an explicit direction; `direction` defaults by naming
/// convention (`*_income` → income, otherwise expense) but should be stated.
pub fn builtin() -> Taxonomy {
    let json = r#"{
        "categories": [
            {"slug": "salary_income", "direction": "income", "names": {"de": "Einkommen", "en": "Salary & Wages"}},
            {"slug": "refund", "direction": "income", "names": {"de": "Erstattung", "en": "Refunds"}},
            {"slug": "transfer", "direction": "income", "names": {"de": "Übertragung", "en": "Transfer"}},
            {"slug": "other_income", "direction": "income", "names": {"de": "Sonstige Einnahmen", "en": "Other Income"}},

            {"slug": "housing", "direction": "expense", "names": {"de": "Wohnen", "en": "Housing"}},
            {"slug": "groceries", "direction": "expense", "names": {"de": "Lebensmittel", "en": "Groceries"}},
            {"slug": "dining", "direction": "expense", "names": {"de": "Restaurant & Café", "en": "Dining & Cafés"}},
            {"slug": "transport", "direction": "expense", "names": {"de": "Transport & Mobilität", "en": "Transport & Mobility"}},
            {"slug": "shopping", "direction": "expense", "names": {"de": "Shopping", "en": "Shopping"}},
            {"slug": "health", "direction": "expense", "names": {"de": "Gesundheit", "en": "Health"}},
            {"slug": "leisure", "direction": "expense", "names": {"de": "Freizeit & Unterhaltung", "en": "Leisure & Entertainment"}},
            {"slug": "subscriptions", "direction": "expense", "names": {"de": "Abos & Dienstleistungen", "en": "Subscriptions & Services"}},
            {"slug": "insurance", "direction": "expense", "names": {"de": "Versicherungen", "en": "Insurance"}},
            {"slug": "savings_investing", "direction": "expense", "names": {"de": "Sparen & Investieren", "en": "Savings & Investing"}},
            {"slug": "education", "direction": "expense", "names": {"de": "Bildung", "en": "Education"}},
            {"slug": "donations", "direction": "expense", "names": {"de": "Spenden", "en": "Donations"}},
            {"slug": "taxes_fees", "direction": "expense", "names": {"de": "Steuern & Gebühren", "en": "Taxes & Fees"}},
            {"slug": "cash_withdrawal", "direction": "expense", "names": {"de": "Bargeld", "en": "Cash Withdrawal"}},
            {"slug": "credit_card_settlement", "direction": "expense", "names": {"de": "Kreditkartenabrechnung", "en": "Credit Card Settlement"}},
            {"slug": "other_expense", "direction": "expense", "names": {"de": "Sonstige Ausgaben", "en": "Other Expenses"}}
        ]
    }"#;
    Taxonomy::from_str(json).expect("built-in taxonomy must be valid")
}

impl Taxonomy {
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(json: &str) -> Result<Self, TaxonomyError> {
        let raw: RawTaxonomy = serde_json::from_str(json).map_err(TaxonomyError::Parse)?;
        Self::build(raw.categories)
    }

    pub fn from_file(path: &Path) -> Result<Self, TaxonomyError> {
        let content = std::fs::read_to_string(path).map_err(TaxonomyError::Io)?;
        Self::from_str(&content)
    }

    pub fn load(path: Option<&Path>) -> Result<Self, TaxonomyError> {
        match path {
            Some(p) => Self::from_file(p),
            None => Ok(builtin()),
        }
    }

    fn build(raw: Vec<RawCategory>) -> Result<Self, TaxonomyError> {
        if raw.is_empty() {
            return Err(TaxonomyError::Invalid("no categories".into()));
        }
        let mut categories = Vec::with_capacity(raw.len());
        let mut by_slug = HashMap::new();
        for (i, rc) in raw.into_iter().enumerate() {
            let slug = rc.slug.trim().to_string();
            if slug.is_empty() {
                return Err(TaxonomyError::Invalid(format!("category {i}: empty slug")));
            }
            if !slug.chars().all(|c| c.is_ascii_lowercase() || c == '_') {
                return Err(TaxonomyError::Invalid(format!(
                    "category {i}: slug `{slug}` must be lowercase ASCII with underscores"
                )));
            }
            if by_slug.contains_key(&slug) {
                return Err(TaxonomyError::Invalid(format!(
                    "category {i}: duplicate slug `{slug}`"
                )));
            }
            if rc.names.is_empty() {
                return Err(TaxonomyError::Invalid(format!(
                    "category {i} (`{slug}`): no names provided"
                )));
            }
            let direction = match rc.direction.as_deref() {
                Some("income") => Direction::Income,
                Some("expense") => Direction::Expense,
                Some(other) => {
                    return Err(TaxonomyError::Invalid(format!(
                        "category {i} (`{slug}`): direction must be \"income\" or \"expense\", got {other:?}"
                    )))
                }
                None => {
                    // Default by naming convention: *_income → income.
                    if slug.ends_with("_income") {
                        Direction::Income
                    } else if slug.ends_with("_expense") {
                        Direction::Expense
                    } else {
                        return Err(TaxonomyError::Invalid(format!(
                            "category {i} (`{slug}`): missing `direction` (income|expense); \
                             it cannot be inferred from the slug"
                        )));
                    }
                }
            };
            by_slug.insert(slug.clone(), categories.len());
            categories.push(Category {
                slug,
                direction,
                names: rc.names,
            });
        }
        Ok(Self {
            categories,
            by_slug,
        })
    }

    pub fn len(&self) -> usize {
        self.categories.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.categories.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Category> {
        self.categories.iter()
    }

    /// Enum values for the JSON schema (canonical slugs).
    pub fn slugs(&self) -> Vec<String> {
        self.categories.iter().map(|c| c.slug.clone()).collect()
    }

    pub fn lookup(&self, slug: &str) -> Option<&Category> {
        self.by_slug.get(slug).map(|&i| &self.categories[i])
    }

    /// Case-insensitive slug match (the model may capitalise).
    pub fn lookup_ci(&self, slug: &str) -> Option<&Category> {
        let lowered = slug.trim().to_lowercase();
        self.lookup(&lowered)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_parses_and_has_no_duplicates() {
        let t = builtin();
        assert!(t.len() >= 20);
        let slugs = t.slugs();
        let mut sorted = slugs.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(slugs.len(), sorted.len(), "slugs must be unique");
        for s in &slugs {
            assert!(s.chars().all(|c| c.is_ascii_lowercase() || c == '_'));
        }
    }

    #[test]
    fn display_name_fallback_chain() {
        let t = builtin();
        let groceries = t.lookup("groceries").unwrap();
        assert_eq!(groceries.display_name("de"), "Lebensmittel");
        assert_eq!(groceries.display_name("en"), "Groceries");
        // Unknown language falls back to canonical (de)
        assert_eq!(groceries.display_name("fr"), "Lebensmittel");
    }

    #[test]
    fn lookup_ci_is_case_insensitive() {
        let t = builtin();
        assert!(t.lookup_ci("GROCERIES").is_some());
        assert!(t.lookup_ci(" Groceries ").is_some());
        assert!(t.lookup_ci("nonexistent_slug").is_none());
    }

    #[test]
    fn custom_taxonomy_from_json() {
        let json = r#"{
            "categories": [
                {"slug": "food", "direction": "expense", "names": {"en": "Food"}}
            ]
        }"#;
        let t = Taxonomy::from_str(json).unwrap();
        assert_eq!(t.len(), 1);
        assert_eq!(
            t.lookup("food").unwrap().display_name("de"),
            "Food",
            "falls back to the only provided name"
        );
        assert_eq!(
            t.lookup("food").unwrap().direction,
            crate::model::Direction::Expense
        );
    }

    #[test]
    fn missing_direction_rejected_unless_naming_convention() {
        // no direction, slug gives no hint → rejected
        let ambiguous = r#"{"categories":[{"slug":"food","names":{"en":"Food"}}]}"#;
        assert!(Taxonomy::from_str(ambiguous).is_err());
        // convention fallback: *_income
        let conventional = r#"{"categories":[{"slug":"my_income","names":{"en":"My Income"}}]}"#;
        let t = Taxonomy::from_str(conventional).unwrap();
        assert_eq!(
            t.lookup("my_income").unwrap().direction,
            crate::model::Direction::Income
        );
        // bad direction value → rejected
        let bad_dir =
            r#"{"categories":[{"slug":"food","direction":"sideways","names":{"en":"Food"}}]}"#;
        assert!(Taxonomy::from_str(bad_dir).is_err());
    }

    #[test]
    fn invalid_taxonomies_rejected() {
        // duplicate slug
        let dup = r#"{"categories":[
            {"slug":"a","direction":"expense","names":{"en":"A"}},
            {"slug":"a","direction":"expense","names":{"en":"B"}}]}"#;
        assert!(matches!(
            Taxonomy::from_str(dup),
            Err(TaxonomyError::Invalid(_))
        ));
        // bad slug characters (model-facing enum must stay ASCII)
        let bad = r#"{"categories":[
            {"slug":"Übertragung","direction":"expense","names":{"en":"T"}}]}"#;
        assert!(Taxonomy::from_str(bad).is_err());
        // empty
        assert!(Taxonomy::from_str(r#"{"categories":[]}"#).is_err());
        // no names
        let nonames = r#"{"categories":[{"slug":"a","direction":"expense","names":{}}]}"#;
        assert!(Taxonomy::from_str(nonames).is_err());
    }
}
