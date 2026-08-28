use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use tracing::{debug, info, warn};

/// Super-simple label library: a JSON file mapping language → label → usage
/// count. The library is injected into the system prompt so the model reuses
/// existing wording instead of inventing variants; labels actually used are
/// recorded back (count incremented, new labels added).
///
/// File format:
/// ```json
/// {"de": {"Lebensmittel": 12, "Miete": 3}, "en": {"Groceries": 1}}
/// ```
#[derive(Debug)]
pub struct LabelLibrary {
    path: PathBuf,
    max_in_prompt: usize,
    state: Mutex<LibraryState>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct LibraryState {
    /// language → label → usage count.
    #[serde(flatten)]
    languages: HashMap<String, HashMap<String, u64>>,
}

/// Parses the library file, tolerating per-entry corruption: a malformed
/// count or a non-object language entry is skipped (with a warning) instead
/// of discarding the whole library — one bad hand-edit must not destroy the
/// previously valid data. Returns `None` only when the file as a whole is
/// not a JSON object, which is treated as full corruption by the caller.
fn parse_library_state(text: &str) -> Option<LibraryState> {
    let top: serde_json::Map<String, serde_json::Value> = serde_json::from_str(text).ok()?;
    let mut state = LibraryState::default();
    let mut skipped = 0usize;
    for (lang, value) in top {
        let Ok(counts) =
            serde_json::from_value::<serde_json::Map<String, serde_json::Value>>(value)
        else {
            warn!(language = %lang, "label library: skipping non-object language entry");
            skipped += 1;
            continue;
        };
        for (label, count) in counts {
            match serde_json::from_value::<u64>(count) {
                Ok(n) => {
                    state
                        .languages
                        .entry(lang.clone())
                        .or_default()
                        .insert(label, n);
                }
                Err(_) => skipped += 1,
            }
        }
    }
    if skipped > 0 {
        warn!(skipped, "label library: skipped malformed entries");
    }
    Some(state)
}

impl LabelLibrary {
    /// Opens (or creates) the library at `path`. A missing file is fine (start
    /// empty); a corrupt file disables the library with a warning rather than
    /// failing startup — labelling must keep working. Individual malformed
    /// entries (a bad count, a non-object language) are skipped so a single
    /// bad hand-edit cannot discard the rest of the library.
    pub fn open(path: PathBuf, max_in_prompt: usize) -> Self {
        let state = match std::fs::read_to_string(&path) {
            Ok(text) => match parse_library_state(&text) {
                Some(s) => s,
                None => {
                    warn!(path = %path.display(), "label library file corrupt; starting empty");
                    LibraryState::default()
                }
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => LibraryState::default(),
            Err(e) => {
                warn!(path = %path.display(), error = %e, "label library unreadable; starting empty");
                LibraryState::default()
            }
        };
        let len: usize = state.languages.values().map(|m| m.len()).sum();
        info!(path = %path.display(), labels = len, "label library loaded");
        Self {
            path,
            max_in_prompt,
            state: Mutex::new(state),
        }
    }

    /// An empty, in-memory-only library (tests).
    #[cfg(test)]
    pub fn disabled() -> Self {
        Self {
            path: PathBuf::new(),
            max_in_prompt: 0,
            state: Mutex::new(LibraryState::default()),
        }
    }

    /// Existing labels for `language`, most-used first, capped at
    /// `max_in_prompt`. Empty when the library is disabled
    /// (`TL_LABEL_LIBRARY=""` or `TL_LIBRARY_PROMPT_MAX=0`).
    pub fn labels_for(&self, language: &str) -> Vec<String> {
        if self.max_in_prompt == 0 {
            return Vec::new();
        }
        let state = self.state.lock().expect("library mutex");
        let mut labels: Vec<(u64, &String)> = state
            .languages
            .get(language)
            .map(|m| m.iter().map(|(l, c)| (*c, l)).collect())
            .unwrap_or_default();
        labels.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(b.1)));
        labels
            .into_iter()
            .map(|(_, l)| l.clone())
            .take(self.max_in_prompt)
            .collect()
    }

    /// Records labels that were actually returned for `language`: increments
    /// the usage count of known labels, inserts new ones. Persists the file;
    /// persistence errors are logged (in-memory state stays correct) and
    /// never fail the labelling request. No-op when the library is disabled
    /// (`max_in_prompt == 0`), so `TL_LIBRARY_PROMPT_MAX=0` disables the
    /// library entirely, like `TL_LABEL_LIBRARY=""`.
    pub fn record(&self, language: &str, labels: &[String]) {
        if self.max_in_prompt == 0 || labels.is_empty() {
            return;
        }
        let mut state = self.state.lock().expect("library mutex");
        let entry = state.languages.entry(language.to_string()).or_default();
        for label in labels {
            *entry.entry(label.clone()).or_insert(0) += 1;
        }
        drop(state);
        self.persist(language);
    }

    fn persist(&self, language: &str) {
        if self.path.as_os_str().is_empty() {
            return;
        }
        let state = self.state.lock().expect("library mutex");
        // Write via temp file + rename so a crash never truncates the file.
        let tmp = self.path.with_extension("json.tmp");
        let write = |to: &std::path::Path| -> std::io::Result<()> {
            let text = serde_json::to_string_pretty(&*state)
                .map_err(|e| std::io::Error::other(e.to_string()))?;
            std::fs::write(to, text)
        };
        // `std::fs::rename` replaces an existing destination on all platforms
        // (Windows uses MoveFileExW with MOVEFILE_REPLACE_EXISTING), so the
        // write path stays atomic everywhere.
        let result = write(&tmp).and_then(|()| std::fs::rename(&tmp, &self.path));
        if let Err(e) = result {
            let _ = std::fs::remove_file(&tmp);
            warn!(path = %self.path.display(), error = %e, "failed to persist label library");
            return;
        }
        debug!(language, path = %self.path.display(), "label library persisted");
    }

    /// Persisted once more on shutdown (best effort).
    #[allow(dead_code)] // called from main; not exercised by unit tests
    pub fn flush(&self) {
        if self.max_in_prompt != 0 && !self.path.as_os_str().is_empty() {
            self.persist("shutdown");
        }
    }

    #[allow(dead_code)] // used by unit tests
    pub fn max_in_prompt(&self) -> usize {
        self.max_in_prompt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_path(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "tl-lib-test-{tag}-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn open_missing_file_starts_empty() {
        let path = tmp_path("missing");
        let lib = LabelLibrary::open(path.clone(), 100);
        assert!(lib.labels_for("de").is_empty());
        assert!(lib.max_in_prompt() > 0);
    }

    #[test]
    fn record_adds_and_counts() {
        let lib = LabelLibrary::open(PathBuf::new(), 100);
        lib.record(
            "de",
            &["Lebensmittel".into(), "Miete".into(), "Lebensmittel".into()],
        );
        assert_eq!(lib.labels_for("de"), vec!["Lebensmittel", "Miete"]);
        lib.record("de", &["Lebensmittel".into()]);
        assert_eq!(lib.labels_for("de"), vec!["Lebensmittel", "Miete"]);
        assert!(lib.labels_for("en").is_empty(), "languages are separate");
    }

    #[test]
    fn labels_for_sorts_by_usage_then_name() {
        let lib = LabelLibrary::open(PathBuf::new(), 100);
        lib.record("de", &["B".into(), "A".into(), "C".into()]);
        lib.record("de", &["B".into(), "B".into()]);
        assert_eq!(lib.labels_for("de"), vec!["B", "A", "C"]);
        lib.record("de", &["B".into(), "A".into(), "C".into()]);
        // B: 5, A: 2, C: 2 — A before C by name on the tie.
        assert_eq!(lib.labels_for("de"), vec!["B", "A", "C"]);
    }

    #[test]
    fn open_loads_existing_file_and_persists_updates() {
        let path = tmp_path("persist");
        std::fs::write(&path, r#"{"de": {"Miete": 5}, "en": {"Groceries": 1}}"#).unwrap();
        let lib = LabelLibrary::open(path.clone(), 100);
        assert_eq!(lib.labels_for("de"), vec!["Miete"]);
        assert_eq!(lib.labels_for("en"), vec!["Groceries"]);

        lib.record("de", &["Miete".into(), "Lebensmittel".into()]);
        let text = std::fs::read_to_string(&path).unwrap();
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["de"]["Miete"], 6);
        assert_eq!(v["de"]["Lebensmittel"], 1);
        assert_eq!(v["en"]["Groceries"], 1);
    }

    #[test]
    fn corrupt_file_starts_empty_not_dead() {
        let path = tmp_path("corrupt");
        std::fs::write(&path, "{not json").unwrap();
        let lib = LabelLibrary::open(path.clone(), 100);
        assert!(lib.labels_for("de").is_empty());
        // A later record still persists over the corrupt file.
        lib.record("de", &["Miete".into()]);
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(v["de"]["Miete"], 1);
    }

    #[test]
    fn malformed_entry_is_skipped_rest_survives() {
        let path = tmp_path("malformed-entry");
        // Negative count and a non-object language entry are both invalid,
        // but the valid entries must survive the load.
        std::fs::write(
            &path,
            r#"{"de": {"Miete": 5, "Bad": -1}, "en": "not an object", "fr": {"Pain": 2}}"#,
        )
        .unwrap();
        let lib = LabelLibrary::open(path.clone(), 100);
        assert_eq!(lib.labels_for("de"), vec!["Miete"]);
        assert_eq!(lib.labels_for("fr"), vec!["Pain"]);
        assert!(lib.labels_for("en").is_empty());

        // The next persist keeps the surviving data (does not blank the file).
        lib.record("fr", &["Pain".into()]);
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(v["de"]["Miete"], 5);
        assert_eq!(v["fr"]["Pain"], 3);
        assert!(v.get("en").is_none());
    }

    #[test]
    fn disabled_library_returns_nothing_and_records_nothing() {
        let lib = LabelLibrary::disabled();
        lib.record("de", &["Miete".into()]);
        assert!(lib.labels_for("de").is_empty());
        assert!(lib.max_in_prompt() == 0);
    }

    #[test]
    fn cap_limits_prompt_list() {
        let lib = LabelLibrary::open(PathBuf::new(), 3);
        for i in 0..10 {
            lib.record("de", &[format!("Label{i}")]);
        }
        assert_eq!(lib.labels_for("de").len(), 3);
    }
}
