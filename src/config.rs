#[derive(Debug, Clone)]
pub struct Config {
    pub bind_addr: String,
    pub ollama_url: String,
    pub model: String,
    pub language: String,
    pub concurrency: usize,
    pub micro_batch: usize,
    pub num_ctx: u32,
    pub vram_budget_mb: u64,
    pub request_timeout_secs: u64,
    pub max_retries: u32,
    pub max_batch: usize,
    /// Path of the label-library JSON file; empty disables the library.
    pub label_library: String,
    /// Max library labels injected into a prompt (per language).
    pub library_prompt_max: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            bind_addr: "127.0.0.1:8080".to_string(),
            ollama_url: "http://127.0.0.1:11434".to_string(),
            model: "qwen3.8:27b-q4_K_M".to_string(),
            language: "de".to_string(),
            concurrency: 1,
            micro_batch: 16,
            num_ctx: 4096,
            vram_budget_mb: 8192,
            request_timeout_secs: 600,
            max_retries: 2,
            max_batch: 100,
            label_library: "labels.json".to_string(),
            library_prompt_max: 200,
        }
    }
}

#[derive(Debug)]
pub enum ConfigError {
    InvalidValue(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::InvalidValue(m) => write!(f, "invalid configuration: {m}"),
        }
    }
}

impl std::error::Error for ConfigError {}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        let mut cfg = Self::default();

        for (var, slot) in [
            ("TL_BIND_ADDR", 0),
            ("TL_OLLAMA_URL", 1),
            ("TL_MODEL", 2),
            ("TL_LANGUAGE", 3),
        ] {
            if let Ok(v) = std::env::var(var) {
                let v = v.trim().to_string();
                if v.is_empty() {
                    return Err(ConfigError::InvalidValue(format!(
                        "{var} must not be empty"
                    )));
                }
                match slot {
                    0 => cfg.bind_addr = v,
                    1 => cfg.ollama_url = v.trim_end_matches('/').to_string(),
                    2 => cfg.model = v,
                    3 => cfg.language = v.to_lowercase(),
                    _ => unreachable!(),
                }
            }
        }

        if let Ok(v) = std::env::var("TL_CONCURRENCY") {
            cfg.concurrency = parse_usize("TL_CONCURRENCY", &v, Some(1), Some(64))?;
        }
        if let Ok(v) = std::env::var("TL_MICRO_BATCH") {
            cfg.micro_batch = parse_usize("TL_MICRO_BATCH", &v, Some(1), Some(64))?;
        }
        if let Ok(v) = std::env::var("TL_NUM_CTX") {
            let n = parse_usize("TL_NUM_CTX", &v, Some(512), Some(1_048_576))?;
            cfg.num_ctx = n as u32;
        }
        if let Ok(v) = std::env::var("TL_VRAM_BUDGET_MB") {
            cfg.vram_budget_mb = parse_usize("TL_VRAM_BUDGET_MB", &v, Some(256), None)? as u64;
        }
        if let Ok(v) = std::env::var("TL_REQUEST_TIMEOUT_SECS") {
            cfg.request_timeout_secs =
                parse_usize("TL_REQUEST_TIMEOUT_SECS", &v, Some(1), Some(600))? as u64;
        }
        if let Ok(v) = std::env::var("TL_MAX_RETRIES") {
            cfg.max_retries = parse_usize("TL_MAX_RETRIES", &v, Some(0), Some(10))? as u32;
        }
        if let Ok(v) = std::env::var("TL_MAX_BATCH") {
            cfg.max_batch = parse_usize("TL_MAX_BATCH", &v, Some(1), Some(10_000))?;
        }
        if let Ok(v) = std::env::var("TL_LABEL_LIBRARY") {
            let v = v.trim().to_string();
            cfg.label_library = v;
        }
        if let Ok(v) = std::env::var("TL_LIBRARY_PROMPT_MAX") {
            cfg.library_prompt_max = parse_usize("TL_LIBRARY_PROMPT_MAX", &v, Some(0), Some(1000))?;
        }
        if cfg.language.len() != 2 || !cfg.language.chars().all(|c| c.is_ascii_alphabetic()) {
            return Err(ConfigError::InvalidValue(format!(
                "TL_LANGUAGE must be a 2-letter ISO 639-1 code, got: {}",
                cfg.language
            )));
        }

        Ok(cfg)
    }
}

fn parse_usize(
    name: &str,
    raw: &str,
    min: Option<usize>,
    max: Option<usize>,
) -> Result<usize, ConfigError> {
    let v: usize = raw.trim().parse().map_err(|_| {
        ConfigError::InvalidValue(format!("{name} must be a non-negative integer, got: {raw}"))
    })?;
    if let Some(min) = min {
        if v < min {
            return Err(ConfigError::InvalidValue(format!(
                "{name} must be >= {min}, got: {v}"
            )));
        }
    }
    if let Some(max) = max {
        if v > max {
            return Err(ConfigError::InvalidValue(format!(
                "{name} must be <= {max}, got: {v}"
            )));
        }
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_sane() {
        let cfg = Config::default();
        assert_eq!(cfg.model, "qwen3.8:27b-q4_K_M");
        assert_eq!(cfg.concurrency, 1);
        assert_eq!(cfg.micro_batch, 16);
        assert_eq!(cfg.num_ctx, 4096);
        assert_eq!(cfg.language, "de");
        assert_eq!(cfg.label_library, "labels.json");
        assert_eq!(cfg.library_prompt_max, 200);
        // Hybrid CPU+GPU inference: the 27B model (18 GB at Q4_K_M) cannot fit
        // into the 8 GB VRAM budget, so the advisory check is expected to warn.
        assert!(cfg.vram_budget_mb >= 8192);
        // Hybrid decode is slow (~3-5 tok/s); each attempt needs a generous
        // timeout or every call degrades to the fallback label.
        assert!(cfg.request_timeout_secs >= 600);
    }

    #[test]
    fn parse_usize_rejects_garbage() {
        assert!(parse_usize("X", "abc", None, None).is_err());
        assert!(parse_usize("X", "-1", None, None).is_err());
        assert!(parse_usize("X", "10", Some(20), None).is_err());
        assert!(parse_usize("X", "10", None, Some(5)).is_err());
        assert_eq!(parse_usize("X", " 42 ", None, None).unwrap(), 42);
    }

    /// Serializes tests that mutate process env vars (they race otherwise).
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn library_config_from_env() {
        let _guard = ENV_LOCK.lock().unwrap();
        struct Cleanup(&'static [&'static str]);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                for var in self.0 {
                    unsafe { std::env::remove_var(var) };
                }
            }
        }
        let _cleanup = Cleanup(&["TL_LANGUAGE", "TL_LABEL_LIBRARY", "TL_LIBRARY_PROMPT_MAX"]);
        unsafe {
            std::env::set_var("TL_LANGUAGE", "en");
            std::env::set_var("TL_LABEL_LIBRARY", "/tmp/my-labels.json");
            std::env::set_var("TL_LIBRARY_PROMPT_MAX", "50");
        }
        let cfg = Config::from_env().unwrap();
        assert_eq!(cfg.label_library, "/tmp/my-labels.json");
        assert_eq!(cfg.library_prompt_max, 50);
        // Empty path disables the library.
        unsafe {
            std::env::set_var("TL_LABEL_LIBRARY", "");
        }
        let cfg = Config::from_env().unwrap();
        assert_eq!(cfg.label_library, "");
    }

    #[test]
    fn language_must_be_iso_code() {
        let _guard = ENV_LOCK.lock().unwrap();
        struct Cleanup(&'static [&'static str]);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                for var in self.0 {
                    unsafe { std::env::remove_var(var) };
                }
            }
        }
        let _cleanup = Cleanup(&["TL_LANGUAGE"]);
        // Exercises the real from_env path end-to-end.
        unsafe {
            std::env::set_var("TL_LANGUAGE", "german");
        }
        assert!(Config::from_env().is_err());
        unsafe {
            std::env::set_var("TL_LANGUAGE", "EN");
        }
        let cfg = Config::from_env().unwrap();
        assert_eq!(cfg.language, "en");
    }
}
