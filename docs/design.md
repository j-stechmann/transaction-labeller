# Design: Transaction Labeller

A Rust REST service that labels bank transactions with **category names
generated dynamically by an LLM**, served locally via Ollama. The client
receives **only the label** — no fixed taxonomy, no slugs, no database.
8 GB VRAM budget on an RTX 3070 Ti; no cloud inference.

## Goals

- Label transactions with a short category name in a configurable language
- Response contains nothing but the label (+ id, optional rationale, model)
- Parallel processing with bounded concurrency
- Local inference within 8 GB VRAM

## Non-goals

- Fixed taxonomies / label registries (labels are the LLM's choice)
- Training/fine-tuning
- Bank-statement CSV parsing (input is structured JSON transactions)
- Persistent storage of results

## Model selection (research summary)

Decision and full comparison in ADR-001. Default: **`qwen3.5:4b`**
(Ollama, Q4_K_M, ~3.4 GB, Apache-2.0). Key facts:

- 201 languages (incl. German — the primary data is a German bank export)
- IFEval 89.8 (instruction following), strong structured-output accuracy
- ~3.4 GB weights → >4.5 GB headroom for KV cache + concurrency on 8 GB
- Must be called with `think: false` (thinking mode exhausts `num_predict`
  before emitting content — discovered via live eval)

## Architecture

```
                    ┌─────────────────────────────────────────┐
Client ──HTTP──▶    │ axum server (REST, /v1/…, Swagger UI)   │
                    │   ├── POST /v1/label  (single tx)       │
                    │   ├── POST /v1/label:batch (parallel)   │
                    │   └── GET  /v1/health                   │
                    └───────────────┬─────────────────────────┘
                          ┌─────────▼──────────┐
                          │ LabelService        │
                          └─────────┬──────────┘
                    ┌───────────────┼────────────────┐
              ┌─────▼─────┐  ┌──────▼──────┐  ┌──────▼──────┐
              │ Prompt     │  │ LLM client  │  │ Label       │
              │ builder    │  │ (Ollama,    │  │ sanitizer + │
              │ (language, │  │  JSON schema│  │ retry +     │
              │  rules)    │  │  format)    │  │ fallback    │
              └────────────┘  └─────────────┘  └─────────────┘
```

- **Prompt builder**: system prompt with output contract + labelling rules
  (language, brevity, in-batch consistency); user prompt with the
  micro-batch's transactions. Fields are embedded with delimiters and
  control-character stripping (prompt-injection mitigation).
- **LLM client**: HTTP to Ollama `/api/chat` with `format: <json schema>`
  (grammar-constrained: `results[].{index,label[,rationale]}`, label 1–64
  chars — no enum, labels are free-form). `num_ctx` explicit (default 8192),
  `num_predict` capped (768), `temperature` 0, `think` false,
  `keep_alive: "10m"`. Retries: 2 attempts (3 total) with exponential backoff
  + jitter, connect timeout 3 s, total timeout 30 s per attempt; 429 and 5xx
  are transient; timeouts degrade item-wise. On 503 the API responds with
  `Retry-After: 5`.
- **Label sanitizer**: labels are free model output — trimmed, whitespace
  collapsed, control chars stripped, hard-capped at 64 chars.
- **Validation/pipeline**: results are mapped to transactions **by position**
  (model-echoed indices are never trusted). Empty/missing labels get one
  individual retry (semaphore-bounded), then a generic fallback label
  (`Sonstige Ausgaben`/`Sonstige Einnahmen`, localized for de/en).

## API

Errors use a uniform body:

```json
{ "error": { "code": "invalid_request", "message": "…", "details": [] } }
```

Status codes: `400` malformed/invalid input (unknown language code, empty
`transactions`, duplicate `id`s, non-finite amounts, over-long fields), `413`
batch too large, `503` LLM backend unreachable/overloaded (with `Retry-After:
5`). LLM timeouts never produce 504 — they degrade item-wise to fallback
(see ADR-004).

`POST /v1/label` — single transaction; `POST /v1/label:batch` — up to
`TL_MAX_BATCH` (default 100), input order preserved, item-wise fallback.

Response (both endpoints):

```json
{
  "results": [
    { "id": "tx-1", "label": "Lebensmittel", "model": "qwen3.5:4b" }
  ],
  "batch_ms": 412
}
```

`label` is the only payload the client needs: the LLM-generated category
name in the requested language. `rationale` appears only when
`include_rationale: true` is set. `id` echoes the input. `model` names the
Ollama model.

Direction is *not* returned: it is implied by the amount sign and the model
sees the signed amount.

### Configuration (environment)

| Var | Default | Meaning |
|---|---|---|
| `TL_BIND_ADDR` | `127.0.0.1:8080` | HTTP bind (loopback assumed; warning if non-loopback — no auth implemented) |
| `TL_OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama base |
| `TL_MODEL` | `qwen3.5:4b` | Model tag |
| `TL_LANGUAGE` | `de` | Label language (ISO 639-1); request `options.language` overrides |
| `TL_CONCURRENCY` | `4` | Parallel LLM requests (client-side semaphore) |
| `TL_MICRO_BATCH` | `8` | Transactions per prompt |
| `TL_NUM_CTX` | `8192` | Ollama `options.num_ctx` per request |
| `TL_MAX_BATCH` | `100` | Max transactions per batch request (413 above) |
| `TL_REQUEST_TIMEOUT_SECS` | `30` | Per-attempt LLM timeout |
| `TL_MAX_RETRIES` | `2` | Retries for transient LLM failures |
| `TL_VRAM_BUDGET_MB` | `8192` | Advisory; logged + checked vs model size |
| `TL_STRICT_VRAM` | off | `1`/`true` → exit(3) if model > 80 % of budget |

Language precedence: `options.language` > `TL_LANGUAGE`.

**Ollama-side parallelism**: real parallelism requires `OLLAMA_NUM_PARALLEL`
(e.g. 4) and `OLLAMA_MAX_LOADED_MODELS` on the Ollama server; otherwise
`TL_CONCURRENCY` requests queue inside Ollama. Documented in README;
`keep_alive` set per request.

### VRAM budget

KV-cache math: Qwen3.5-4B KV is small (GQA, 4 KV heads, partial attention);
at `num_ctx=8192`, fp16 KV ≈ 50–80 MB per request — 4 concurrent requests ≈
≤ 320 MB. Weights 3.4 GB + KV + CUDA context/driver (~500 MB) +
desktop/compositor reserve (~300 MB if present) ≈ **~4.6 GB worst case**,
well within 8 GB.

Advisory check at startup: query Ollama `/api/tags` for the model's `size`;
warn (exit non-zero with `TL_STRICT_VRAM`) if weights exceed
`TL_VRAM_BUDGET_MB × 0.8`.

## Testing strategy

1. **Unit tests** (in-crate): prompt building (language, rules), JSON schema
   generation (no enum, label length bounds), response parsing/label
   sanitization (fences, prose, brace-in-string, whitespace, length cap),
   fallback language selection, config parsing.
2. **Integration tests**: spin the axum app with a **mock LLM server** that
   returns canned/adaptive JSON; assert end-to-end API behaviour, minimal
   response shape, concurrency limits (mock delays + in-flight counters),
   retry/fallback paths, OpenAPI spec content.
3. **Golden result tests**: `tests/golden/cases.json` — 20 real-world-style
   transactions (German, English) through the pipeline against the mock;
   exact positional/label/id assertions.
4. **Live evaluation harness** (`--ignored`, requires running Ollama):
   the whole set labelled in **one request** (dynamic labels are
   batch-consistent, so single-batch is the honest evaluation); each case
   lists semantically acceptable labels and the produced label must match
   one. Measured **1.00** with `qwen3.5:4b`, stable across runs at
   temperature 0. Asserts ≥ 0.8; skipped by default so CI needs no GPU.
5. **Parser robustness tests**: markdown-fenced JSON, surrounding prose,
   brace-in-string, duplicate entries, out-of-range indices, garbage output.

## Git flow

- `main`: releasable only. `develop`: integration. Features via `feature/*`,
  release via `release/*`, fixes via `fix/*` or `hotfix/*` from main.
  Tags `vX.Y.Z` on main.

## Dependencies

axum, tokio, reqwest (rustls), serde/serde_json, tracing/tracing-subscriber,
thiserror, futures, rand, utoipa + utoipa-swagger-ui (OpenAPI docs).

## Risks & mitigations

- **Label wording drifts between requests** → inherent to dynamic labels;
  mitigated by in-batch consistency instruction and temperature 0. Clients
  normalize downstream if they need stable grouping.
- **Model emits empty/garbage labels** → grammar-constrained decoding
  (min/max length) + sanitizer + per-item retry + localized fallback label.
- **Prompt injection via transaction fields** → field sanitization; worst
  case is a wrong label, no data leaves the machine either way.
- **Ollama unavailable** → `/v1/health` reports degraded; requests return 503
  with structured error and `Retry-After`.
- **Slow model** → micro-batching, semaphore concurrency, per-attempt
  timeout (30 s), 2 retries with backoff; timeouts degrade item-wise.
- **VRAM overflow** → startup check vs budget; default model chosen at 3.4 GB.

## Observability

`tracing` structured logs; per-batch log line includes item count and
latency. Startup logs the resolved config and VRAM check result.
`/v1/health` reports backend reachability.