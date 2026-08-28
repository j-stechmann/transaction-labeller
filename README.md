# transaction-labeller

A local REST service that labels bank transactions with **spending/income
categories** using an LLM served by [Ollama](https://ollama.com) — no cloud
inference, no data leaves the machine. Designed for an 8 GB VRAM budget
(e.g. RTX 3070 Ti).

```
Client ──HTTP──▶ axum REST API ──▶ pipeline (chunk → parallel LLM calls →
                                 validate → retry → fallback) ──▶ Ollama
```

## Features

- **REST API**: single + batch endpoints, uniform error envelope, health & taxonomy
- **Parallel labelling**: transactions are chunked into micro-batches (several
  transactions per prompt) and processed by a semaphore-bounded pool of
  concurrent LLM calls
- **Guaranteed taxonomy**: the model decodes against a JSON-schema enum
  (grammar-constrained) and every label is re-validated server-side; failures
  retry individually, then fall back to `other_expense`/`other_income`
- **Configurable label language** per request (`options.language`, ISO 639-1)
  or server-wide (`TL_LANGUAGE`); labels are stable via canonical ASCII slugs
- **Direction safety**: income categories can never be attached to outflows
  (and vice versa) — checked against the taxonomy's per-category direction
- **8 GB VRAM aware**: startup advisory check against the Ollama model list;
  default model `qwen3.5:4b` uses ~3.4 GB

## Model choice

| | |
|---|---|
| **Default** | `qwen3.5:4b` (Q4_K_M, ~3.4 GB) |
| Why | 201 languages incl. German; IFEval 89.8 (instruction following); strong structured-output accuracy; Apache-2.0 |
| Alternatives | `gemma4:12b` (better raw quality, 7.6 GB — no concurrency headroom), `qwen3:8b` (thinking model, 5.2 GB), `qwen3.5:9b` (6.6 GB, tight) |

Measured on the bundled golden set (20 real-world-style German/English
transactions, temperature 0): **0.90 macro accuracy**, stable across runs
(`cargo test -- --ignored live_eval --nocapture`).

> **Note**: `qwen3.5` is a *thinking* model. transaction-labeller sends
> `think: false` — otherwise the model burns its entire output budget on
> reasoning and never emits the classification.

## Quick start

```bash
# 1. Pull the model (3.4 GB)
ollama pull qwen3.5:4b

# 2. Allow real parallelism (optional but recommended)
#    Otherwise TL_CONCURRENCY requests queue inside Ollama.
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_MAX_LOADED_MODELS=1
systemctl restart ollama   # or restart ollama serve

# 3. Build & run
cargo run --release
# → listening on 127.0.0.1:8080

# 4. Label a transaction
curl -s localhost:8080/v1/label -H 'content-type: application/json' -d '{
  "transaction": {
    "id": "tx-1",
    "counterparty": "REWE SAGT DANKE",
    "purpose": "Einkauf 14.02",
    "amount": -42.13,
    "currency": "EUR",
    "date": "2026-02-14"
  },
  "options": { "language": "de" }
}'
```

Response:

```json
{
  "results": [
    {
      "id": "tx-1",
      "category": "groceries",
      "category_label": "Lebensmittel",
      "direction": "expense",
      "status": "ok",
      "model": "qwen3.5:4b"
    }
  ],
  "batch_ms": 412
}
```

## API

### `POST /v1/label`

Label a single transaction.

### `POST /v1/label:batch`

Label up to `TL_MAX_BATCH` (default 100) transactions in one request.
Larger volumes: chunk client-side. The response preserves input order
(`results[i]` corresponds to `transactions[i]`); a batch never fails
wholesale because of one bad item — that item gets
`"status": "fallback_unknown"` with category `other_expense`/`other_income`.

### `GET /v1/health`

Liveness + Ollama reachability. `200 {"status":"ok"}` or
`503 {"status":"degraded", ...}`.

### `GET /v1/taxonomy?language=en`

Effective taxonomy: `{"language":"en","categories":[{"slug":"groceries","label":"Groceries"}, ...]}`.

### Errors

Uniform body: `{"error":{"code":"invalid_request|backend_unavailable","message":"...","details":[]}}`

| Status | Meaning |
|---|---|
| 400 | malformed input (bad language, duplicate ids, field too long, …) |
| 413 | batch exceeds `TL_MAX_BATCH` |
| 422 | body parses but fails validation (e.g. non-finite amount) |
| 503 | LLM backend unreachable/overloaded (`Retry-After: 5`) |

### Field semantics

- `category` — canonical ASCII slug, **stable across languages**. Key on this.
- `category_label` — localized display name for the requested language.
- `direction` — derived from the amount sign (`amount == 0` → expense).
- `status` — `ok` or `fallback_unknown` (label could not be validated).

## Configuration (environment)

| Variable | Default | Meaning |
|---|---|---|
| `TL_BIND_ADDR` | `127.0.0.1:8080` | HTTP bind address |
| `TL_OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama base URL |
| `TL_MODEL` | `qwen3.5:4b` | Ollama model tag |
| `TL_LANGUAGE` | `de` | Default label language (ISO 639-1) |
| `TL_CONCURRENCY` | `4` | Max parallel LLM requests |
| `TL_MICRO_BATCH` | `8` | Transactions per prompt |
| `TL_NUM_CTX` | `8192` | Ollama `num_ctx` (prompt window) |
| `TL_MAX_BATCH` | `100` | Max transactions per batch request |
| `TL_REQUEST_TIMEOUT_SECS` | `30` | Per-attempt LLM timeout |
| `TL_MAX_RETRIES` | `2` | Retries for transient LLM failures |
| `TL_VRAM_BUDGET_MB` | `8192` | Advisory VRAM budget |
| `TL_STRICT_VRAM` | off | `1`/`true` → exit(3) if model > 80 % of budget |
| `TL_TAXONOMY` | built-in | Path to a custom taxonomy JSON |

VRAM math (default config): 3.4 GB weights + ≤ ~320 MB KV cache (4 × 8192 ctx)
+ ~800 MB CUDA/driver/desktop ≈ **4.6 GB worst case** on an 8 GB card.

## Custom taxonomy

```json
{
  "categories": [
    { "slug": "food", "direction": "expense", "names": { "de": "Essen", "en": "Food" } },
    { "slug": "my_income", "names": { "en": "My Income" } }
  ]
}
```

- `slug`: lowercase ASCII + underscores; this is the model-facing enum value
  and the API identifier.
- `direction`: `income` or `expense`. Optional only when inferable from the
  slug (`*_income` / `*_expense`).
- `names`: display names per ISO 639-1 code; unknown languages fall back to
  `de`, then to any provided name.
- A generic fallback category (`other_income`/`other_expense` or any slug
  containing `other`) must exist for **both directions**, otherwise startup
  fails.

## Testing

```bash
cargo test                                   # unit + integration + golden (mock LLM, no GPU)
cargo test -- --ignored live_eval --nocapture  # live eval against real Ollama (needs model)
```

- **Unit tests**: config parsing, taxonomy validation, prompt rendering,
  schema generation, positional output parsing (markdown fences, prose,
  brace-in-string), direction derivation.
- **Integration tests** (`tests/integration.rs`): full HTTP API against a
  mock Ollama (`tests/mock_llm.rs`) — API contract, concurrency bounds,
  item-wise fallback, transient-500 retry, uniform errors.
- **Golden tests** (`tests/golden/`): 20 real-world-style transactions run
  through the pipeline; contract preservation asserted deterministically.
- **Live eval** (`--ignored`): same set against the real model with
  temperature 0; reports per-category recall and confusions; asserts ≥ 0.8.

## Operations

- Graceful shutdown on SIGTERM/SIGINT (in-flight requests finish).
- Retries: transient failures (network, 429, 5xx) get 2 attempts with
  exponential backoff + jitter; timeouts degrade item-wise to fallback.
- Structured logs via `tracing` (`RUST_LOG=transaction_labeller=debug`).
- The service performs **no authentication**; keep it on loopback (a warning
  is logged when binding non-loopback).

## Git flow

`main` is releasable only; `develop` is the integration branch; work happens
on `feature/*`, releases via `release/*` → `main` (tagged `vX.Y.Z`), urgent
fixes via `hotfix/*` from `main`.

See [docs/design.md](docs/design.md) for the full design document.