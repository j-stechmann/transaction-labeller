# transaction-labeller

A local REST service that labels bank transactions with **category names
generated dynamically by an LLM** via [Ollama](https://ollama.com) — no cloud
inference, no data leaves the machine. The client receives **only the label**:
there is no fixed taxonomy and no slug registry. The LLM invents short,
consistent category names in the language you request — but it draws from a
**label library**: a simple JSON file of labels already in use that the model
must reuse verbatim when one fits. New wording is added to the library
automatically, so labels stay stable across requests instead of drifting.

```
Client ──HTTP──▶ axum REST API ──▶ pipeline (chunk → parallel LLM calls →
                                 sanitize → retry → fallback) ──▶ Ollama
                                        │
                                        ▼
                              labels.json (label library)
```

## Features

- **Dynamic labels**: the LLM invents the category names (`Lebensmittel`,
  `Groceries`, `Miete`, …) — nothing is pre-determined
- **Label library**: existing labels are injected into every prompt and the
  model must reuse them verbatim when they fit; new labels are learned
  automatically (JSON file, one `label: count` map per language)
- **Only the label**: response is `{"id", "label"}` (single) / `{"results": [{id, label}, …]}` (batch) — nothing else
- **REST API**: single + batch endpoints, uniform error envelope, health
- **Parallel labelling**: transactions are chunked into micro-batches (several
  transactions per prompt), run through a semaphore-bounded pool of
  concurrent LLM calls; the model is instructed to keep wording consistent
  within a batch
- **Configurable label language** per request (`options.language`, ISO 639-1)
  or server-wide (`TL_LANGUAGE`)
- **Accuracy-first default**: `qwen3.8:27b-q4_K_M` in hybrid CPU+GPU mode
  (~18 GB weights: part on the 3070 Ti, rest in system RAM); a fast
  GPU-only profile (`qwen3.5:4b`) remains one env override away

## Model choice

| | |
|---|---|
| **Default** | `qwen3.8:27b-q4_K_M` (Q4_K_M, ~18 GB, hybrid CPU+GPU) |
| Why | Accuracy-first: only qwen3.8 size is 27B; 256K context, strong multilingual + instruction following; Apache-2.0. Hybrid mode trades latency (≈ 1–2 min per micro-batch) for quality — accepted. |
| Fast profile | `qwen3.5:4b` (Q4_K_M, ~3.4 GB, GPU-only): seconds per batch, measured 1.00 on the golden eval. `TL_MODEL=qwen3.5:4b TL_CONCURRENCY=4 TL_MICRO_BATCH=8 TL_NUM_CTX=8192 TL_REQUEST_TIMEOUT_SECS=30` |
| Rejected | `27b-q8_0` (30 GB, ~½ speed, no accuracy gain), `27b-bf16` (56 GB, does not fit), all `-mlx`/`-mxfp8`/`-nvfp4` tags (macOS only) |

Measured on the bundled golden set (20 real-world-style German/English
transactions, temperature 0, one batch, no label library) — `cargo test --
--ignored live_eval --nocapture`. Accuracy carries over to the 8 GB target;
wall times below are from the dev box (5070 Ti 16 GB, 49/66 layers on GPU)
and will be slower on the 3070 Ti (see ADR-010):

| Model | Accuracy | Stability | Wall time (20 tx) |
|---|---|---|---|
| `qwen3.8:27b-q4_K_M` (hybrid) | **0.95** (19/20) | identical across 3 runs | ~33 s |
| `qwen3.5:4b` (GPU, fast profile) | 0.90 (18/20) | identical across 2 runs | ~4 s |

(The 4B reached 1.00 in earlier runs with the label library injected —
ADR-006/ADR-010.) The 27B miss is a wording variant ("Streaming" for an
Amazon Prime SEPA debit); the 4B misses are semantic errors.

> **Note**: both are *thinking* model families. transaction-labeller sends
> `think: false` — otherwise the model burns its entire output budget on
> reasoning and never emits the label.

### Hardware notes (3070 Ti 8 GB + 5600X + 32 GB)

- The 27B Q4 does **not** fit in 8 GB VRAM; Ollama offloads most layers to
  system RAM (~30/66 layers on GPU → ~3–5 tok/s decode). The startup VRAM
  advisory therefore warns — this is expected in hybrid mode
  (`TL_STRICT_VRAM` must stay off).
- Keep other GPU-heavy services in mind: free VRAM determines how many
  layers stay on the 3070 Ti and thus the decode speed. If batches still
  time out (each attempt is 600 s by default, raiseable to 3600 via
  `TL_REQUEST_TIMEOUT_SECS`), timed-out micro-batches degrade item-wise and
  a 100-tx batch can take much longer than the 15–25 min estimate.

## Quick start

```bash
# 1. Pull the model (~18 GB)
ollama pull qwen3.8:27b-q4_K_M

# 2. Tune the Ollama server for hybrid inference (recommended)
#    Single request at a time; q8_0 KV cache halves KV memory at ~no loss.
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0
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

Response — id + label, nothing else:

```json
{"id": "tx-1", "label": "Lebensmittel"}
```

For batches:

```json
{"results": [
  {"id": "a", "label": "Lebensmittel"},
  {"id": "b", "label": "Miete"},
  {"id": "c", "label": "Einkommen"}
]}
```

## API

Interactive OpenAPI documentation is served by the binary:

- **Swagger UI**: `http://127.0.0.1:8080/swagger-ui`
- **OpenAPI 3.1 JSON**: `http://127.0.0.1:8080/api-docs/openapi.json`

### `POST /v1/label`

Label a single transaction. Body: `{"transaction": {...}, "options": {...}}`.
Response: `{"id": "…", "label": "…"}`.

### `POST /v1/label:batch`

Label up to `TL_MAX_BATCH` (default 100) transactions in one request.
Larger volumes: chunk client-side. Response: `{"results": [{id, label}, …]}` —
one result per transaction, in input order (`results[i]` ↔
`transactions[i]`). A batch never fails wholesale because of one bad item —
that item receives a generic fallback label (`Sonstige Ausgaben` /
`Sonstige Einnahmen`, English equivalents for other languages).

### `GET /v1/labels`

Inspect the label library: `?language=de` (defaults to the server language).
Returns the labels the model is instructed to reuse, most-used first:

```json
{"labels": ["Lebensmittel", "Miete", "Einkommen"]}
```

The library file (`TL_LABEL_LIBRARY`, default `labels.json`) is created on
first use; disable with `TL_LABEL_LIBRARY=""`.

### `GET /v1/health`

Liveness + Ollama reachability. `200 {"status":"ok"}` or
`503 {"status":"degraded", ...}`.

### Errors

Uniform body: `{"error":{"code":"invalid_request|backend_unavailable","message":"...","details":[]}}`

| Status | Meaning |
|---|---|
| 400 | malformed input (bad language, duplicate ids, field too long, …) |
| 413 | batch exceeds `TL_MAX_BATCH` |
| 422 | body parses but fails validation (e.g. non-finite amount) |
| 503 | LLM backend unreachable/overloaded (`Retry-After: 5`) |

### Response field semantics

- `label` — the LLM-generated category name in the requested language. The
  model draws from the label library and reuses established wording verbatim
  when it fits, so wording is stable across requests once established;
  genuinely new categories get fresh (auto-learned) labels. Normalize
  downstream if you need additional grouping.
- `id` — echoes the input transaction id, so association never depends on
  array position. (Ids must still be unique within a request; duplicates are
  rejected with 400.)
- That's the whole response.

## Configuration (environment)

| Variable | Default | Meaning |
|---|---|---|
| `TL_BIND_ADDR` | `127.0.0.1:8080` | HTTP bind address |
| `TL_OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama base URL |
| `TL_MODEL` | `qwen3.8:27b-q4_K_M` | Ollama model tag |
| `TL_LANGUAGE` | `de` | Default label language (ISO 639-1) |
| `TL_CONCURRENCY` | `1` | Max parallel LLM requests |
| `TL_MICRO_BATCH` | `16` | Transactions per prompt |
| `TL_NUM_CTX` | `8192` | Ollama `num_ctx` (prompt window) |
| `TL_MAX_BATCH` | `100` | Max transactions per batch request |
| `TL_LABEL_LIBRARY` | `labels.json` | Label-library JSON file (empty = disabled) |
| `TL_LIBRARY_PROMPT_MAX` | `200` | Max library labels shown per prompt (`0` disables the library entirely) |
| `TL_REQUEST_TIMEOUT_SECS` | `600` | Per-attempt LLM timeout (max 3600) |
| `TL_MAX_RETRIES` | `2` | Retries for transient LLM failures |
| `TL_VRAM_BUDGET_MB` | `8192` | Advisory VRAM budget |
| `TL_STRICT_VRAM` | off | `1`/`true` → exit(3) if model > 80 % of budget |

Fast profile (GPU-only, low latency) — override the defaults:

```bash
TL_MODEL=qwen3.5:4b TL_CONCURRENCY=4 TL_MICRO_BATCH=8 TL_NUM_CTX=8192 \
TL_REQUEST_TIMEOUT_SECS=30 cargo run --release
```

VRAM math (default config, hybrid): the 27B Q4_K_M weights (~18 GB) cannot
fit in 8 GB VRAM — Ollama places ~30/66 layers on the 3070 Ti and the rest
in system RAM (~3–5 tok/s decode). KV cache at `num_ctx=8192`, q8_0, 1
request ≈ ~100–200 MB. The startup advisory warns that the model exceeds the
VRAM budget — expected in hybrid mode (only for the default model; a
different `TL_MODEL` warning should be investigated).

## Label language

Request `options.language` (ISO 639-1) or set `TL_LANGUAGE`. The system
prompt instructs the model to write the entire label in that language; the
generic fallback labels are localized for `de` and `en` and default to
English otherwise. The label library is per language, so `Lebensmittel` (de)
and `Groceries` (en) never interfere.

## Label library

`labels.json` (configurable) keeps one `label: usage-count` map per language:

```json
{
  "de": { "Lebensmittel": 12, "Miete": 3 },
  "en": { "Groceries": 1 }
}
```

- **Read**: existing labels are injected into the system prompt; the model
  must reuse a listed label verbatim when it fits and only invents new
  wording when none does.
- **Write**: every label actually returned is recorded (count incremented,
  new labels added) and persisted atomically (temp file + rename). The file
  is human-editable — tweak or delete entries freely; do it while the
  service is stopped.
- Corrupt/unreadable file → logged warning, library starts empty, labelling
  keeps working.
- Disable entirely with `TL_LABEL_LIBRARY=""` (no prompts change, nothing is
  written).

## Testing

```bash
cargo test                                   # unit + integration + golden (mock LLM, no GPU)
cargo test -- --ignored live_eval --nocapture  # live eval against real Ollama (needs model)
```

- **Unit tests**: config parsing, prompt rendering (incl. library injection),
  schema generation, positional label parsing (markdown fences, prose,
  brace-in-string), label sanitization, fallback language selection, label
  library (load/record/persist/corruption/cap).
- **Integration tests** (`tests/integration.rs`): full HTTP API against a
  mock Ollama (`tests/mock_llm.rs`) — API contract, minimal response shape,
  concurrency bounds, item-wise fallback, transient-500 retry, uniform
  errors, OpenAPI spec content, label-library injection/recording/`/v1/labels`.
- **Golden tests** (`tests/golden/`): 20 real-world-style transactions run
  through the pipeline; contract preservation asserted deterministically
  against the mock.
- **Live eval** (`--ignored`): the whole set labelled in **one request**
  (dynamic labels are batch-consistent, so single-batch is the honest
  evaluation); each case lists semantically acceptable labels, and the
  produced label must match one of them. Asserts ≥ 0.8. Measured **1.00**
  with `qwen3.5:4b`; qwen3.8 hybrid numbers in ADR-010.

## Operations

- Graceful shutdown on SIGTERM/SIGINT (in-flight requests finish).
- Retries: transient failures (network, 429, 5xx) get 2 attempts with
  exponential backoff + jitter; timeouts degrade item-wise to the fallback
  label.
- Structured logs via `tracing` (`RUST_LOG=transaction_labeller=debug`).
- The service performs **no authentication**; keep it on loopback (a warning
  is logged when binding non-loopback).

## Git flow

`main` is releasable only; `develop` is the integration branch; work happens
on `feature/*`, releases via `release/*` → `main` (tagged `vX.Y.Z`), urgent
fixes via `hotfix/*` from `main`.

See [docs/design.md](docs/design.md) for the full design document and
[docs/adr.md](docs/adr.md) for decision records.