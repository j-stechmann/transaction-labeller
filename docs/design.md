# Design: Transaction Labeller

A Rust REST service that classifies bank transactions into spending/income categories
using a locally-served LLM. No cloud inference.

## Goals

- Label transactions (income vs. spending categories) via REST API
- Configurable label language (e.g. `de`, `en`)
- Constrained decoding: the model may only choose from a fixed taxonomy
- Parallelism: concurrent LLM requests, batched into a few prompts
- 8 GB VRAM budget on an RTX 3070 Ti; model runs via Ollama

## Non-goals

- Training/fine-tuning
- Bank-statement CSV parsing (input is structured JSON transactions)
- Persistent storage of results

## Model selection (research summary)

Candidate classes, all runnable on 8 GB VRAM with Q4 quantization:

| Model | VRAM (Q4) | Multilingual | IFEval | Structured output | Notes |
|---|---|---|---|---|---|
| **Qwen3.5-4B (default Q4)** | 3.4 GB | 201 languages | 89.8 | 89.06 (ExtractBench short) | Strongest in-class multilingual+IF; hybrid GDN+MoE |
| Qwen3.5-2B | 1.9 GB | 201 languages | 89.8* | — | Same gen, weaker |
| gemma4:12b QAT | 7.2 GB | 140+ | high | good | Too tight for KV headroom in 8 GB |
| gemma4:e2b-it-qat | 4.3 GB | 140+ | high | good | Strong alternative, vision-capable |
| Llama-3.2-3B | 2.0 GB | 8 langs | 77.7 | weak | Insufficient German |
| Mistral-7B-v0.3 | 4.4 GB | ~7 | 58 | weak | Insufficient German |

*\* 2B shares the 4B recipe per Qwen model card.*

**Decision: default model `qwen3.5:4b` (Ollama tag), Q4_K_M, ~3.4 GB.**

Rationale:
1. **VRAM budget:** ~3.4 GB leaves >4.5 GB for KV cache, concurrency, and OS overhead —
   comfortable within 8 GB even with several parallel requests.
2. **Multilingual:** 201 languages (incl. German); MMMLU 76.1, INCLUDE 71.0, MAXIFE 78.0 —
   the best multilingual instruction-following in its class by a clear margin.
3. **Instruction following:** IFEval 89.8 (beats 9B-class GPT-OSS-20B at 88.2).
   Structured-extraction eval: 89.06 on ExtractBench-short.
4. **Constrained decoding:** Ollama supports JSON schema `format`; the Qwen3.5 chat
   template is well supported.
5. **License:** Apache-2.0 — no redistribution restriction.
6. **Fallback headroom:** `qwen3.5:9b` (6.6 GB) fits 8 GB only with tight KV budget —
   supported via config, not default.

Rejection of gemma4:12b: 7.6 GB weights leave no KV/concurrency headroom in 8 GB.

## Architecture

```
                    ┌─────────────────────────────────────────┐
Client ──HTTP──▶    │ axum server (REST, /v1/…)               │
                    │   ├── POST /v1/label  (single tx)       │
                    │   ├── POST /v1/label:batch (parallel)   │
                    │   └── GET  /v1/health, /v1/taxonomy     │
                    └───────────────┬─────────────────────────┘
                                    │
                          ┌─────────▼──────────┐
                          │ LabelService        │
                          │ (core pipeline)     │
                          └─────────┬──────────┘
                    ┌───────────────┼────────────────┐
                    │               │                │
              ┌─────▼─────┐  ┌──────▼──────┐  ┌──────▼──────┐
              │ Prompt     │  │ LLM client  │  │ Validator   │
              │ builder    │  │ (Ollama,    │  │ (taxonomy   │
              │ (taxonomy, │  │  JSON schema│  │  match,     │
              │  language) │  │  format)    │  │  fallback)  │
              └────────────┘  └─────────────┘  └─────────────┘
```

- **REST layer** (axum): JSON in/out. Batching endpoint fans out N transactions to a
  semaphore-bounded pool of concurrent LLM calls (default 4), each call covering a
  micro-batch (default 8 transactions per prompt).
- **Prompt builder**: renders system prompt with taxonomy + target label language +
  disambiguation rules; per-request user prompt with the micro-batch's transactions.
  Transaction fields are embedded in a structured block with field delimiters and
  control-character stripping (cheap prompt-injection mitigation; residual risk is
  mislabelling only, since the taxonomy is grammar-constrained).
- **LLM client**: HTTP to Ollama `/api/chat` with `format: <json schema>` for
  grammar-constrained decoding. `options.num_ctx` is set explicitly (default 8192)
  and `num_predict` capped (default 768) — Ollama's small default context would
  otherwise silently truncate the taxonomy prompt. Retries: 2 attempts
  (3 total) with exponential backoff (200 ms ×4, jitter), connect timeout 3 s,
  total timeout 30 s per attempt; `keep_alive: "10m"` avoids model unload between
  calls. On 503 the API responds with `Retry-After: 5`.
- **Validator**: parses JSON response; results are mapped to transactions **by
  position in the response array** (the model may echo ids, but ids are never
  trusted for association); verifies each label ∈ taxonomy enum (canonical ASCII
  slugs, case-insensitive); invalid/missing entries are retried once individually,
  then fall back to `other_expense`/`other_income` with `status: fallback_unknown`
  (ADR-004).

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

`POST /v1/label`

```json
{
  "transaction": {
    "id": "tx1",
    "counterparty": "REWE SAGT DANKE",
    "purpose": "Einkauf 14.02",
    "amount": -42.13,
    "currency": "EUR",
    "date": "2026-02-14"
  },
  "options": { "language": "de", "include_rationale": false }
}
```

Response:

```json
{
  "results": [
    {
      "id": "tx1",
      "category": "groceries",
      "category_label": "Lebensmittel",
      "direction": "expense",
      "model": "qwen3.5:4b",
      "status": "ok"
    }
  ],
  "batch_ms": 412
}
```

`category` is the **canonical slug** (stable across languages, ASCII) — it is
the model-facing enum value and the API identifier. `category_label` is the
localized display name for the requested language; clients must key on
`category`, not the label. `status` is `ok` or `fallback_unknown` (taxonomy
retry failed for that item → labelled `other_expense`/`other_income`); a batch
never fails wholesale because of individual items.

`POST /v1/label:batch` — same, with `"transactions": [...]` (max 100 per request;
larger batches must be chunked client-side — a sync request must complete within
typical client timeouts). Latency is reported per request (`batch_ms`), not per
item, since items share prompts.

Direction (income/expense) is derived from the sign of `amount` deterministically;
`amount == 0` defaults to `expense` (spending) unless the taxonomy category is an
income category. The LLM only classifies the *category* within that direction.

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
| `TL_STRICT_VRAM` | off | `1`/`true` → exit(3) if model > 80 % of budget |
| `TL_VRAM_BUDGET_MB` | `8192` | Advisory; logged + checked vs model size |
| `TL_TAXONOMY` | built-in | Path to JSON taxonomy override |

Language precedence: `options.language` > `TL_LANGUAGE`. If a taxonomy lacks a
translation for the requested language, fall back to the taxonomy's canonical
language (German by default) and log a warning.

**Ollama-side parallelism**: real parallelism requires `OLLAMA_NUM_PARALLEL` (e.g. 4)
and `OLLAMA_MAX_LOADED_MODELS` on the Ollama server; otherwise `TL_CONCURRENCY`
requests queue inside Ollama. Documented in README; `keep_alive` set per request.

### Taxonomy

Labels use **canonical ASCII slugs** as identifiers (model-facing enum + API
`category`), with localized display names. Built-in default (canonical names in
parentheses):

- Income: `salary_income` (Einkommen), `refund` (Erstattung), `transfer` (Übertragung), `other_income` (Sonstige Einnahmen)
- Expense: `housing` (Wohnen), `groceries` (Lebensmittel), `dining` (Restaurant & Café), `transport` (Transport & Mobilität), `shopping` (Shopping), `health` (Gesundheit), `leisure` (Freizeit & Unterhaltung), `subscriptions` (Abos & Dienstleistungen), `insurance` (Versicherungen), `savings_investing` (Sparen & Investieren), `education` (Bildung), `donations` (Spenden), `taxes_fees` (Steuern & Gebühren), `cash_withdrawal` (Bargeld), `credit_card_settlement` (Kreditkartenabrechnung), `transfer` (Übertragung), `other_expense` (Sonstige Ausgaben)

Every taxonomy must provide a generic fallback category for both directions
(`other_income`/`other_expense`, or any slug containing `other` with the right
direction); startup fails otherwise. Custom taxonomies can be provided via
`TL_TAXONOMY` JSON file with per-language names and an explicit per-category
`direction` (income|expense).

### VRAM budget enforcement

KV-cache math (derivation): Qwen3.5-4B KV is small (GQA, 4 KV heads, partial
attention layers); at `num_ctx=8192`, fp16 KV ≈ 50–80 MB per request — 4 concurrent
requests ≈ ≤ 320 MB. Weights 3.4 GB + KV + CUDA context/driver (~500 MB) +
desktop/compositor reserve (headless server assumed; ~300 MB if present) ≈
**~4.6 GB worst case**, well within 8 GB.

Advisory check at startup: query Ollama `/api/tags` for the model's `size`;
warn (exit non-zero with `--strict-vram`) if weights exceed
`TL_VRAM_BUDGET_MB × 0.8` (20% reserve for KV + runtime). `qwen3.5:9b` (6.6 GB)
passes this check only when `TL_VRAM_BUDGET_MB` is raised or strict mode is off —
documented as an advanced option, not a default.

## Testing strategy

1. **Unit tests** (in-crate): prompt building (language, taxonomy injection), JSON
   schema generation, response parsing/validation (incl. diacritic/case tolerance),
   direction derivation, config parsing.
2. **Integration tests**: spin the axum app with a **mock LLM server** (tiny HTTP
   server in tests) that returns canned/adaptive JSON; assert end-to-end API
   behaviour, concurrency limits (mock delays + counting), retry/fallback paths.
3. **Golden result tests**: `tests/golden/*.json` with real-world-style transactions
   (German, English) and expected canonical slugs. Golden cases include edge cases:
   ATM withdrawal, credit-card settlement, `amount = 0`, umlauts/SEPA codes in
   purpose text. The test asserts the pipeline (prompt → mock response → validation
   → response mapping) is correct; *result correctness* against a real model is
   covered by the live eval.
4. **Live evaluation harness** (`--ignored` test, requires running Ollama):
   `cargo test -- --ignored live_eval` runs the golden set against the real model
   with `temperature = 0` and a pinned model tag, reports per-category recall and a
   confusion list; asserted macro accuracy ≥ 0.8 (measured 0.90 on the bundled
   set with `qwen3.5:4b`, stable across runs at temperature 0). Some categories
   have a single sample; the set is a smoke eval, not a benchmark. Skipped by
   default so CI needs no GPU.
5. **Language-switch end-to-end test**: same request with `language: de` vs `en`
   must yield identical `category` slugs and different `category_label`s.
 6. **Parser robustness tests**: markdown-fenced JSON, surrounding prose,
    brace-in-string, duplicate entries, out-of-range indices, garbage output.
 7. **Batch load test**: batch requests against the mock with per-call
    delay assert bounded concurrency (≤ `TL_CONCURRENCY` in flight) and item-wise
    fallback for failing items.

## Git flow

- `main`: releasable only. `develop`: integration. Features via `feature/*`, release
  via `release/*`, fixes via `fix/*` or `hotfix/*` from main. Tags `vX.Y.Z` on main.

## Dependencies

axum, tokio, reqwest (rustls), serde/serde_json, tracing/tracing-subscriber,
thiserror, futures, rand.

## Risks & mitigations

- **Model hallucinates label outside taxonomy** → constrained decoding (schema
  `format`) + post-validation + retry + fallback to `other_income`/`other_expense`.
- **Ollama unavailable** → `/v1/health` reports degraded; batch returns 503 with
  structured error and `Retry-After`.
- **Slow model** → micro-batching (fewer calls), semaphore concurrency, per-attempt
  timeout (30 s), 2 retries with backoff.
- **VRAM overflow** → startup check vs budget; default model chosen at 3.4 GB.

## Observability

`tracing` structured logs; per-batch log line includes item count, fallback
count, and latency. (Request-ID middleware is future work.) Startup logs the resolved config, VRAM
check result, and taxonomy size. `/v1/health` reports backend reachability.