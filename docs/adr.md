# ADR-001: Model selection — qwen3.5:4b

Date: 2026-08-28
Status: Accepted

## Context

The service must classify bank transactions locally (8 GB VRAM budget,
RTX 3070 Ti, Fedora server, no cloud inference). Labels must be available in
configurable languages; the primary data is a German bank export (Girokonto
CSV: `Buchungsdatum`, `Verwendungszweck`, `Umsatztyp` = Ausgang/Eingang), so
strong German understanding is a hard requirement. The LLM only picks a
category slug from a fixed taxonomy; direction comes from the amount sign.

## Decision

Default model: **`qwen3.5:4b`** (Ollama tag, Q4_K_M, ~3.4 GB, Apache-2.0),
served via Ollama with `think: false`, `temperature: 0`, `num_ctx: 8192`.

## Alternatives considered

| Model | VRAM (Q4) | Verdict |
|---|---|---|
| gemma4:12b (QAT) | 7.6 GB | Strongest raw quality, but weights alone consume 95 % of budget; no KV/concurrency headroom. Rejected. |
| gemma4:e2b/e4b-it-qat | 4.3–6.1 GB | Viable alternative; slightly weaker multilingual IF than Qwen3.5-4B per published benchmarks. Kept as documented alternative. |
| qwen3.5:9b | 6.6 GB | Passes only with reduced ctx; documented as non-default upgrade. |
| qwen3:8b | 5.2 GB | Thinking model; same think-disable workaround needed; older generation. |
| Llama-3.2-3B | 2.0 GB | Only 8 officially supported languages; weak German. Rejected. |
| Mistral-7B-v0.3 | 4.4 GB | ~7 supported languages; weaker instruction following. Rejected. |

## Rationale

1. **VRAM fit with headroom**: 3.4 GB weights + ≤ ~320 MB KV (4 concurrent ×
   8192 ctx) + ~800 MB CUDA/driver ≈ 4.6 GB worst case — comfortable on 8 GB.
2. **Multilingual**: 201 languages/dialects; MMMLU 76.1, INCLUDE 71.0,
   MAXIFE 78.0 — best-in-class multilingual instruction following at this size.
3. **Instruction following**: IFEval 89.8; structured extraction 89.06
   (ExtractBench-short). Classification is an instruction-following task.
4. **Structured output**: Ollama grammar-constrained decoding (JSON schema
   enum) works with its chat template; verified live.
5. **License**: Apache-2.0.

## Consequences

- Thinking-mode models burn `num_predict` on reasoning: the client must send
  `think: false` (implemented). Discovered via the live eval — accuracy was
  0/20 with thinking on, 0.70–0.90 after disabling.
- Prompt quality dominates accuracy more than model size at this scale:
  adding three disambiguation rules to the system prompt moved macro accuracy
  from 0.85 → 0.90 (stable across 6 runs, temperature 0).
- Remaining known confusions (Amazon-Prime-membership vs order;
  transfer-to-savings vs savings deposit) are inherently ambiguous without
  account-holder context.

# ADR-002: Canonical slugs as API identity

Date: 2026-08-28
Status: Accepted

## Context

Labels must be available in a configurable language, but API consumers need a
stable key. Model-facing enum values with non-ASCII characters (`Übertragung`)
are also fragile in grammar-constrained decoding.

## Decision

Every taxonomy category has a canonical lowercase-ASCII `slug` (`groceries`)
and localized `names` (`Lebensmittel`). The slug is:
- the value in the JSON-schema enum sent to Ollama,
- the `category` field in API responses (client key),
- validated server-side, case-insensitively.

`category_label` carries the localized display name.

## Consequences

- Switching `language` never changes `category`; only `category_label`.
- Custom taxonomies must use ASCII slugs (validated at load; error otherwise).
- Direction is an explicit per-category attribute (`income`|`expense`);
  income categories that don't end in `_income` (e.g. `refund`) are handled
  correctly.

# ADR-003: Positional result association

Date: 2026-08-28
Status: Accepted

## Context

The model returns `{"results":[{"index":N,"category":...}, ...]}` for a
micro-batch. Trusting the model-echoed `index` or `id` risks silent
mis-association (model drops, duplicates, or renumbers entries).

## Decision

Results are associated **positionally**: `results[k]` maps to the k-th
transaction in the prompt; the echoed `index` is ignored. Entries missing a
valid category string leave their slot empty → per-item retry → fallback.

## Consequences

- A truncated response degrades only its tail, silently and safely.
- The prompt requires "one result per transaction, same order" to make the
  positional contract hold; validated in golden tests.

# ADR-004: Item-wise degradation, never wholesale failure

Date: 2026-08-28
Status: Accepted

## Decision

- Primary call timeout → all-None for that chunk → per-item retry → fallback
  (`other_income`/`other_expense`, `status: fallback_unknown`). The batch
  request still returns 200.
- Backend unreachable / non-transient HTTP error after retries → 503 +
  `Retry-After: 5` for the whole request (nothing was labelled).
- Invalid labels (not in taxonomy or wrong direction) → one individual retry
  (semaphore-bounded), then fallback.

## Consequences

- A client never loses 99 good labels because 1 was unclassifiable.
- 503 is reserved for "the backend is down", which is actionable
  (retry later), unlike "this one transaction is weird".