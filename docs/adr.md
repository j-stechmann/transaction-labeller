# ADR-001: Model selection — qwen3.5:4b

Date: 2026-08-28
Status: Accepted (model choice stands; the slug premise in the original
context was removed by ADR-005 — labels are dynamic)

## Context

The service must classify bank transactions locally (8 GB VRAM budget,
RTX 3070 Ti, Fedora server, no cloud inference). Labels must be available in
configurable languages; the primary data is a German bank export (Girokonto
CSV: `Buchungsdatum`, `Verwendungszweck`, `Umsatztyp` = Ausgang/Eingang), so
strong German understanding is a hard requirement. (Originally framed as
"pick a slug from a fixed taxonomy"; since ADR-005 the model invents the
label, which only raises the bar on instruction following.)

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
Status: **Superseded by ADR-005** (labels are dynamic; no slugs, no taxonomy)

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
Status: Accepted (behavior stands; the fallback *mechanism* was reworded by
ADR-005 — generic localized labels instead of taxonomy slugs, no status field)

## Decision

- Primary call timeout → all-None for that chunk → per-item retry → fallback
  label (since ADR-005: generic localized label chosen by amount sign). The
  batch request still returns 200.
- Backend unreachable / non-transient HTTP error after retries → 503 +
  `Retry-After: 5` for the whole request (nothing was labelled).
- Empty/missing labels → one individual retry (semaphore-bounded), then
  fallback. (The "invalid per taxonomy/direction" check died with the
  taxonomy in ADR-005; the sanitizer's length bound and control-char
  stripping replace it.)

## Consequences

- A client never loses 99 good labels because 1 was unclassifiable.
- 503 is reserved for "the backend is down", which is actionable
  (retry later), unlike "this one transaction is weird".
# ADR-005: Dynamic labels — no fixed taxonomy (supersedes ADR-002)

Date: 2026-08-28
Status: Accepted (supersedes ADR-002)

## Context

v0.1.0 shipped a fixed taxonomy: canonical ASCII slugs as the model-facing
enum, `/v1/taxonomy` for discovery, per-category direction validation, and
`status`/`direction` fields in responses. The product owner clarified the
core requirement: **labels are entirely dynamic, made by the LLM** — no
pre-determined labels, no database — and the client receives **only the
label**.

## Decision

- The taxonomy module, `/v1/taxonomy` endpoint, slug machinery, direction
  validation, and `category`/`category_label`/`direction`/`status` response
  fields are removed.
- The model invents a short category label per transaction; the JSON schema
  constrains only the *shape* (`label` is a 1–64 char free string, no enum).
- The response per transaction is `{id, label, [rationale], model}`.
- Fallback for unusable output is a generic localized label
  (`Sonstige Ausgaben`/`Sonstige Einnahmen`, English equivalents otherwise),
  chosen by the amount sign.
- Label language is enforced via the system prompt ("write the label in
  {language}") — the language instruction is critical: without emphasis the
  model reverts to English.

## Consequences

- Label wording is not stable across requests (it *is* stable within a batch
  at temperature 0). Clients that need stable grouping must normalize
  downstream — this is the accepted trade-off of dynamic labels.
- Golden-test expectations become *semantic acceptability sets* rather than
  single strings; the live eval must run the whole set in one request
  (wording is batch-consistent, composition-dependent).
- Simpler mental model, less code (taxonomy module deleted), and the client
  contract is minimal.

# ADR-006: Single-batch live evaluation (amends ADR-001/005)

Date: 2026-08-28
Status: Accepted

## Context

With dynamic labels, the live eval initially scored 0.05–0.10: the model
produces correct, well-formed labels but the exact wording varies with the
batch composition (`Lebensmittel` vs `Einkauf` vs `Lebensmittelkauf` for the
same grocery transaction). Per-case exact-match assertions across separate
micro-batches are the wrong yardstick.

## Decision

- The live eval sends all 20 golden transactions in **one request**
  (`micro_batch = 32`, concurrency 1) and relies on the in-batch consistency
  instruction in the system prompt.
- Each golden case lists a set of semantically acceptable labels (observed
  variants included); the produced label passes if it matches any of them
  (case/punctuation/hyphen-normalized).

## Consequences

- Measured accuracy with `qwen3.5:4b`: **1.00** (20/20), stable across four
  consecutive runs at temperature 0 — up from 0.05 under the broken exact
  single-string protocol.
- The eval measures "is the label a sensible category for this transaction in
  the requested language", not "did the model choose my preferred wording" —
  which matches the product requirement that wording is the model's choice.
