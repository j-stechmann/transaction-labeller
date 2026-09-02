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
- The response per transaction is the label only (see ADR-007).
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

# ADR-007: Label-only responses (amends ADR-005)

Date: 2026-08-28
Status: Accepted (amended by ADR-008: id is echoed, the rest stands)

## Context

v0.2.0's response per transaction was `{id, label, [rationale], model}` plus
a `batch_ms` envelope field. The product owner asked for the returned JSON to
be as simple as possible: "I really only need the label."

## Decision

- `POST /v1/label` → `{"label": "…"}`
- `POST /v1/label:batch` → `{"labels": ["…", …]}` (positional, `labels[i]` ↔
  `transactions[i]`)
- Removed: id echo, `model` tag, `batch_ms`, and the `include_rationale`
  option (rationale was opt-in metadata; nobody asked for it).
- Input `id`s remain required and are used only to reject duplicates within
  a request; batch association is positional.
- Timing/latency is available in the structured logs instead of the response.

## Consequences

- The client contract is a single string per transaction — trivially
  consumable.
- Batch clients associate by index instead of by id echo; duplicate-id
  rejection still guards against accidental double-submission.
- OpenAPI schemas shrink to `SingleLabelResponse` / `BatchLabelResponse`,
  each with exactly one property.

# ADR-008: id echo in responses (amends ADR-007)

Date: 2026-08-28
Status: Accepted (amends ADR-007)

## Context

ADR-007 removed the id echo, making batch association purely positional.
The product owner reconsidered: ids in requests and responses are worthwhile
so clients don't tangle label↔transaction association with array mechanics.

## Decision

- `POST /v1/label` → `{"id": "…", "label": "…"}`
- `POST /v1/label:batch` → `{"results": [{"id", "label"}, …]}` — positional
  order preserved *and* ids echoed; clients may rely on either.
- The rest of ADR-007 stands: no model tag, no timing, no rationale.
- Duplicate ids within a request remain a 400.

## Consequences

- Association is explicit and order-independent for clients; the pipeline
  still guarantees positional order as a second safety net.
- Response remains minimal: two fields per transaction.

# ADR-009: Label library — persisted reuse of existing labels

Date: 2026-08-28
Status: Accepted (amends ADR-005; keeps "no fixed taxonomy" intact)

## Context

ADR-005 made labels fully dynamic. The known trade-off: wording drifts
across requests with batch composition (`Lebensmittel` vs `Einkauf` vs
`Lebensmittelkauf` for the same groceries). The in-batch consistency
instruction fixes single batches, but across requests/batches the model has
no memory. The product owner wants consistency: if a suitable label already
exists, the model should return **that** label instead of inventing a new
variant — which implies remembering used labels somewhere.

## Decision

A **label library**: a single JSON file (`TL_LABEL_LIBRARY`, default
`labels.json`) mapping language → `{label: usage_count}`.

- **Read**: per request, the labels for the request language (most-used
  first, capped at `TL_LIBRARY_PROMPT_MAX` = 200) are injected into the
  system prompt with an explicit MUST-reuse-verbatim rule; inventing new
  wording is allowed only when nothing fits.
- **Write**: after each successful micro-batch, every label actually
  returned is recorded (count incremented, new labels appended) and the file
  is persisted atomically (temp file + rename; Windows copy fallback, since
  `rename` fails there when the destination exists). Recording happens only
  after the whole request succeeded, so a client retry after a 503 never
  double-counts labels.
- **Discovery**: read-only `GET /v1/labels?language=…` (server default
  language when omitted).
- **Failure posture**: missing/corrupt/unwritable file never fails a
  labelling request — warn and continue (empty in-memory state; a corrupt
  file is overwritten on the next persist). Individual malformed entries (a
  bad count, a non-object language) are skipped at load instead of failing
  the whole parse, so one bad hand-edit cannot discard the rest of the
  library; the surviving entries are kept and re-persisted.
- **Opt-out**: `TL_LABEL_LIBRARY=""` restores exact ADR-005 behaviour
  (no injection, no writes).
- The library is *not* a taxonomy: it has no direction metadata, no slugs,
  no validation role, and it grows only from what the model actually
  produced (plus manual edits while the service is stopped). It constrains
  wording preference, never the API contract.

## Alternatives considered

| Option | Verdict |
|---|---|
| No persistence, prompt-only consistency | Status quo of ADR-005; drift across requests remains. Rejected — this is the problem being solved. |
| SQLite | Unnecessary machinery for a per-language string→count map; harder to inspect/edit by hand. Rejected. |
| Vector/embedding similarity matching | Better fuzzy reuse, but a model + index for a 4 GB-VRAM local service is overkill; exact-match prompt injection achieves most of the benefit at zero cost. Rejected (revisit if wording variants proliferate). |
| Fixed taxonomy (undo ADR-005) | Contradicts the core requirement that labels are the LLM's invention. Rejected. |

## Consequences

- Label wording stabilizes across requests once a wording is established;
  the response contract (`{id, label}`) is unchanged.
- The prompt grows with the library (bounded by `TL_LIBRARY_PROMPT_MAX`;
  200 labels ≈ well under 2k tokens, safe at `num_ctx` 8192 with
  `TL_MICRO_BATCH` 8).
- Language isolation is inherent (per-language maps): German and English
  libraries never mix.
- Concurrent writes funnel through one `Mutex`; persistence is
  best-effort — a failed disk write logs a warning and the in-memory state
  stays authoritative.
- Tests: prompt-injection assertions via captured system prompts in the
  mock; library round-trip/corruption/cap unit tests; `/v1/labels`
  integration coverage.

# ADR-010: Default model — qwen3.8:27b-q4_K_M, hybrid CPU+GPU (amends ADR-001)

Date: 2026-09-02
Status: Accepted (amends ADR-001; accuracy now outranks latency)

## Context

ADR-001 picked `qwen3.5:4b` when accuracy per second mattered most. The
requirement changed: **accuracy is now the top priority** and latency is
explicitly acceptable to trade away. The host is an RTX 3070 Ti (8 GB VRAM,
mostly headless but shared with other hosting services) + Ryzen 5600X +
32 GB RAM, so CPU+GPU hybrid inference is on the table.

The successor family `qwen3.8` ships **only as 27B** (all 12 Ollama tags are
27b variants; no 4b/8b exists). It is a thinking + vision model; thinking can
be disabled per request (`think: false` — already sent, see ADR-001
consequences).

## Decision

Default model: **`qwen3.8:27b-q4_K_M`** (18 GB at Q4_K_M, Apache-2.0),
inferred in **hybrid mode**: a minority of layers on the 3070 Ti, the
majority in system RAM.

Config defaults change to match hybrid reality:

| Setting | Old (qwen3.5:4b) | New (27B hybrid) | Why |
|---|---|---|---|
| `TL_CONCURRENCY` | 4 | **1** | hybrid decode cannot parallelize; parallel requests would thrash |
| `TL_MICRO_BATCH` | 8 | **16** | amortize the library prefill over more transactions |
| `TL_NUM_CTX` | 8192 | **4096** | prompt ≈ 2.5k tokens; halves CPU-side KV cache |
| `TL_REQUEST_TIMEOUT_SECS` | 30 | **600** | hybrid decode ≈ 3–5 tok/s; 30 s timed out every call → mass fallback |

The VRAM advisory check now warns for the default model by design (18 GB
weights > 80 % of the 8 GB budget); the warning is annotated as expected for
hybrid mode. `TL_STRICT_VRAM` must stay off with the 27B.

Ollama server-side (documented in README): `OLLAMA_NUM_PARALLEL=1`,
`OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_FLASH_ATTENTION=1`,
`OLLAMA_KV_CACHE_TYPE=q8_0`.

## Alternatives considered

| Option | Verdict |
|---|---|
| qwen3.8:27b-q8_0 (30 GB) | Fits in hybrid too, but ~½ decode speed for ≈ 0 accuracy gain on a grammar-constrained classification task. Rejected. |
| qwen3.8:27b-bf16 (56 GB) | Exceeds 8 GB VRAM + 32 GB RAM combined. Rejected. |
| qwen3.8:27b-mlx / mxfp8 / nvfp4 | MLX tags — macOS only. Rejected. |
| qwen3.5:4b (ADR-001) | Kept as documented **fast profile**: GPU-only, seconds per batch, measured 1.00 on the golden eval. Select via `TL_MODEL=qwen3.5:4b TL_CONCURRENCY=4 TL_MICRO_BATCH=8 TL_NUM_CTX=8192 TL_REQUEST_TIMEOUT_SECS=30`. |
| gemma4:12b (ADR-001) | Still rejected: no hybrid advantage over a 27B at Q4, weaker multilingual IF. |

## Consequences

- **Measured (2026-09-02, RTX 5070 Ti 16 GB + 5600X dev box, temperature 0,
  20-case golden set, one batch, no label library)** — accuracy transfers to
  the 8 GB target (same model/quant/prompt; the CPU/GPU split does not
  change output), the wall times do **not**:

  | Model | Accuracy | Stability | Wall time (dev box) |
  |---|---|---|---|
  | `qwen3.8:27b-q4_K_M` (hybrid, 49/66 layers GPU) | **0.95** (19/20) | identical across 3 runs | ~33 s per 20-tx eval |
  | `qwen3.5:4b` (GPU-only, old default) | **0.90** (18/20) | identical across 2 runs | ~4 s per 20-tx eval |

  qwen3.8's single miss (`sepa_reference_in_purpose` → "Streaming" instead
  of "Shopping/Abo") is a defensible label for an Amazon-Prime SEPA debit;
  the 4B's two misses ("Transfer" for savings transfer, "Kreditkarte" for a
  pre-authorization) are semantic errors, not wording variants. Dev-box
  hybrid throughput: **~10.9 tok/s decode, 57–141 tok/s prefill**
  (~11.3 GB weights on GPU, ~4.5 GB CPU-mapped).
- **Latency on the 8 GB target (3070 Ti + 5600X)**: only ~30/66 layers fit
  on GPU (~7 GB usable); expect ~3–5 tok/s decode. Micro-batches of 8–16
  transactions take ~1–2 min including the library prefill; a 100-tx batch
  ≈ 15–25 min. Accepted by the product owner.
- **Accuracy**: the golden-set live eval with the 4B measured 1.00 under the
  ADR-006 protocol (with library injected); the 27B holds 0.95 without a
  library and beats the 4B on the same no-library protocol (0.90). More
  robust on novel/ambiguous transactions.
- The 27B is a thinking model: `think: false` remains mandatory (already
  implemented, `src/llm.rs`).
- The fast profile remains a one-env-var override away
  (`tests/golden.rs` live eval honors `TL_MODEL` directly).
