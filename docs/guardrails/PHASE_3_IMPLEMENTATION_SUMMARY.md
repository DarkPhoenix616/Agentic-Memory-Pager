# Phase 3 Implementation Summary: Obfuscation, Signing, Execution Levels

**Owner:** T4 (LangGraph Harness) | **Status:** T4 scope complete, one T1 placeholder open
**Tests:** `tests/` 125 passed (65 from earlier phases + 60 new in `tests/test_phase3_obfuscation_signing.py`)

## Setup required

Set `AMP_SIGNING_KEY` (any secret string) in the environment of the harness **and** the MCP server.
Without it nothing breaks, but no memory can be verified, so every memory is shown to the LLM as `verified="false"` / `READ_ONLY`.

## Decisions confirmed before implementing

| Question | Decision |
| :--- | :--- |
| Signature algorithm | HMAC-SHA256 keyed by `AMP_SIGNING_KEY` (a plain hash can be recomputed by whoever edits the row) |
| Missing or invalid signature | Memory is kept but forced to `READ_ONLY` (not dropped) |
| Obfuscated injection | Redact exact matches in place; fuzzy and entropy hits only raise the risk score |

## What T4 implemented

| Where | Change |
| :--- | :--- |
| `harness/guardrails.py` | `_detect_obfuscation()` returns an `ObfuscationReport` (redacted text, exact hits, fuzzy hits, entropy blobs). `_screen_injection()` now calls it, so **every existing validator got the upgrade with no validator edits**. |
| `harness/guardrails.py` | Exact hits: Layer 1 patterns re-run on a normalized copy (NFKD + accents stripped, zero-width/bidi chars dropped, Cyrillic/Greek homoglyphs, fullwidth, leetspeak with `1` tried as `i` and `l`, spaced letters collapsed) and base64 blobs that decode to an injection phrase. Only the matched span is replaced; text around it is untouched. |
| `harness/guardrails.py` | Soft hits (score only): fuzzy match against 9 long trigger phrases via stdlib `difflib` (no new dependency), +25 each, cap 50; high-entropy tokens (>= 40 chars, >= 4.5 bits/char), +10 each, cap 20. |
| `harness/signing.py` (new) | `sign_memory()` / `verify_memory()`. Signature covers `id`, `session_id`, `memory_type`, `compaction_source`, `provenance`, `risk_score` and the payload. Not covered: `vector_embedding_id` (assigned after signing). |
| `harness/guardrails.py` | `UnifiedMemory.signature: Optional[str]` (default `None`, so old rows still load). |
| `harness/graph.py` | Both compactor nodes sign the envelope as the last step. |
| `harness/graph.py` | `_assign_execution_level(risk_level, provenance, verified)`. Base: `user_input` = SEARCH_ONLY, others = DATA_ONLY. `suspicious` steps down one level; `high_risk` or unverified = READ_ONLY. Trust never increases with risk (tested). |
| `harness/graph.py` | `_format_memory_line()` now renders `<MEMORY_ENTRY ... verified execution_level risk_score provenance signature>` per memory (inside the existing `<UNTRUSTED_MEMORY>` wrapper). Attribute values are HTML-escaped. The system prompt gained rules 5-6 explaining levels and `verified="false"`. A signature that is present but wrong logs a warning. |
| `harness/guardrails.py` | `</MEMORY_ENTRY>` added to the spoofed-tag pattern, same reason as `</UNTRUSTED_MEMORY>` in Phase 2. |

## Differences from the architecture doc (on purpose)

1. **Sign at store time, verify at retrieval.** The doc computes the hash at retrieval, which cannot detect tampering (it hashes whatever it fetched).
2. **Per-memory `<MEMORY_ENTRY>` tags inside one `<UNTRUSTED_MEMORY>` wrapper**, instead of one `<UNTRUSTED_MEMORY>` per memory. The security rules and the Layer 1 tag list stay unchanged.
3. **Level mapping:** the doc gives `codebase` + `suspicious` DATA_ONLY; here it is READ_ONLY so levels never rise with risk. No `FULL` level, since no trusted `system` provenance exists.
4. The doc's single `_detect_obfuscation()` is kept by name but returns a report, because redaction and scoring need different parts of it.

## Placeholders for other teammates

| Owner | Where | Task |
| :--- | :--- | :--- |
| **T1** | `server/amp_memory_server.py` `store_memory()` | If you change any signed field (e.g. raise `risk_score` from the Phase 2 re-scoring TODO), call `memory.signature = sign_memory(memory)` as the **last** step, or your change invalidates T4's signature. Make sure the server process has the same `AMP_SIGNING_KEY`. |
| **T1** | (from Phase 2, still open) | `_llama_guard_score()`, server-side `page_context()` risk filter, backfill of pre-Phase-2 rows. Pre-Phase-3 rows are unsigned and show as READ_ONLY until re-signed. |
| T2, T3 | none | Their stub regions are untouched; signing happens after their schema is built. |

## Verification

- `pytest tests`: **125 passed**, `python harness/graph.py` smoke test passes, all changed files compile.
- Covered: each disguise type redacted in place with surrounding text intact, benign base64 and ordinary text with digits/symbols unchanged, idempotent screening, schema validators, each scoring signal, no double counting, signing tamper cases (payload, risk_score, provenance, session_id, wrong key, missing key, signature moved to another memory), JSON round trip, execution-level table and monotonicity, attribute-injection via `file_path`, compactor nodes with and without a key.
- **End-to-end against real ChromaDB (temporary database, not your `storage/`):** a memory stored through the real `store_memory()` verified after the Chroma round trip and rendered `verified="true"` / `SEARCH_ONLY`; after editing its summary directly in ChromaDB it rendered `verified="false"` / `READ_ONLY`. This was a one-off script, not a committed test.
- Side effect: that run made ChromaDB download its default embedding model (about 79 MB) into `C:\Users\aj281\.cache\chroma`. It is a one-time cache and the temp database was deleted.

## Measured cost

Screening (every validator call): about 18 ms for 8,000 characters. Risk scoring (once per compaction): about 160 ms for 8,000 characters, almost all of it the fuzzy pass; capped at the first 20,000 characters (about 370 ms).

## Known limits

- Spaced-out letters are only caught when words are separated by 2+ spaces; with single spaces everywhere, the words merge and the phrase is missed.
- Not covered: ROT13, hex, nested or url-safe base64, and phrases in other languages. The fuzzy list is 9 English phrases.
- A signed memory loaded under *stricter* future Layer 1 patterns gets re-redacted on load, fails verification and becomes READ_ONLY. That is the safe direction.
- Execution levels are only text in the prompt on their own. **Phase 4 (Layer 4) enforces them**: a tool call derived from a READ_ONLY or DATA_ONLY memory is rejected (`harness/tool_guard.py`).
- Carried over: symbol docstrings and signatures are scored but not redacted by Layer 1.

## Next: Phase 4 (T4) — done, see `PHASE_4_IMPLEMENTATION_SUMMARY.md`

Layer 4 was built as an allowlist plus an origin check that enforces `execution_level` on tool calls, with per-memory metadata kept in graph state (`context_memories`). Behavioral anomaly detection and an audit log were considered and dropped as outside what Layers 1-3 need.
