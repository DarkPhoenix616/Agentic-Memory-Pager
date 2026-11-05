# Phase 2 Implementation Summary: Risk Scoring (Layer 2)

**Owner:** T4 (LangGraph Harness) | **Status:** T4 scope complete, T1 placeholders open
**Tests:** `tests/` 65 passed (36 existing + 29 new in `tests/test_phase2_risk_scoring.py`)

## What T4 implemented

| Where | Change |
| :--- | :--- |
| `harness/guardrails.py` | `UnifiedMemory` gains `risk_score` (0-100), `risk_level`, `provenance`. `risk_level` is **always derived** from `risk_score` by a validator, so a caller cannot label a score of 90 as "safe". |
| `harness/guardrails.py` | Shared constants `SUSPICIOUS_THRESHOLD = 30`, `HIGH_RISK_THRESHOLD = 70`, and `risk_level_for_score()`. |
| `harness/guardrails.py` | `_classify_adversarial_ensemble(text, source)`: local heuristic scorer. `score_payload(payload, provenance)`: public entry point that scores every model-readable field (summary, decisions, bugs, constraints, code docstrings). |
| `harness/graph.py` | `ast_compactor_node` / `rlm_summarizer_node` score the payload and set `provenance` (`codebase` / `user_input`) on the envelope. |
| `harness/graph.py` | `_filter_high_risk()` drops memories with `risk_score >= 70`. Applied in `llm_reasoning_node` **and** `memory_search_node`. |
| `harness/graph.py` | `_format_memory_line()` now shows `risk=` and `provenance=` in each memory header. |

### Scoring rules (additive, capped at 100)

| Signal | Points |
| :--- | :--- |
| Layer 1 pattern hit or `[INJECTION_REDACTED]` marker | 40 each (two hits = high risk) |
| Tool-call bait (`delete_memory(`, `TOOL:`, "wipe the vault") | 30 each, cap 60. Halved for `codebase` source |
| Model-directed orders ("you must", "never reveal") | 20 each, cap 40 |
| Imperative-sentence density | up to 25 (cannot reach "suspicious" alone, so "Run the tests." stays safe) |

Levels: `<30` safe, `30-69` suspicious, `>=70` high_risk (not shown to the LLM).

### Two fixes found while doing this (both in T4 code)

1. **`memory_search_node` bypassed the Phase 1 boundary.** It put raw vault text into a `SystemMessage` (system-level authority) with no `<UNTRUSTED_MEMORY>` tags and no filter. It now filters and wraps results like `llm_reasoning_node`.
2. **Boundary escape:** a stored memory containing `</UNTRUSTED_MEMORY>` could close the tag early. `untrusted_memory` is now in the Layer 1 spoofed-tag pattern.

## Placeholders for other teammates

| Owner | Where | Task |
| :--- | :--- | :--- |
| **T1** | `harness/guardrails.py` `_llama_guard_score()` | Implement the Llama Guard 3 (or similar) call, returning 0-100, or `None` if unavailable. Currently returns `None`, so scoring is heuristic-only. The ensemble uses `max(heuristic, model)`. |
| **T1** | `server/amp_memory_server.py` `store_memory()` | Re-score server-side with `score_payload()` and keep `max(client_score, server_score)`. Do not trust a client-supplied score. Fields persist automatically (envelope is stored as JSON). |
| **T1** | `server/amp_memory_server.py` `page_context()` (consider `search_vault()` too) | Drop candidates with `risk_score >= HIGH_RISK_THRESHOLD` before the token-budget loop. Import the constant, do not hard-code 70. |
| **T1** | same file | Decide whether to backfill pre-Phase-2 rows. They load as `risk_score=0` ("safe") until re-scored. |
| **T2** | `ast_compactor_node` | No action needed. If uploaded documents (not source files) are ever routed here, use `provenance="uploaded_doc"`. |
| **T3** | `rlm_summarizer_node` | No action needed. T4's scoring block sits after the `END TODO [Teammate 3]` marker and works with whatever schema you build. |

Note: the earlier root-level `TEAMMATE_GUIDANCE_PHASE_2.md` assigned provenance to T2/T3. That is superseded by this file, since T4 now sets provenance in the nodes.

## Test results

- `pytest tests`: **65 passed**. New tests cover scoring and thresholds, false-positive guard on imperative docs, the model hook (can raise but never lower a score), envelope bounds / derived level / legacy JSON, the retrieval filter, both retrieval nodes, and both compactor nodes.
- Phase 1 Layer 3 prompt checks (never ran earlier because the system Python lacked `langchain_core`): all pass under the project venv.
- `python harness/graph.py` smoke test: passes.
- `server/test_sandbox.py`: not a T4 file. It fails collection from the repo root (`from sandbox import ...`) and collects 0 tests from `server/`. Pre-existing, unchanged.

## Known limits

- The scorer is heuristic. Paraphrased attacks with no trigger words, tool names or "you must" phrasing score low until T1's model hook exists.
- Not tested against a live ChromaDB (the server hard-codes the real `storage/chroma_db` path). Graph nodes were tested with the MCP wrappers mocked.
- Symbol docstrings and signatures are scored but not redacted by Layer 1 (only `summary` and text fields are). `_format_memory_line` currently shows signatures only, never docstrings.

## Next: Phase 3 (T4)

`_detect_obfuscation()` (fuzzy match, entropy, unicode), memory signatures (sha256), `_assign_execution_level()`.
