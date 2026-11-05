# Phase 4 Implementation Summary: Tool-Call Validation (Layer 4)

**Owner:** T4 (LangGraph Harness) | **Status:** T4 scope complete, one T1 placeholder open
**Tests:** `tests/` 181 passed (125 from Phases 1-3 + 56 in `tests/test_phase4_tool_validation.py`)

Layer 4 is deliberately small. It does one job: make Layer 3's execution levels binding, so that Layers 1-3 work as a
complete defense. Anything that did not serve that job (behavioral anomaly rules, an audit log file) was built and then
removed; see [What was removed](#what-was-removed-and-why).

## Setup

Nothing new. `AMP_SIGNING_KEY` from Phase 3 still matters: without it every memory is READ_ONLY, so Layer 4 rejects any
search shaped by a memory.

## How it works

The LLM requests a tool with a `TOOL: name("args")` line. Before anything runs, `llm_reasoning_node` passes every such
line through `harness/tool_guard.py`:

1. **Allowlist.** Only `search_vault` may be requested. `delete_memory`, `store_memory` and any unknown tool are REJECTED,
   whatever the context. Deletion stays with `memory_audit_node`, which runs only on the admin `audit_requested` flag.
2. **Origin check.** The LLM cannot say which memory prompted a call, so the call is compared with the memories in its
   context. Words the user typed are ignored. A memory that contains at least two of the call's remaining ("novel")
   words, or all of them when only one is novel, is a **suspected source**. Every suspected source must have
   `execution_level` SEARCH_ONLY (what `search_vault` requires); a READ_ONLY or DATA_ONLY source means REJECT.
3. **Decision.** ALLOW or REJECT. Each REJECT is logged as a warning (`harness.tool_guard`); nothing is written to a file.

A REJECT discards the whole reply (its prose may carry injected text). Then, **once**:

- the suspected source memories are removed from the next prompt,
- `_build_retry_notice()` appends a correction naming the blocked tool, the reason and the memory ids not to use,
- the LLM is called again.

A clean answer or an allowed search completes the turn. A second REJECT ends it with a safe notice (no tool run, nothing
changed). There is never more than one retry.

## How Layer 4 supports Layers 1-3

| Layer | Produces | Layer 4 uses it |
| :--- | :--- | :--- |
| 1 Screening | Redacted text; spoofed closing tags removed; attributes escaped | Memory text reaches the origin check already cleaned; redacted phrases no longer match a call's words |
| 2 Risk scoring | `risk_score`, `risk_level`; `high_risk` dropped before the prompt | High-risk memories are never in context, so Layer 4 needs no risk rule of its own |
| 3 Boundaries, signing, levels | `verified`, `execution_level` | The origin check reads these directly; unverified or tampered memories are READ_ONLY and cannot drive a search |

Without Layer 4 the READ_ONLY label is only text in the prompt, and the search would run. This is the single piece of
enforcement in the design.

## What T4 implemented

| Where | Change |
| :--- | :--- |
| `harness/tool_guard.py` (new) | `parse_tool_calls`, `validate_tool_call_origin`, `validate_tool_calls`, `ToolCall`, `ValidationResult`, `EXECUTION_LEVELS`, `TOOL_REQUIRED_LEVEL`, `MUTATING_TOOLS`. Pure Python; no LangGraph import, no I/O. |
| `harness/graph.py` | `llm_reasoning_node` runs an attempt loop (original + one retry). `_invoke_llm()` extracted (the LLM stub, one place to swap in a real provider, and the seam tests use). The old `startswith("TOOL: search_vault")` regex is gone; all `TOOL:` lines are parsed. |
| `harness/graph.py` | New state field `context_memories`: id, level, risk, verified and text of every memory the LLM has seen **this session**. Search results reach the LLM as plain message text, so their levels would otherwise be lost. Deduplicated and capped to the newest 200 (`_merge_context`). |
| `harness/graph.py` | `memory_search_node` sends results as a `HumanMessage` marked `additional_kwargs={"amp_source": "memory"}`, not a `SystemMessage`, so memory text never has system authority. `_is_user_authored`, `_user_text` and `_latest_user_text` exclude marked messages, so memory is never mistaken for something the user said. |
| `harness/graph.py` | Helpers `_memory_view`, `_trust_of`, `_context_entry`, `_render_memory_prompt`, `_build_retry_notice`. `_format_memory_line` was refactored onto them with unchanged output. System prompt gained rule 7. |

### Design choices worth knowing

- **Attribution uses the full context, not the retry's prompt.** The retry removes a flagged memory from the *rendered prompt*,
  but `context_memories` still holds it. A second identical request is therefore attributed to the same memory and blocked
  again, instead of passing because the memory is no longer in the prompt.
- **Attribution spans the session.** A memory surfaced by a search several turns ago is still checked.
- **Search cards are not system text.** Marking them as memory-sourced `HumanMessage`s keeps them out of the system role and
  out of the "words the user typed" set used by the origin check.

## What was removed, and why

An earlier version of Phase 4 also had behavioral anomaly rules (rate spike, repeated call, risky context, control syntax,
long arguments, multiple calls), an ALERT decision, per-session `tool_history`, and a JSONL audit log
(`harness/audit.py`, `AMP_AUDIT_LOG`). None of it is needed for Layers 1-3 to work completely; the anomaly rules were
heuristics with arbitrary thresholds and the audit log was extra infrastructure. Removed: `detect_behavioral_anomalies`,
`audit_log_tool_call`, `harness/audit.py`, the `tool_history` state field, the ALERT decision and the `.gitignore` entry for
`storage/audit/`.

## Differences from the original architecture doc (on purpose)

1. **Logic lives in `tool_guard.py`**, not inside `graph.py`; `graph.py` only wires it in.
2. **Check "did this originate from user input?"** is implemented as the attribution above. A call whose words trace to no
   memory counts as user-originated.
3. **No behavioral anomaly detection, no ALERT, no audit log** (see above).
4. **One retry on a rejection**, so a blocked call does not simply drop the user's request.
5. **Not built:** per-sensitivity ChromaDB collections and the `FULL` level (no trusted `system` provenance exists), and an
   adversarial red-team harness.

## Placeholders for other teammates

| Owner | Where | Task |
| :--- | :--- | :--- |
| **T1** | `server/amp_memory_server.py` `delete_memory()` | The tool has no caller check. Layer 4 blocks LLM-originated deletes, but only on the harness path. Do not expose `delete_memory` / `list_all_memories` / `store_memory` to the LLM as MCP tools; consider an admin token and server-side deletion logging. |
| **T1** | (still open from Phases 2 and 3) | `_llama_guard_score()`, server-side re-scoring and re-signing in `store_memory()`, server-side high-risk filter in `page_context()`, backfill of old rows. |
| T2, T3 | none | Their stub regions are untouched. |

## Verification

- `pytest tests`: **181 passed**; all changed files compile; `python harness/graph.py` smoke test passes.
- **Mutation checks**, each making the corresponding tests fail: the level comparison disabled (every memory treated as
  fully trusted), the retry disabled (one attempt only), and the allowlist disabled.
- Covered: parsing (quoted, single-quoted, no parentheses, case-insensitive, CRLF, inline mentions ignored); the allowlist;
  the READ_ONLY / DATA_ONLY / SEARCH_ONLY matrix; user words not attributed; partial overlap; unknown level never passes;
  rejection logged and allow not logged; the graph (allowed search, unverified or codebase-sourced search blocked, delete
  blocked with no memories, blocked reply text discarded, exactly one retry, no retry when not needed, retry recovers with
  an answer or a different search, retry prompt drops the flagged memory and adds the notice, a later turn checked against
  an earlier search card); `context_memories` dedupe and cap; the message-role helpers.
- **Compiled-graph tests with a scripted LLM:** a benign search loop completes without a retry; a second, injected search
  built from an unverified memory in the first search's results is blocked, the poisoned search never runs, and the turn
  still completes after the retry; if the retry repeats the request, the turn ends with the safe notice after exactly three
  LLM calls.
- Not tested against a real LLM or a live MCP server; the LLM is still a stub.

## Known limits

- **Attribution is lexical.** A paraphrased query that shares fewer than two distinctive words with its source memory is not
  attributed. A benign query that happens to reuse two words found only in a DATA_ONLY memory is rejected (a false positive;
  the safe direction). Tool words ("search", "memory", "vault", ...) are ignored.
- **A safe, verified `user_input` memory may legitimately shape a search** (it is SEARCH_ONLY by design). Impact is limited to
  `search_vault`, which is read-only and scoped to the session.
- **Only the `TOOL:` line protocol is parsed.** If a real provider with native tool calling is added (e.g. `bind_tools`), route
  its calls through `validate_tool_calls` before executing them. A prose mention of `delete_memory(...)` is not a call and is
  not blocked.
- **No rate or repetition limit.** A runaway search loop is stopped by LangGraph's `recursion_limit`, not by Layer 4.
- **The retry's correction is an instruction to the model**, not a hard control. The hard control is that a repeat of the same
  call is attributed and blocked again. If a flagged memory's text also sits in earlier message history, the retry's prompt
  does not remove it from there.
- One REJECT blocks the whole reply, including any allowed calls in it.
- The cap on `context_memories` (200) is arbitrary; capped-out entries are forgotten.
- Carried over from Phase 3: obfuscation coverage limits (ROT13, hex, other languages).

## Remaining work

Phases 1-4 of the T4 plan are done. Open items are the T1 placeholders above and, when a real LLM is wired in, re-running the
Phase 4 scenarios against it.
