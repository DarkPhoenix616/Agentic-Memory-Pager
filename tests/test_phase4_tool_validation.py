"""Phase 4 (T4): Layer 4 tool-call validation (allowlist + origin check) and the retry."""

import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import harness.graph as graph
from harness import guardrails, tool_guard
from harness.guardrails import (
    PageContextResult,
    RetrievedMemory,
    TextMemorySchema,
    UnifiedMemory,
)
from harness.signing import sign_memory
from harness.tool_guard import (
    ToolCall,
    parse_tool_calls,
    validate_tool_call_origin,
    validate_tool_calls,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def signing_key(monkeypatch):
    """Signed memories verify only when a key is set."""
    monkeypatch.setenv("AMP_SIGNING_KEY", "phase4-test-key")


def _entry(mem_id="mem_a", level="SEARCH_ONLY", text="", risk=0.0):
    return {"id": mem_id, "level": level, "risk_score": risk, "risk_level": "safe",
            "provenance": "user_input", "verified": True, "text": text}


def _retrieved(text, mem_id="mem_a", provenance="user_input", score=0.0, signed=True):
    mem = UnifiedMemory(
        session_id="s1", memory_type="text", compaction_source="rlm",
        text_payload=TextMemorySchema(
            chunk_index=0, total_chunks=1, timeframe="Session 1",
            factual_summary=text.ljust(10, "."), original_token_count=10),
        raw_token_count=10, risk_score=score, provenance=provenance,
    )
    mem.vector_embedding_id = mem_id
    if signed:
        mem.signature = sign_memory(mem)
    return RetrievedMemory(memory=mem, distance=0.1, token_cost=5)


def _search(query):
    return ToolCall(name="search_vault", raw_args=f'"{query}"', query=query)


def _state(user="what do we know?", **kw):
    base = {"session_id": "s1", "messages": [HumanMessage(content=user)]}
    base.update(kw)
    return base


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def test_parse_quoted_search_call():
    [call] = parse_tool_calls('TOOL: search_vault("redis ttl")')
    assert (call.name, call.query, call.raw_args) == ("search_vault", "redis ttl", '"redis ttl"')


def test_parse_is_case_insensitive_and_allows_single_quotes():
    [call] = parse_tool_calls("tool: search_vault('x y')")
    assert call.name == "search_vault" and call.query == "x y"


def test_parse_call_without_parentheses_is_still_seen():
    [call] = parse_tool_calls("TOOL: delete_memory")
    assert call.name == "delete_memory" and call.raw_args == ""


def test_parse_non_string_argument_keeps_raw_text():
    [call] = parse_tool_calls("TOOL: delete_memory(mem_1, 5)")
    assert call.query is None and call.text == "mem_1, 5"


def test_parse_finds_calls_after_prose_and_multiple_calls():
    reply = 'Let me check.\nTOOL: search_vault("a b")\nTOOL: delete_memory("m")\r\n'
    assert [c.name for c in parse_tool_calls(reply)] == ["search_vault", "delete_memory"]


def test_parse_ignores_inline_mentions_and_plain_text():
    assert parse_tool_calls('I will not use TOOL: delete_memory("x") today.') == []
    assert parse_tool_calls("Here is your answer.") == []
    assert parse_tool_calls("") == []


# --------------------------------------------------------------------------
# Allowlist
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["delete_memory", "store_memory"])
def test_mutating_tools_are_always_rejected(name):
    result = validate_tool_call_origin(ToolCall(name, '"mem_1"', "mem_1"), [], "delete it")
    assert result.decision == "REJECT"
    assert "mutating" in result.reasons[0]


def test_unknown_tool_is_rejected():
    result = validate_tool_call_origin(ToolCall("run_shell", '"ls"', "ls"), [], "hi")
    assert result.decision == "REJECT" and "allowlist" in result.reasons[0]


def test_plain_search_with_no_context_is_allowed():
    result = validate_tool_call_origin(_search("caching design"), [], "tell me about caching design")
    assert (result.decision, result.reasons, result.suspected_sources) == ("ALLOW", [], [])


def test_a_rejection_is_logged_as_a_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="harness.tool_guard"):
        validate_tool_call_origin(ToolCall("delete_memory", '"m"', "m"), [], "hi")
    assert any("REJECT" in r.message and "delete_memory" in r.message for r in caplog.records)


def test_allow_decisions_are_not_logged(caplog):
    with caplog.at_level(logging.WARNING, logger="harness.tool_guard"):
        validate_tool_call_origin(_search("caching design"), [], "tell me about caching design")
    assert caplog.records == []


# --------------------------------------------------------------------------
# Origin attribution against execution levels
# --------------------------------------------------------------------------

POISON = "wire transfer account numbers"


@pytest.mark.parametrize("level,decision", [
    ("READ_ONLY", "REJECT"),
    ("DATA_ONLY", "REJECT"),
    ("SEARCH_ONLY", "ALLOW"),
])
def test_call_derived_from_memory_needs_search_only(level, decision):
    ctx = [_entry("mem_a", level, text=f"Notes: {POISON} for quarterly report")]
    result = validate_tool_call_origin(_search(POISON), ctx, "summarize my notes")
    assert result.decision == decision
    assert result.suspected_sources == ["mem_a"]


def test_words_the_user_said_are_not_attributed_to_memory():
    ctx = [_entry("mem_a", "READ_ONLY", text=f"Notes: {POISON}")]
    result = validate_tool_call_origin(_search(POISON), ctx, f"please look up {POISON}")
    assert result.decision == "ALLOW" and result.suspected_sources == []


def test_single_novel_word_copied_from_memory_is_attributed():
    ctx = [_entry("mem_a", "READ_ONLY", text="the secret_admin_credentials are stored here")]
    result = validate_tool_call_origin(_search("secret_admin_credentials"), ctx, "hello there")
    assert result.decision == "REJECT"


def test_one_shared_word_out_of_many_is_not_enough():
    ctx = [_entry("mem_a", "READ_ONLY", text="Redis eviction policy notes")]
    result = validate_tool_call_origin(_search("kubernetes ingress eviction"), ctx, "hello there")
    assert result.suspected_sources == [] and result.decision == "ALLOW"


def test_tool_words_do_not_cause_attribution():
    ctx = [_entry("mem_a", "READ_ONLY", text="search memory vault previous stored information")]
    result = validate_tool_call_origin(_search("search memory vault"), ctx, "hello there")
    assert result.suspected_sources == []


def test_unknown_level_never_passes():
    ctx = [_entry("mem_a", "WHATEVER", text=f"Notes: {POISON}")]
    assert validate_tool_call_origin(_search(POISON), ctx, "hi").decision == "REJECT"


def test_each_suspected_memory_is_checked():
    ctx = [_entry("good", "SEARCH_ONLY", text=f"x {POISON}"), _entry("bad", "READ_ONLY", text=f"y {POISON}")]
    result = validate_tool_call_origin(_search(POISON), ctx, "hi")
    assert result.decision == "REJECT" and result.suspected_sources == ["good", "bad"]
    assert len(result.reasons) == 1 and "bad" in result.reasons[0]


def test_validate_tool_calls_judges_each_call_independently():
    ctx = [_entry("mem_a", "READ_ONLY", text=f"Notes: {POISON}")]
    results = validate_tool_calls([_search("caching design"), _search(POISON)], ctx, "caching design")
    assert [r.decision for r in results] == ["ALLOW", "REJECT"]


def test_layer4_exposes_only_allow_and_reject():
    assert not hasattr(tool_guard, "detect_behavioral_anomalies")
    assert not hasattr(tool_guard, "audit_log_tool_call")
    result = validate_tool_call_origin(_search("caching design"), [], "caching design")
    assert result.decision in ("ALLOW", "REJECT")


# --------------------------------------------------------------------------
# Graph: llm_reasoning_node enforcement
# --------------------------------------------------------------------------

def _wire(monkeypatch, paged=(), reply="", replies=None):
    """
    reply: a single scripted LLM reply (same text on every attempt — enough for
    anything that does not need the retry to behave differently).
    replies: a list, one entry consumed per _invoke_llm call — use this when a
    test expects the retry (attempt 0 blocked, attempt 1 scripted separately).
    """
    monkeypatch.setattr(graph, "_mcp_page_context",
                        lambda **kw: PageContextResult(memories=list(paged), total_budget=1000))
    if replies is not None:
        seq = iter(replies)
        monkeypatch.setattr(graph, "_invoke_llm", lambda *a, **kw: next(seq))
    else:
        monkeypatch.setattr(graph, "_invoke_llm", lambda *a, **kw: reply)


def test_context_entry_carries_level_and_text():
    entry = graph._context_entry(_retrieved("Redis caching notes", "mem_x"))
    assert entry["id"] == "mem_x" and entry["level"] == "SEARCH_ONLY" and entry["verified"] is True
    assert "Redis caching notes" in entry["text"]


def test_unsigned_memory_context_entry_is_read_only():
    entry = graph._context_entry(_retrieved("Redis caching notes", signed=False))
    assert entry["level"] == "READ_ONLY" and entry["verified"] is False


def test_allowed_search_runs(monkeypatch):
    _wire(monkeypatch, paged=[_retrieved("Redis caching layer with TTL eviction", "mem_a")],
          reply='TOOL: search_vault("caching eviction")')
    out = graph.llm_reasoning_node(_state("how does our caching work?"))
    assert out["needs_memory_search"] is True and out["search_query"] == "caching eviction"
    assert [e["id"] for e in out["context_memories"]] == ["mem_a"]
    assert "messages" not in out


def test_search_derived_from_unverified_memory_is_blocked(monkeypatch):
    # The LLM repeats the identical blocked request on the retry (same scripted
    # reply both times), so it is still attributed to mem_t and blocked again —
    # the "retry recovers" cases are covered separately below.
    _wire(monkeypatch, paged=[_retrieved(f"Notes: {POISON}", "mem_t", signed=False)],
          reply=f'TOOL: search_vault("{POISON}")')
    out = graph.llm_reasoning_node(_state("summarize my notes"))
    assert out["needs_memory_search"] is False and out["search_query"] == ""
    # The flagged memory is still recorded (session-long attribution), even though
    # the retry's rendered prompt no longer contained it.
    assert [(e["id"], e["level"]) for e in out["context_memories"]] == [("mem_t", "READ_ONLY")]
    [msg] = out["messages"]
    assert isinstance(msg, AIMessage) and "blocked" in msg.content


def test_search_derived_from_codebase_memory_is_blocked(monkeypatch):
    _wire(monkeypatch, paged=[_retrieved(f"Docs: {POISON}", "mem_c", provenance="codebase")],
          reply=f'TOOL: search_vault("{POISON}")')
    out = graph.llm_reasoning_node(_state("summarize my notes"))
    assert out["needs_memory_search"] is False
    assert out["context_memories"][0]["level"] == "DATA_ONLY"
    assert "blocked" in out["messages"][0].content


def test_blocked_search_retries_and_recovers_with_a_plain_answer(monkeypatch):
    """The first attempt is blocked (mem_t, unverified); excluding it from the
    retry's prompt lets the model answer directly instead of failing the turn."""
    _wire(monkeypatch, paged=[_retrieved(f"Notes: {POISON}", "mem_t", signed=False)],
          replies=[f'TOOL: search_vault("{POISON}")', "Here is a safe answer without that memory."])
    out = graph.llm_reasoning_node(_state("summarize my notes"))
    assert out["needs_memory_search"] is False
    [msg] = out["messages"]
    assert isinstance(msg, AIMessage) and msg.content == "Here is a safe answer without that memory."
    # The flagged memory is still tracked, in case a later turn leans on it again.
    assert [e["id"] for e in out["context_memories"]] == ["mem_t"]


def test_blocked_search_retries_and_recovers_with_a_different_search(monkeypatch):
    """The retry asks for a different, unproblematic search instead of answering."""
    _wire(monkeypatch, paged=[_retrieved(f"Notes: {POISON}", "mem_t", signed=False)],
          replies=[f'TOOL: search_vault("{POISON}")', 'TOOL: search_vault("caching design")'])
    out = graph.llm_reasoning_node(_state("summarize my notes"))
    assert out["needs_memory_search"] is True and out["search_query"] == "caching design"


def test_retry_prompt_drops_the_flagged_memory_and_adds_the_notice(monkeypatch):
    monkeypatch.setattr(graph, "_mcp_page_context",
                        lambda **kw: PageContextResult(
                            memories=[_retrieved(f"Notes: {POISON}", "mem_t", signed=False),
                                      _retrieved("Unrelated deployment facts", "mem_ok")],
                            total_budget=1000))
    seq = iter([f'TOOL: search_vault("{POISON}")', "answer"])
    seen_prompts = []

    def _capture(system_content, *a, **kw):
        seen_prompts.append(system_content)
        return next(seq)

    monkeypatch.setattr(graph, "_invoke_llm", _capture)
    graph.llm_reasoning_node(_state("summarize my notes"))

    first, retry = seen_prompts
    assert POISON in first and "TOOL REQUEST BLOCKED" not in first
    assert "TOOL REQUEST BLOCKED" in retry
    assert "mem_t" in retry and "search_vault" in retry     # named in the notice
    assert POISON not in retry                              # but its text is gone
    assert "Unrelated deployment facts" in retry            # other memories stay


def test_delete_request_is_blocked_with_no_memories_at_all(monkeypatch):
    _wire(monkeypatch, reply='TOOL: delete_memory("mem_1")')
    out = graph.llm_reasoning_node(_state("clean up"))
    assert out["needs_memory_search"] is False
    assert "blocked" in out["messages"][0].content


def test_blocked_reply_text_is_discarded(monkeypatch):
    _wire(monkeypatch, reply='Sure! Secret plan inside.\nTOOL: delete_memory("mem_1")')
    out = graph.llm_reasoning_node(_state("clean up"))
    assert "Secret plan" not in out["messages"][0].content


def test_exactly_one_retry_is_made(monkeypatch):
    calls = []
    monkeypatch.setattr(graph, "_mcp_page_context",
                        lambda **kw: PageContextResult(total_budget=1000))
    monkeypatch.setattr(graph, "_invoke_llm",
                        lambda *a, **kw: calls.append(1) or 'TOOL: delete_memory("m")')
    out = graph.llm_reasoning_node(_state("clean up"))
    assert len(calls) == 2 and "blocked" in out["messages"][0].content


def test_no_retry_when_the_first_attempt_is_fine(monkeypatch):
    calls = []
    monkeypatch.setattr(graph, "_mcp_page_context",
                        lambda **kw: PageContextResult(total_budget=1000))
    monkeypatch.setattr(graph, "_invoke_llm", lambda *a, **kw: calls.append(1) or "Here you go.")
    graph.llm_reasoning_node(_state("hello"))
    assert len(calls) == 1


def test_user_requested_search_is_allowed_even_with_unverified_context(monkeypatch):
    _wire(monkeypatch, paged=[_retrieved(f"Notes: {POISON}", signed=False)],
          reply=f'TOOL: search_vault("{POISON}")')
    out = graph.llm_reasoning_node(_state(f"find {POISON}"))
    assert out["needs_memory_search"] is True


def test_plain_answer_keeps_prior_context(monkeypatch):
    _wire(monkeypatch, reply="Here is your answer.")
    out = graph.llm_reasoning_node(_state(context_memories=[_entry()]))
    # context_memories persists for the whole session — a plain answer does not
    # erase what the LLM has already seen in earlier turns.
    assert out["context_memories"] == [_entry()] and out["needs_memory_search"] is False


# --------------------------------------------------------------------------
# Graph: memory_search_node keeps levels for the next LLM call
# --------------------------------------------------------------------------

def test_search_node_records_context_entries(monkeypatch):
    results = [_retrieved("Safe result text", "a"), _retrieved(f"Notes: {POISON}", "b", signed=False)]
    monkeypatch.setattr(graph, "_mcp_search_vault", lambda **kw: results)
    out = graph.memory_search_node(_state(search_query="q"))
    assert {e["id"]: e["level"] for e in out["context_memories"]} == {"a": "SEARCH_ONLY", "b": "READ_ONLY"}


def test_search_node_does_not_record_filtered_high_risk_memories(monkeypatch):
    monkeypatch.setattr(graph, "_mcp_search_vault",
                        lambda **kw: [_retrieved("bad one", "x", score=95.0)])
    assert graph.memory_search_node(_state(search_query="q"))["context_memories"] == []


def test_merge_context_deduplicates_by_id():
    merged = graph._merge_context([_entry("a"), _entry("b")], [_entry("b"), _entry("c")])
    assert [e["id"] for e in merged] == ["a", "b", "c"]


def test_merge_context_caps_to_the_most_recent():
    existing = [_entry(f"old{i}") for i in range(graph._MAX_CONTEXT_MEMORIES)]
    merged = graph._merge_context(existing, [_entry("new")])
    assert len(merged) == graph._MAX_CONTEXT_MEMORIES
    assert merged[-1]["id"] == "new" and merged[0]["id"] == "old1"  # old0 dropped


# --------------------------------------------------------------------------
# Graph: message-authority helpers (search cards are not system text)
# --------------------------------------------------------------------------

def _memory_card(text="card"):
    return HumanMessage(content=text, additional_kwargs={"amp_source": "memory"})


def test_is_user_authored_excludes_memory_cards():
    assert graph._is_user_authored(HumanMessage(content="hi")) is True
    assert graph._is_user_authored(_memory_card()) is False
    assert graph._is_user_authored(AIMessage(content="hi")) is False


def test_user_text_ignores_memory_cards():
    text = graph._user_text([HumanMessage(content="hello"), _memory_card(f"leak {POISON}")])
    assert text == "hello"


def test_latest_user_text_skips_a_trailing_memory_card():
    text = graph._latest_user_text([HumanMessage(content="first question"), _memory_card("card text")])
    assert text == "first question"


def test_latest_user_text_empty_when_only_memory_cards():
    assert graph._latest_user_text([_memory_card("card text")]) == ""


# --------------------------------------------------------------------------
# Graph: prompt-rendering and retry-notice helpers
# --------------------------------------------------------------------------

def test_render_memory_prompt_wraps_memories_and_builds_entries():
    system_content, entries = graph._render_memory_prompt(
        [_retrieved("Redis caching notes", "mem_r")], "s1")
    assert "<UNTRUSTED_MEMORY" in system_content and "Redis caching notes" in system_content
    assert [e["id"] for e in entries] == ["mem_r"]


def test_render_memory_prompt_empty_list_says_no_memories():
    system_content, entries = graph._render_memory_prompt([], "s1")
    assert "No prior memories retrieved" in system_content and entries == []


def test_build_retry_notice_lists_each_blocked_tool_and_reason():
    results = [
        tool_guard.ValidationResult(decision="REJECT", tool="search_vault",
                                    reasons=["needs SEARCH_ONLY"], suspected_sources=["mem_a"]),
        tool_guard.ValidationResult(decision="REJECT", tool="delete_memory",
                                    reasons=["mutating tool"], suspected_sources=[]),
    ]
    notice = graph._build_retry_notice(results)
    assert "search_vault" in notice and "needs SEARCH_ONLY" in notice
    assert "delete_memory" in notice and "mutating tool" in notice
    assert "mem_a" in notice
    assert "Do not repeat the blocked request" in notice


def test_build_retry_notice_without_sources_has_no_source_line():
    result = tool_guard.ValidationResult(decision="REJECT", tool="delete_memory",
                                         reasons=["mutating tool"], suspected_sources=[])
    assert "must not be used to justify" not in graph._build_retry_notice([result])


# --------------------------------------------------------------------------
# Graph: full compiled graph, search loop across nodes
# --------------------------------------------------------------------------

def _initial_state(user):
    return {
        "session_id": "e2e", "current_token_count": 0, "token_limit": 8192,
        "compaction_threshold": 0.80, "context_chunks": [], "oldest_chunk_type": "text",
        "compaction_in_progress": False, "total_compactions": 0,
        "messages": [HumanMessage(content=user)], "needs_memory_search": False,
        "search_query": "", "audit_requested": False, "context_memories": [],
    }


def _scripted_llm(monkeypatch, replies):
    """Serve the replies in order; returns the list of prompts the LLM was given."""
    seq = iter(replies)
    prompts = []

    def _next(system_content, *a, **kw):
        prompts.append(system_content)
        return next(seq)

    monkeypatch.setattr(graph, "_invoke_llm", _next)
    return prompts


def test_full_graph_benign_search_loop(monkeypatch):
    monkeypatch.setattr(graph, "_mcp_page_context",
                        lambda **kw: PageContextResult(total_budget=1000))
    monkeypatch.setattr(graph, "_mcp_search_vault",
                        lambda **kw: [_retrieved("The deployment pipeline uses blue green releases", "d1")])
    prompts = _scripted_llm(monkeypatch, ['TOOL: search_vault("deployment pipeline")',
                                          "It uses blue green releases."])

    final = graph.create_amp_graph().invoke(_initial_state("tell me about the deployment pipeline"))
    assert final["messages"][-1].content == "It uses blue green releases."
    assert len(prompts) == 2  # no retry
    # The search card is a marked HumanMessage, not a SystemMessage.
    assert any(isinstance(m, HumanMessage) and m.additional_kwargs.get("amp_source") == "memory"
               and "blue green" in m.content for m in final["messages"])
    # Context persists for the session, it is not cleared at turn end.
    assert [e["id"] for e in final["context_memories"]] == ["d1"]


def test_full_graph_blocks_injected_followup_call(monkeypatch):
    """Turn 1 is benign. Its search result holds an unverified poisoned memory, and
    on turn 2 the LLM (influenced by it) asks for a search built from that memory.
    The retry lets the turn still complete with a safe answer."""
    monkeypatch.setattr(graph, "_mcp_page_context",
                        lambda **kw: PageContextResult(total_budget=1000))
    monkeypatch.setattr(graph, "_mcp_search_vault",
                        lambda **kw: [_retrieved(f"Deployment note: {POISON}", "p1", signed=False)])
    searched = []
    real = graph.memory_search_node
    monkeypatch.setattr(graph, "memory_search_node",
                        lambda state: searched.append(state["search_query"]) or real(state))
    _scripted_llm(monkeypatch, [
        'TOOL: search_vault("deployment pipeline")',
        f'TOOL: search_vault("{POISON}")',
        "Understood — here is what I can verify without that memory.",
    ])

    final = graph.create_amp_graph().invoke(_initial_state("tell me about the deployment pipeline"))
    assert searched == ["deployment pipeline"]  # the second (poisoned) search never ran
    assert final["messages"][-1].content == "Understood — here is what I can verify without that memory."
    # p1 is still recorded, so a later turn leaning on it is caught too.
    assert [e["id"] for e in final["context_memories"]] == ["p1"]


def test_full_graph_gives_up_after_one_retry_when_the_block_persists(monkeypatch):
    """If the retry still asks for the same blocked thing, the turn ends with the
    safe notice rather than retrying forever."""
    monkeypatch.setattr(graph, "_mcp_page_context",
                        lambda **kw: PageContextResult(total_budget=1000))
    monkeypatch.setattr(graph, "_mcp_search_vault",
                        lambda **kw: [_retrieved(f"Deployment note: {POISON}", "p1", signed=False)])
    searched = []
    real = graph.memory_search_node
    monkeypatch.setattr(graph, "memory_search_node",
                        lambda state: searched.append(state["search_query"]) or real(state))
    poisoned_call = f'TOOL: search_vault("{POISON}")'
    prompts = _scripted_llm(monkeypatch, ['TOOL: search_vault("deployment pipeline")',
                                          poisoned_call, poisoned_call])

    final = graph.create_amp_graph().invoke(_initial_state("tell me about the deployment pipeline"))
    assert "blocked" in final["messages"][-1].content
    assert len(prompts) == 3          # turn 1, then attempt + exactly one retry on turn 2
    assert searched == ["deployment pipeline"]


def test_a_later_turn_is_checked_against_an_earlier_search_card(monkeypatch):
    """Session-long attribution: the poisoned card arrived turns ago, not this turn."""
    monkeypatch.setattr(graph, "_mcp_page_context",
                        lambda **kw: PageContextResult(total_budget=1000))
    _wire(monkeypatch, reply=f'TOOL: search_vault("{POISON}")')
    earlier = [_entry("old_card", "READ_ONLY", text=f"Deployment note: {POISON}")]
    out = graph.llm_reasoning_node(_state("what next?", context_memories=earlier))
    assert out["needs_memory_search"] is False and "blocked" in out["messages"][0].content


def test_compiled_graph_still_answers_plainly(monkeypatch):
    monkeypatch.setattr(graph, "_mcp_page_context",
                        lambda **kw: PageContextResult(total_budget=1000))
    final = graph.create_amp_graph().invoke(_initial_state("hello"))
    assert "[LLM Stub]" in final["messages"][-1].content


def test_system_prompt_mentions_the_monitor():
    prompt = graph._build_secure_system_prompt("ctx", "s1")
    assert "7. A monitor outside you checks every TOOL: line" in prompt
    assert graph.EXECUTION_LEVELS == ("READ_ONLY", "DATA_ONLY", "SEARCH_ONLY")


def test_guardrails_import_is_unchanged_by_phase4():
    assert guardrails.HIGH_RISK_THRESHOLD == 70.0
