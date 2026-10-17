"""
harness/graph.py — AMP LangGraph Orchestrator

Wires 8 nodes into a StateGraph that monitors the context window
and routes to compaction or LLM reasoning based on token usage.

Nodes:
    1. token_monitor_node       — Count tokens; route to compact or reason.
    2. compaction_dispatch_node — Check oldest chunk type; route to AST or RLM.
    3. ast_compactor_node       — [Teammate 2] Compact code chunks via AST.
    4. rlm_summarizer_node      — [Teammate 3] Summarize text chunks via RLM.
    5. post_compaction_node     — Reset flags, increment counter, loop back.
    6. llm_reasoning_node       — Core LLM call with paged memory injection.
    7. memory_search_node       — Targeted vault search on LLM request.
    8. memory_audit_node        — Periodic stale memory pruning.

Usage:
    from harness.graph import create_amp_graph
    graph = create_amp_graph()
    result = graph.invoke(initial_state)
"""

from __future__ import annotations

import os
import re
import sys
import json
import logging
from typing import Any, Dict, List, Literal, Annotated

import tiktoken
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# Make project root importable from any working directory.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from harness.guardrails import (
    UnifiedMemory,
    CodeMemorySchema,
    TextMemorySchema,
    RetrievedMemory,
    PageContextResult,
)

logger = logging.getLogger(__name__)

# Shared tokenizer — cl100k_base matches the MCP server.
_encoder = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    """Count tokens in a string using cl100k_base."""
    return len(_encoder.encode(text))


# ---------------------------------------------------------------------------
# AMP Graph State
# Flat TypedDict shared across all nodes. Fields are overwritten directly
# except `messages`, which uses add_messages to append instead of replace.
# ---------------------------------------------------------------------------

class AMPState(TypedDict):
    """Shared state for the AMP LangGraph graph."""
    session_id: str

    # Token budget
    current_token_count: int       # live token count in the active context
    token_limit: int               # model max context (e.g. 128 000)
    compaction_threshold: float    # trigger fraction, default 0.80

    # Active working memory — each dict: {role, content, type: "code"|"text"}
    context_chunks: List[dict]

    # Compaction routing
    oldest_chunk_type: Literal["code", "text"]  # selects AST vs RLM engine
    compaction_in_progress: bool                 # re-entrancy lock
    total_compactions: int                        # telemetry counter

    # LangGraph message list (add_messages reducer appends automatically)
    messages: Annotated[List[BaseMessage], add_messages]

    # Routing flags set by nodes
    needs_memory_search: bool  # True when LLM requests a vault search
    search_query: str          # query string for memory_search_node
    audit_requested: bool      # True to trigger memory_audit_node


# ---------------------------------------------------------------------------
# MCP helpers — thin wrappers around FastMCP server functions.
# Swap direct imports for HTTP client calls when running remotely.
# ---------------------------------------------------------------------------

def _mcp_store_memory(memory: UnifiedMemory) -> UnifiedMemory:
    """Persist a memory to ChromaDB."""
    from server.amp_memory_server import store_memory
    return store_memory(memory)


def _mcp_page_context(query: str, total_budget: int, session_id: str) -> PageContextResult:
    """Retrieve relevant memories within a token budget."""
    from server.amp_memory_server import page_context
    return page_context(query=query, total_budget=total_budget, session_id=session_id)


def _mcp_search_vault(query: str, top_k: int, session_id: str) -> List[RetrievedMemory]:
    """Semantic search across the vault."""
    from server.amp_memory_server import search_vault
    return search_vault(query=query, top_k=top_k, session_id=session_id)


def _mcp_list_all_memories() -> List[UnifiedMemory]:
    """Fetch all memory records."""
    from server.amp_memory_server import list_all_memories
    return list_all_memories()


def _mcp_delete_memory(memory_id: str) -> str:
    """Delete a memory record by ID."""
    from server.amp_memory_server import delete_memory
    return delete_memory(memory_id=memory_id)


# ---------------------------------------------------------------------------
# Node 1: Token Monitor
# ---------------------------------------------------------------------------

def token_monitor_node(state: AMPState) -> Dict[str, Any]:
    """Count all tokens in messages + context_chunks and update current_token_count."""
    message_tokens = sum(
        _count_tokens(f"{msg.type}: {msg.content}")
        for msg in state["messages"]
    )
    chunk_tokens = sum(
        _count_tokens(json.dumps(chunk))
        for chunk in state["context_chunks"]
    )
    total = message_tokens + chunk_tokens
    logger.info(
        "[TokenMonitor] tokens=%d  limit=%d  usage=%.1f%%",
        total, state["token_limit"], total / state["token_limit"] * 100,
    )
    return {"current_token_count": total}


def _route_from_token_monitor(state: AMPState) -> str:
    """Route to compaction if usage >= threshold, else to LLM reasoning."""
    ratio = state["current_token_count"] / state["token_limit"]
    if ratio >= state["compaction_threshold"]:
        logger.info("[TokenMonitor] %.1f%% full — routing to compaction.", ratio * 100)
        return "compaction_dispatch_node"
    return "llm_reasoning_node"


# ---------------------------------------------------------------------------
# Node 2: Compaction Dispatch
# ---------------------------------------------------------------------------

def compaction_dispatch_node(state: AMPState) -> Dict[str, Any]:
    """
    Set the compaction lock and detect the oldest chunk type.
    "code" → routes to ASTCompactor; "text" → routes to RLMSummarizer.
    """
    chunks = state["context_chunks"]
    oldest_type: Literal["code", "text"] = chunks[0].get("type", "text") if chunks else "text"
    logger.info("[CompactionDispatch] Oldest chunk type: '%s'", oldest_type)
    return {"compaction_in_progress": True, "oldest_chunk_type": oldest_type}


def _route_from_compaction_dispatch(state: AMPState) -> str:
    """Route to the correct compaction engine based on oldest chunk type."""
    return "ast_compactor_node" if state["oldest_chunk_type"] == "code" else "rlm_summarizer_node"


# ---------------------------------------------------------------------------
# Node 3: AST Compactor  [Teammate 2]
# ---------------------------------------------------------------------------

def ast_compactor_node(state: AMPState) -> Dict[str, Any]:
    """
    Compact the oldest code chunk into a CodeMemorySchema and store it.
    Evicts the raw chunk after successful storage. [Teammate 2 owns AST logic.]
    """
    chunks = state["context_chunks"]
    if not chunks:
        logger.warning("[ASTCompactor] No chunks — skipping.")
        return {}

    oldest_chunk = chunks[0]
    raw_content: str = oldest_chunk.get("content", "")
    raw_token_count = _count_tokens(raw_content)

    # TODO [Teammate 2]: Replace stub with tree-sitter AST parsing.
    # Input:  raw_content (str)
    # Output: code_schema (CodeMemorySchema) with imports, symbols, dependencies
    code_schema = CodeMemorySchema(
        file_path=oldest_chunk.get("file_path", "unknown/file.py"),
        language=oldest_chunk.get("language", "python"),
        imports=[],
        symbols=[],
        dependencies=[],
        summary="[AST stub — Teammate 2 will replace]",
        original_token_count=raw_token_count,
    )
    # END TODO [Teammate 2]

    # Store first, then evict (write-before-delete to avoid data loss on failure)
    envelope = UnifiedMemory(
        session_id=state["session_id"],
        memory_type="code",
        compaction_source="ast",
        code_payload=code_schema,
        raw_token_count=raw_token_count,
    )
    stored = _mcp_store_memory(envelope)
    logger.info("[ASTCompactor] Stored: %s | Evicted %d tokens.", stored.vector_embedding_id, raw_token_count)

    return {
        "context_chunks": chunks[1:],
        "current_token_count": max(0, state["current_token_count"] - raw_token_count),
    }


# ---------------------------------------------------------------------------
# Node 4: RLM Summarizer  [Teammate 3]
# ---------------------------------------------------------------------------

def rlm_summarizer_node(state: AMPState) -> Dict[str, Any]:
    """
    Summarize the oldest text chunk into a TextMemorySchema and store it.
    Evicts the raw chunk after successful storage. [Teammate 3 owns RLM logic.]
    """
    chunks = state["context_chunks"]
    if not chunks:
        logger.warning("[RLMSummarizer] No chunks — skipping.")
        return {}

    oldest_chunk = chunks[0]
    raw_content: str = oldest_chunk.get("content", "")
    raw_token_count = _count_tokens(raw_content)

    # TODO [Teammate 3]: Replace stub with multi-pass RLM summarization.
    # Input:  raw_content (str)
    # Output: text_schema (TextMemorySchema) with factual_summary, decisions, bugs, constraints
    text_schema = TextMemorySchema(
        chunk_index=0,
        total_chunks=1,
        timeframe="[Teammate 3 will set]",
        factual_summary="[RLM stub — Teammate 3 will replace]",
        architectural_decisions=[],
        bugs_resolved=[],
        user_constraints=[],
        original_token_count=raw_token_count,
    )
    # END TODO [Teammate 3]

    envelope = UnifiedMemory(
        session_id=state["session_id"],
        memory_type="text",
        compaction_source="rlm",
        text_payload=text_schema,
        raw_token_count=raw_token_count,
    )
    stored = _mcp_store_memory(envelope)
    logger.info("[RLMSummarizer] Stored: %s | Evicted %d tokens.", stored.vector_embedding_id, raw_token_count)

    return {
        "context_chunks": chunks[1:],
        "current_token_count": max(0, state["current_token_count"] - raw_token_count),
    }


# ---------------------------------------------------------------------------
# Node 5: Post Compaction
# ---------------------------------------------------------------------------

def post_compaction_node(state: AMPState) -> Dict[str, Any]:
    """Release the compaction lock and loop back to re-check the token budget."""
    new_count = state["total_compactions"] + 1
    logger.info("[PostCompaction] Cycle #%d complete.", new_count)
    return {"compaction_in_progress": False, "total_compactions": new_count}


# ---------------------------------------------------------------------------
# Node 6: LLM Reasoning
# ---------------------------------------------------------------------------

def llm_reasoning_node(state: AMPState) -> Dict[str, Any]:
    """
    Core reasoning node. Injects paged memories into the prompt, calls the LLM,
    and either answers the user or requests a vault search.
    """
    # Find the latest human message to use as the memory retrieval query
    last_human_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_human_msg = msg.content
            break

    # Pull relevant memories within a token budget (transient — not stored in context_chunks)
    paged: PageContextResult = _mcp_page_context(
        query=last_human_msg or "general context",
        total_budget=1000,
        session_id=state["session_id"],
    )

    memory_lines: List[str] = []
    for rm in paged.memories:
        if rm.memory.memory_type == "text" and rm.memory.text_payload:
            memory_lines.append(f"[Memory] {rm.memory.text_payload.factual_summary}")
        elif rm.memory.memory_type == "code" and rm.memory.code_payload:
            sigs = ", ".join(s.signature for s in rm.memory.code_payload.symbols[:5])
            memory_lines.append(f"[Code: {rm.memory.code_payload.file_path}] {sigs}")

    retrieved_context = "\n".join(memory_lines) if memory_lines else "No prior memories retrieved."

    system_content = (
        "You are an intelligent coding assistant with long-term memory.\n"
        "Relevant memories from previous sessions:\n\n"
        f"{retrieved_context}\n\n"
        "To recall something specific, respond with: TOOL: search_vault(\"<query>\")\n"
        "Otherwise, answer the user directly."
    )

    # LLM CALL — replace this stub with your provider:
    #   from langchain_community.chat_models import ChatOllama
    #   llm = ChatOllama(model="llama3")
    #   reply_text = llm.invoke([SystemMessage(system_content)] + state["messages"]).content
    reply_text = (
        f"[LLM Stub] Responding to: '{last_human_msg}'. "
        f"Retrieved {len(paged.memories)} memory/memories from vault."
    )

    if reply_text.strip().startswith("TOOL: search_vault"):
        match = re.search(r'search_vault\("(.+?)"\)', reply_text)
        query = match.group(1) if match else last_human_msg
        logger.info("[LLMReasoning] Vault search requested: '%s'", query)
        return {"needs_memory_search": True, "search_query": query}

    logger.info("[LLMReasoning] Answer generated.")
    return {"messages": [AIMessage(content=reply_text)], "needs_memory_search": False, "search_query": ""}


def _route_from_llm_reasoning(state: AMPState) -> str:
    """Route to vault search if requested, else end the turn."""
    return "memory_search_node" if state.get("needs_memory_search", False) else END


# ---------------------------------------------------------------------------
# Node 7: Memory Search
# ---------------------------------------------------------------------------

def memory_search_node(state: AMPState) -> Dict[str, Any]:
    """
    Run a targeted vault search when the LLM needs specific recall.
    Injects results back as a system message for the next LLM call.
    """
    query = state.get("search_query", "")
    if not query:
        logger.warning("[MemorySearch] No query — skipping.")
        return {"needs_memory_search": False}

    logger.info("[MemorySearch] Searching: '%s'", query)
    results: List[RetrievedMemory] = _mcp_search_vault(
        query=query, top_k=5, session_id=state["session_id"]
    )

    if not results:
        context_card = f"[Vault Search: '{query}'] No results."
    else:
        lines = [f"[Vault Search: '{query}']"]
        for i, rm in enumerate(results, 1):
            if rm.memory.memory_type == "text" and rm.memory.text_payload:
                lines.append(f"  {i}. {rm.memory.text_payload.factual_summary} (dist={rm.distance:.3f})")
            elif rm.memory.memory_type == "code" and rm.memory.code_payload:
                lines.append(f"  {i}. [{rm.memory.code_payload.file_path}] {len(rm.memory.code_payload.symbols)} symbols (dist={rm.distance:.3f})")
        context_card = "\n".join(lines)

    logger.info("[MemorySearch] Found %d result(s).", len(results))
    return {"messages": [SystemMessage(content=context_card)], "needs_memory_search": False, "search_query": ""}


# ---------------------------------------------------------------------------
# Node 8: Memory Audit
# ---------------------------------------------------------------------------

def memory_audit_node(state: AMPState) -> Dict[str, Any]:
    """
    Validate all stored memories, delete schema-invalid records,
    and prune duplicate code memories (keep newest per file path).
    """
    logger.info("[MemoryAudit] Starting audit...")
    all_memories: List[UnifiedMemory] = _mcp_list_all_memories()
    deleted = 0
    seen_file_paths: Dict[str, str] = {}  # file_path → memory_id

    for memory in all_memories:
        mem_id = memory.vector_embedding_id

        # Delete records that fail schema validation
        try:
            UnifiedMemory.model_validate(memory.model_dump())
        except Exception as exc:
            logger.warning("[MemoryAudit] Invalid record %s — deleting. %s", mem_id, exc)
            _mcp_delete_memory(mem_id)
            deleted += 1
            continue

        # For code memories, keep only the newest per file path
        if memory.memory_type == "code" and memory.code_payload:
            fpath = memory.code_payload.file_path
            if fpath in seen_file_paths:
                logger.info("[MemoryAudit] Duplicate for '%s' — removing older %s.", fpath, seen_file_paths[fpath])
                _mcp_delete_memory(seen_file_paths[fpath])
                deleted += 1
            seen_file_paths[fpath] = mem_id

    logger.info("[MemoryAudit] Done. Records: %d | Deleted: %d", len(all_memories), deleted)
    return {"audit_requested": False}


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------

def create_amp_graph(checkpointer=None) -> Any:
    """
    Compile the AMP LangGraph StateGraph.

    Topology:
        START → token_monitor_node
        token_monitor_node → compaction_dispatch_node  (usage >= 80%)
        token_monitor_node → llm_reasoning_node        (usage < 80%)
        compaction_dispatch_node → ast_compactor_node  (oldest = code)
        compaction_dispatch_node → rlm_summarizer_node (oldest = text)
        ast/rlm → post_compaction_node → token_monitor_node (loop)
        llm_reasoning_node → memory_search_node → llm_reasoning_node (loop)
        llm_reasoning_node → END
        memory_audit_node  → END  (admin-triggered, not in main loop)
    """
    builder = StateGraph(AMPState)

    builder.add_node("token_monitor_node",       token_monitor_node)
    builder.add_node("compaction_dispatch_node", compaction_dispatch_node)
    builder.add_node("ast_compactor_node",       ast_compactor_node)
    builder.add_node("rlm_summarizer_node",      rlm_summarizer_node)
    builder.add_node("post_compaction_node",     post_compaction_node)
    builder.add_node("llm_reasoning_node",       llm_reasoning_node)
    builder.add_node("memory_search_node",       memory_search_node)
    builder.add_node("memory_audit_node",        memory_audit_node)

    builder.add_edge(START, "token_monitor_node")

    builder.add_conditional_edges(
        "token_monitor_node", _route_from_token_monitor,
        {"compaction_dispatch_node": "compaction_dispatch_node",
         "llm_reasoning_node": "llm_reasoning_node"},
    )
    builder.add_conditional_edges(
        "compaction_dispatch_node", _route_from_compaction_dispatch,
        {"ast_compactor_node": "ast_compactor_node",
         "rlm_summarizer_node": "rlm_summarizer_node"},
    )

    builder.add_edge("ast_compactor_node",   "post_compaction_node")
    builder.add_edge("rlm_summarizer_node",  "post_compaction_node")
    builder.add_edge("post_compaction_node", "token_monitor_node")

    builder.add_conditional_edges(
        "llm_reasoning_node", _route_from_llm_reasoning,
        {"memory_search_node": "memory_search_node", END: END},
    )
    builder.add_edge("memory_search_node", "llm_reasoning_node")
    builder.add_edge("memory_audit_node",  END)

    compile_kwargs: Dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    return builder.compile(**compile_kwargs)


# ---------------------------------------------------------------------------
# Smoke Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    print("=" * 60)
    print("AMP LangGraph Harness — Smoke Test")
    print("=" * 60)

    graph = create_amp_graph()

    initial_state: AMPState = {
        "session_id":             "smoke-test-01",
        "current_token_count":    0,
        "token_limit":            8192,
        "compaction_threshold":   0.80,
        "context_chunks":         [],
        "oldest_chunk_type":      "text",
        "compaction_in_progress": False,
        "total_compactions":      0,
        "messages":               [HumanMessage(content="What is the AMP memory pager?")],
        "needs_memory_search":    False,
        "search_query":           "",
        "audit_requested":        False,
    }

    print("\n[Input] User: What is the AMP memory pager?\n")
    final_state = graph.invoke(initial_state)
    print("\n[Output] Assistant:", final_state["messages"][-1].content)
    print(f"\n[Stats] Compactions: {final_state['total_compactions']} | Tokens: {final_state['current_token_count']}")
    print("\n✅ Graph compiled and ran successfully.")
