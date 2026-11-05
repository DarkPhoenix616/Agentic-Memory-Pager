import os
import sys
import uuid
import tiktoken
from datetime import datetime, timezone
from typing import List, Optional
from mcp.server.fastmcp import FastMCP
import chromadb


encoder=tiktoken.get_encoding("cl100k_base") # gpt-4 one to count the number of tokens in code or dialogue. Isko hum token limit ke liye use karenge.
# ---------------------------------------------------------------------------
# Path setup — anchor storage to <project_root>/storage/chroma_db regardless
# of which directory this script is run from.
#
# This file lives at: <project_root>/server/amp_memory_server.py
# So project_root = parent of this file's directory.
# ---------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
# AMP_CHROMA_PATH overrides the location (tests use a temporary folder).
CHROMA_DB_PATH = os.environ.get("AMP_CHROMA_PATH") or os.path.join(PROJECT_ROOT, "storage", "chroma_db")

os.makedirs(CHROMA_DB_PATH, exist_ok=True)

# Make the project root importable so `from harness.guardrails import ...`
# works when this file is run directly (python server\amp_memory_server.py).
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from harness.guardrails import UnifiedMemory, RetrievedMemory, PageContextResult, SandboxResult, _scrub_secrets
from server.sandbox import run_code

SANDBOX_MAX_TIMEOUT = 10  # seconds; run_memory_code clamps the caller's timeout to 1..10

# Clients reach it at http://localhost:<port>/mcp (AMP_MCP_URL in the harness).
# AMP_MCP_PORT overrides the port; the host stays localhost, so only this machine can connect.
MCP_PORT = int(os.environ.get("AMP_MCP_PORT", "8001"))
mcp = FastMCP("amp-memory", host="localhost", port=MCP_PORT)

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection("amp_memory")


# ---------------------------------------------------------------------------
# UnifiedMemory <-> ChromaDB helpers
#
# Chroma stores a plain text "document" (what gets embedded and searched) plus
# flat metadata. A UnifiedMemory is a nested object, so:
#   - the searchable text is derived from the payload (_embedding_text)
#   - the full validated UnifiedMemory is saved as JSON in metadata["envelope"]
#     so it can be rebuilt exactly on the way out (_to_memory)
# ---------------------------------------------------------------------------
def _embedding_text(memory: UnifiedMemory) -> str:
    """Text that is embedded for semantic search, and counted for token budgets."""
    if memory.memory_type == "text":
        return memory.text_payload.factual_summary
    code = memory.code_payload
    parts = [code.file_path]
    if code.summary:
        parts.append(code.summary)
    parts.extend(symbol.signature for symbol in code.symbols)
    return "\n".join(parts)


def _to_memory(chroma_id: str, meta: Optional[dict]) -> Optional[UnifiedMemory]:
    """Rebuild a UnifiedMemory from a stored row. Rows saved before the
    UnifiedMemory format (no envelope) cannot be rebuilt and are skipped."""
    if not meta or "envelope" not in meta:
        return None
    memory = UnifiedMemory.model_validate_json(meta["envelope"])
    memory.vector_embedding_id = chroma_id
    return memory


@mcp.tool()
def store_memory(memory: UnifiedMemory) -> UnifiedMemory:
    """
    Store a memory envelope into long-term memory.

    Args:
        memory: A UnifiedMemory (session_id, memory_type, compaction_source,
            raw_token_count and the matching code_payload or text_payload).
            Invalid input is rejected by the model's validation.

    Returns:
        The stored UnifiedMemory, with vector_embedding_id set to the ID
        ChromaDB assigned.
    """
    # TODO [T1 / Teammate 1] PHASE 2 — re-score server-side; do not trust the caller.
    #   The envelope now carries risk_score / risk_level / provenance (see
    #   harness/guardrails.py). T4's graph nodes set them, but any client of this
    #   MCP tool could send risk_score=0 for a malicious memory. Before storing:
    #       payload = memory.text_payload or memory.code_payload
    #       score, _ = score_payload(payload, memory.provenance)   # from harness.guardrails
    #       memory.risk_score = max(memory.risk_score, score)
    #       memory.risk_level = risk_level_for_score(memory.risk_score)
    #   The envelope is saved as JSON below, so the new fields persist as-is.
    #   Also implement _llama_guard_score() in harness/guardrails.py (the model hook).
    # TODO [T1 / Teammate 1] PHASE 3 — RE-SIGN LAST. T4's graph nodes already sign
    #   each memory (harness/signing.py, HMAC-SHA256 keyed by AMP_SIGNING_KEY), and the
    #   signature covers risk_score and provenance. If you change any signed field
    #   above (e.g. raise risk_score), re-sign as the final step before storing:
    #       from harness.signing import sign_memory
    #       memory.signature = sign_memory(memory)
    #   Otherwise your change invalidates T4's signature and the memory is shown to
    #   the LLM as verified="false" / READ_ONLY. The server process needs the SAME
    #   AMP_SIGNING_KEY value as the harness. With no key set, sign_memory() returns None
    #   (memory stored unsigned, treated as READ_ONLY); nothing raises.
    memory_id = f"mem_{uuid.uuid4().hex[:8]}"
    memory.vector_embedding_id = memory_id

    collection.add(
        documents=[_embedding_text(memory)],
        ids=[memory_id],
        metadatas=[{
            "session_id": memory.session_id,
            "memory_type": memory.memory_type,
            "compaction_source": memory.compaction_source,
            "timestamp": memory.timestamp.isoformat(),
            "envelope": memory.model_dump_json(),
        }]
    )

    return memory


@mcp.tool()
def search_vault(query: str, top_k: int = 5, session_id: str = "") -> List[RetrievedMemory]:
    """
    Search long-term memory for content semantically related to the query.

    Args:
        query: The search query
        top_k: Maximum number of results to return (default: 5)
        session_id: Filter results to a specific session/workspace
    Returns:
        List of RetrievedMemory records, best match first. Each holds the
        stored UnifiedMemory (memory), its vector distance to the query
        (distance, smaller means closer) and its token cost (token_cost).
        Empty list if no matches found.
    """
    query = query.strip()
    if not query:
        return []

    # Guard against asking for more results than exist in the collection —
    # ChromaDB errors if n_results exceeds the total stored document count.
    count = collection.count()
    if count == 0:
        return []
    effective_k = min(top_k, count)

    where = {"session_id": session_id} if session_id else None
    results = collection.query(
        query_texts=[query],
        n_results=effective_k,
        where=where
    )

    matches = []
    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    for i in range(len(ids)):
        memory = _to_memory(ids[i], metadatas[i])
        if memory is not None:
            matches.append(RetrievedMemory(
                memory=memory,
                distance=distances[i],
                token_cost=len(encoder.encode(_embedding_text(memory)))
            ))

    return matches


@mcp.tool()
def delete_memory(memory_id: str) -> str:
    """
    Delete a specific memory record by ID.

    Args:
        memory_id: The ID of the memory to delete
        
    Returns:
        Confirmation message
    """
    # TODO [T1 / Teammate 1] PHASE 4 — this tool has no caller check. T4's harness
    #   (harness/tool_guard.py, Layer 4) rejects any delete_memory request that comes
    #   out of the LLM, but that only protects the harness path. Do NOT expose
    #   delete_memory / list_all_memories / store_memory to the LLM as callable MCP
    #   tools; they are for the harness and admins. Consider requiring an admin token
    #   argument here, and log each deletion (id, caller, time) on the server side.
    try:
        collection.delete(ids=[memory_id])
        return f"Deleted {memory_id}"
    except Exception as e:
        return f"Error deleting {memory_id}: {str(e)}"


@mcp.tool()
def list_all_memories() -> List[UnifiedMemory]:
    """
    List every memory currently stored (useful for debugging/inspection).

    Returns:
        List of all stored UnifiedMemory records.
    """
    all_data = collection.get()
    memories = []
    ids = all_data.get("ids", [])
    metadatas = all_data.get("metadatas", [])

    for i in range(len(ids)):
        memory = _to_memory(ids[i], metadatas[i])
        if memory is not None:
            memories.append(memory)

    return memories

@mcp.tool()
def page_context(query: str, total_budget: int = 1000, session_id: str = "", candidate_pool: int = 20) -> PageContextResult:
    """
    Retrieve context from long-term memory for a query, limited by a token budget.

    Args:
        query: What the agent needs to recall
        total_budget: Maximum total tokens the returned memories may use (default: 1000)
        session_id: Optional. If given, only use this session's memories
        candidate_pool: How many top matches to consider before applying the budget (default: 20)

    Returns:
        A PageContextResult with:
          memories: the selected RetrievedMemory records, best match first
          tokens_used: total token cost of the selected memories
          total_budget: the budget that was requested
          skipped: how many matches were left out because they did not fit
        A memory's token cost is the token count of its searchable text (the
        text summary, or for code the file path, summary and signatures).
    """
    query = query.strip()
    if not query or total_budget <= 0:
        return PageContextResult(total_budget=total_budget)

    # Reuse search_vault. Chroma returns candidates closest-first already.
    candidates = search_vault(query, top_k=candidate_pool, session_id=session_id)
    # TODO [T1 / Teammate 1] PHASE 2 — filter high-risk candidates here, BEFORE
    #   the token-budget loop below, so they do not consume budget:
    #       from harness.guardrails import HIGH_RISK_THRESHOLD
    #       candidates = [c for c in candidates if c.memory.risk_score < HIGH_RISK_THRESHOLD]
    #   Import the shared constant; do not hard-code 70. T4's graph already
    #   filters on its side (defense in depth); this keeps the server safe for
    #   other callers. Consider the same filter in search_vault().
    # TODO [T1] Rows stored before Phase 2 load with risk_score=0 ("safe"). Decide
    #   whether to backfill them with score_payload() via list_all_memories().
    if not candidates:
        return PageContextResult(total_budget=total_budget)

    selected = []
    used_budget = 0
    skipped = 0
    for candidate in candidates:
        tokens = candidate.token_cost

        if used_budget + tokens > total_budget:
            skipped += 1
            continue  # skipping current one, but a smaller match further down may still fit

        selected.append(candidate)
        used_budget += tokens

    return PageContextResult(
        memories=selected,
        tokens_used=used_budget,
        total_budget=total_budget,
        skipped=skipped
    )
    

def _sandbox_record(memory: UnifiedMemory) -> dict:
    """One memory as the simple record that run_memory_code's code sees."""
    text = memory.text_payload
    code = memory.code_payload
    return {
        "id": memory.vector_embedding_id or str(memory.id),
        "time": memory.timestamp.isoformat(),
        "kind": memory.memory_type,
        "text": _embedding_text(memory),
        "decisions": list(text.architectural_decisions) if text else [],
        "bugs": list(text.bugs_resolved) if text else [],
        "constraints": list(text.user_constraints) if text else [],
        "file": code.file_path if code else "",
    }


@mcp.tool()
def run_memory_code(code: str, session_id: str = "", timeout: int = 5) -> SandboxResult:
    """
    Run a short Python program over the stored memories, in the sandbox.

    For questions that need computing over many memories rather than reading
    a few: the latest decision about something, how many bugs were fixed,
    every constraint, memories in a time range.

    Args:
        code: Python that reads the list `memories` and print()s its answer.
            Each memory is a dict: id, time (ISO string), kind ("text" or
            "code"), text, decisions, bugs, constraints (lists of strings) and
            file (code file path or ""). They are ordered oldest first.
            Only json, re, math, statistics, collections, itertools, datetime
            and string can be imported; open, eval, exec and dunder names are
            blocked.
        session_id: Only this session's memories (all sessions if empty).
        timeout: Seconds before the program is stopped (1 to 10, default 5).

    Returns:
        A SandboxResult: ok, output (what it printed), error and
        memories_loaded. Output and error are secret-scrubbed and truncated.
    """
    memories = [m for m in list_all_memories() if not session_id or m.session_id == session_id]
    memories.sort(key=lambda m: m.timestamp)
    records = [_sandbox_record(m) for m in memories]
    seconds = max(1, min(SANDBOX_MAX_TIMEOUT, int(timeout)))
    return SandboxResult(**run_code(code, records, timeout=seconds, scrub=_scrub_secrets))


if __name__ == "__main__":
    print(f"ChromaDB persisting to: {CHROMA_DB_PATH}")
    print(f"MCP endpoint: http://localhost:{MCP_PORT}/mcp")
    mcp.run(transport="streamable-http")