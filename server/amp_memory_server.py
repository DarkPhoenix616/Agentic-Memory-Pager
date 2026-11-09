import json
import logging
import os
import sys
import threading
import uuid
import tiktoken
from datetime import datetime, timezone
from typing import List, Optional
from mcp.server.fastmcp import Context, FastMCP
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

from harness.guardrails import (
    HIGH_RISK_THRESHOLD, PageContextResult, RetrievedMemory, SandboxResult, UnifiedMemory, _scrub_secrets,
    risk_level_for_score, score_payload,
)
from harness.local_keys import ensure_admin_token, ensure_signing_key, is_admin
from harness.signing import SIGNING_KEY_ENV, sign_memory, verify_memory
from server.sandbox import run_code

logger = logging.getLogger(__name__)

SANDBOX_MAX_TIMEOUT = 10  # seconds; run_memory_code clamps the caller's timeout to 1..10
MAX_TOP_K = 100           # search_vault / page_context never consider more matches than this
HIGH_RISK_HEADROOM = 10   # extra matches fetched so that dropping high-risk ones does not shrink results

# Every delete_memory call (and every refused admin call) is appended here as one JSON line.
AUDIT_LOG_ENV = "AMP_AUDIT_LOG"
DEFAULT_AUDIT_LOG = os.path.join(PROJECT_ROOT, "storage", "amp_audit.log")
_audit_lock = threading.Lock()

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
    UnifiedMemory format (no envelope) cannot be rebuilt and are skipped.

    Rows saved before risk scoring existed have no risk_score in their JSON and
    would load as "safe" (0). They are scored here on the way out, so the
    high-risk filters below still apply to them; the score is not written back
    (python -m server.backfill does that)."""
    if not meta or "envelope" not in meta:
        return None
    stored = json.loads(meta["envelope"])
    memory = UnifiedMemory.model_validate(stored)
    memory.vector_embedding_id = chroma_id
    if "risk_score" not in stored:
        _rescore(memory)
    return memory


def _rescore(memory: UnifiedMemory) -> bool:
    """Score the payload here, on the server, and keep the higher of the stored
    and the server's score. True if the score went up."""
    payload = memory.text_payload if memory.memory_type == "text" else memory.code_payload
    score, _ = score_payload(payload, memory.provenance)
    if score <= memory.risk_score:
        return False
    memory.risk_score = score
    memory.risk_level = risk_level_for_score(score)     # no validate_assignment: keep the two in step by hand
    return True


def _is_high_risk(memory: UnifiedMemory) -> bool:
    return memory.risk_score >= HIGH_RISK_THRESHOLD


def _all_memories() -> List[UnifiedMemory]:
    """Every stored memory that can be rebuilt, high-risk ones included."""
    all_data = collection.get()
    memories = []
    for chroma_id, meta in zip(all_data.get("ids", []), all_data.get("metadatas", [])):
        memory = _to_memory(chroma_id, meta)
        if memory is not None:
            memories.append(memory)
    return memories


def _caller(ctx: Optional[Context]) -> str:
    """Who called a tool, for the audit log: the MCP client's name, or in-process."""
    if ctx is None:
        return "in-process"
    try:
        info = ctx.session.client_params.clientInfo
        return f"mcp:{info.name}" + (f"/{info.version}" if info.version else "")
    except Exception:
        return "mcp:unknown"


def _audit(action: str, memory_id: str, caller: str, result: str, session_id: str = "") -> None:
    """Append one line to the audit log (AMP_AUDIT_LOG, default storage/amp_audit.log)."""
    record = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "action": action,
        "memory_id": memory_id,
        "session_id": session_id,
        "caller": caller,
        "result": result,
    }
    logger.info("[Audit] %s", record)
    path = os.environ.get(AUDIT_LOG_ENV) or DEFAULT_AUDIT_LOG
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with _audit_lock, open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    except OSError as exc:
        logger.error("[Audit] Could not write %s: %s", path, exc)


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

    The caller's risk_score is not trusted: the server scores the payload
    itself (score_payload) and keeps the higher of the two. The signature
    covers risk_score, so when that raises the score a memory that arrived
    with a valid signature is signed again, as the last step. A memory that
    arrived unsigned, or with a signature that does not verify, is never
    signed here: storing through this tool cannot make a memory "verified".
    The server reads the same AMP_SIGNING_KEY as the chat app (harness/local_keys.py).
    """
    arrived_verified = verify_memory(memory)
    if memory.signature and not arrived_verified and os.environ.get(SIGNING_KEY_ENV):
        logger.warning("[store_memory] Memory %s arrived with a signature that does not verify; "
                       "it is stored as unverified.", memory.id)
    old_score = memory.risk_score
    if _rescore(memory):
        logger.info("[store_memory] Server score for memory %s is higher: %.1f -> %.1f (%s).",
                    memory.id, old_score, memory.risk_score, memory.risk_level)
        if arrived_verified:
            memory.signature = sign_memory(memory)      # re-sign last: every signed field is final now
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
        High-risk memories (risk_score >= HIGH_RISK_THRESHOLD) are never
        returned, whoever the caller is.
    """
    query = query.strip()
    top_k = min(int(top_k), MAX_TOP_K)
    if not query or top_k <= 0:
        return []

    # Guard against asking for more results than exist in the collection —
    # ChromaDB errors if n_results exceeds the total stored document count.
    count = collection.count()
    if count == 0:
        return []
    # A few extra matches, so dropping high-risk ones below does not shrink the list.
    effective_k = min(top_k + HIGH_RISK_HEADROOM, count)

    where = {"session_id": session_id} if session_id else None
    results = collection.query(
        query_texts=[query],
        n_results=effective_k,
        where=where
    )

    matches = []
    hidden = 0
    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    for i in range(len(ids)):
        memory = _to_memory(ids[i], metadatas[i])
        if memory is None:
            continue
        if _is_high_risk(memory):
            hidden += 1
            continue
        matches.append(RetrievedMemory(
            memory=memory,
            distance=distances[i],
            token_cost=len(encoder.encode(_embedding_text(memory)))
        ))
        if len(matches) == top_k:
            break

    if hidden:
        logger.info("[search_vault] Left out %d high-risk memories.", hidden)
    return matches


@mcp.tool()
def delete_memory(memory_id: str, admin_token: str = "", ctx: Optional[Context] = None) -> str:
    """
    Delete a specific memory record by ID. Admin only.

    Args:
        memory_id: The ID of the memory to delete
        admin_token: The vault's admin token (AMP_ADMIN_TOKEN, else the
            storage/.amp_admin_token file). Without it nothing is deleted.

    Returns:
        "Deleted <id>", or why nothing was deleted.

    Every call, allowed or refused, is written to the audit log with the time,
    the memory and session ID and the caller. The chat model can never reach
    this tool: Layer 4 (harness/tool_guard.py) rejects it, and only the
    harness's memory audit holds the token.
    """
    caller = _caller(ctx)
    if not is_admin(admin_token):
        _audit("delete_memory", memory_id, caller, "refused: missing or wrong admin token")
        return f"Refused: deleting {memory_id} needs the admin token."
    try:
        found = collection.get(ids=[memory_id])
        if not found.get("ids"):
            _audit("delete_memory", memory_id, caller, "not found")
            return f"Not found: {memory_id}"
        session_id = ((found.get("metadatas") or [None])[0] or {}).get("session_id", "")
        collection.delete(ids=[memory_id])
    except Exception as e:
        _audit("delete_memory", memory_id, caller, f"error: {e}")
        return f"Error deleting {memory_id}: {str(e)}"
    _audit("delete_memory", memory_id, caller, "deleted", session_id=session_id)
    return f"Deleted {memory_id}"


@mcp.tool()
def list_all_memories(admin_token: str = "", ctx: Optional[Context] = None) -> List[UnifiedMemory]:
    """
    List every memory currently stored, from every session (for audits and
    the chat app's vault view). Admin only.

    Args:
        admin_token: The vault's admin token (AMP_ADMIN_TOKEN, else the
            storage/.amp_admin_token file).

    Returns:
        List of all stored UnifiedMemory records, high-risk ones included.
    """
    if not is_admin(admin_token):
        _audit("list_all_memories", "", _caller(ctx), "refused: missing or wrong admin token")
        raise PermissionError("list_all_memories needs the admin token.")
    return _all_memories()

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
        High-risk memories are left out before the budget is applied, so they
        never use up budget (search_vault drops them).
    """
    query = query.strip()
    if not query or total_budget <= 0:
        return PageContextResult(total_budget=total_budget)

    # Reuse search_vault: closest-first, high-risk memories already removed
    # (also rows stored before risk scoring, which _to_memory scores on the way out).
    candidates = search_vault(query, top_k=candidate_pool, session_id=session_id)
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


def _use_loaded_memories(code: str) -> Optional[str]:
    """The code with any top-level `memories = ...` line removed, or None if it
    assigns `memories` in some other way (a loop variable, inside a function).

    Small models often start with a placeholder (`memories = [...]`) or re-type
    the memories they saw in their prompt. Dropping that line makes the program
    read the real, complete list that the sandbox loads."""
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code                       # run_code reports the syntax error
    drop = set()
    for node in tree.body:                # top level: `memories = ...`, nothing else assigned
        if (isinstance(node, (ast.Assign, ast.AnnAssign))
                and all(isinstance(t, ast.Name) and t.id == "memories"
                        for t in (node.targets if isinstance(node, ast.Assign) else [node.target]))):
            drop.update(range(node.lineno, node.end_lineno + 1))
    for node in ast.walk(tree):           # any other assignment to `memories` is refused
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.For, ast.comprehension, ast.With)):
            targets = [getattr(node, "target", None)] if not isinstance(node, ast.With) else \
                [item.optional_vars for item in node.items]
        else:
            continue
        for target in targets:
            if target is None or getattr(node, "lineno", 0) in drop:
                continue
            if any(isinstance(n, ast.Name) and n.id == "memories" for n in ast.walk(target)):
                return None
    lines = code.splitlines()
    return "\n".join(line for number, line in enumerate(lines, 1) if number not in drop)


@mcp.tool()
def run_memory_code(code: str, session_id: str = "", timeout: int = 5) -> SandboxResult:
    """
    Run a short Python program over the stored memories, in the sandbox.

    For questions that need computing over many memories rather than reading
    a few: the latest decision about something, how many bugs were fixed,
    every constraint, memories in a time range.

    Args:
        code: Python that reads the list `memories` (already loaded; a top-level
            `memories = ...` line is ignored, other assignments are refused)
            and print()s its answer.
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
        High-risk memories are not loaded, like in search_vault: the output
        goes back to the model, so it must not carry their text.
    """
    code = _use_loaded_memories(code)
    if code is None:
        return SandboxResult(ok=False, error="Blocked: `memories` is already loaded with the stored "
                             "memories; read it, do not assign to it.")
    memories = [m for m in _all_memories()
                if (not session_id or m.session_id == session_id) and not _is_high_risk(m)]
    memories.sort(key=lambda m: m.timestamp)
    records = [_sandbox_record(m) for m in memories]
    seconds = max(1, min(SANDBOX_MAX_TIMEOUT, int(timeout)))
    return SandboxResult(**run_code(code, records, timeout=seconds, scrub=_scrub_secrets))


if __name__ == "__main__":
    # The same two secrets as the chat app (same env vars, else the same files under storage/),
    # so memories it signed verify here and its admin calls are accepted.
    print(f"ChromaDB persisting to: {CHROMA_DB_PATH}")
    print(f"Signing key: {ensure_signing_key()}; admin token: {ensure_admin_token()}")
    print(f"MCP endpoint: http://localhost:{MCP_PORT}/mcp")
    mcp.run(transport="streamable-http")