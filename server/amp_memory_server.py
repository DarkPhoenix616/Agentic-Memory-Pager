import json
import logging
import os
import re
import sys
import threading
import uuid
import tiktoken
from datetime import datetime, timezone
from typing import List, Optional
from mcp.server.fastmcp import Context, FastMCP
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


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
    HIGH_RISK_THRESHOLD, MemoryInfo, PageContextResult, RetrievedMemory, SandboxResult, SessionInfo,
    UnifiedMemory, _scrub_secrets, risk_level_for_score, score_payload,
)
from harness import conflicts
from harness.local_keys import ensure_admin_token, ensure_signing_key, is_admin
from harness.signing import SIGNING_KEY_ENV, sign_memory, verify_memory
from server import session_crypto
from server.sandbox import run_code
from server.session_crypto import SessionKeyError

logger = logging.getLogger(__name__)

SANDBOX_MAX_TIMEOUT = 10  # seconds; run_memory_code clamps the caller's timeout to 1..10
MAX_TOP_K = 100           # search_vault / page_context never consider more matches than this
HIGH_RISK_HEADROOM = 10   # extra matches fetched so that dropping high-risk ones does not shrink results
CONFLICT_CANDIDATES = 3   # a new text memory is checked against its 3 closest older ones...
CONFLICT_DISTANCE = 1.2   # ...that are this close (squared L2; same topic ~0.4, unrelated ~1.75)
MAX_CONFLICTS = 20
PAGE_MAX_DISTANCE_ENV = "AMP_PAGE_MAX_DISTANCE"
# page_context's relevance cut-off (report section 5o, report_eval/relevance_cutoff): in the 16-fact
# chat the memory a question needed was at most 1.31 away, while general questions ("Explain the
# CAP theorem") paged in ~650 tokens of unrelated memories. Questions about the user or the chat
# itself ("What do you remember about me?") get no cut-off: their memories are 1.2-1.9 away.
DEFAULT_PAGE_MAX_DISTANCE = 1.4
_ABOUT_THE_CHAT = re.compile(
    r"\b(i|me|my|mine|we|us|our|ours|remember|remind|recall|earlier|before|previous(?:ly)?|told|said|say|"
    r"mentioned|discussed|talked|chat|conversation|again)\b", re.IGNORECASE)

# Every delete_memory call (and every refused admin call) is appended here as one JSON line.
AUDIT_LOG_ENV = "AMP_AUDIT_LOG"
DEFAULT_AUDIT_LOG = os.path.join(PROJECT_ROOT, "storage", "amp_audit.log")
_audit_lock = threading.Lock()

# Clients reach it at http://localhost:<port>/mcp (AMP_MCP_URL in the harness).
# AMP_MCP_PORT overrides the port; the host stays localhost, so only this machine can connect.
MCP_PORT = int(os.environ.get("AMP_MCP_PORT", "8001"))
mcp = FastMCP("amp-memory", host="localhost", port=MCP_PORT)

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
# ChromaDB's default embedding model (all-MiniLM-L6-v2). The server embeds the
# plain text itself and stores only the vectors: no readable text is stored.
EMBED = DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection("amp_memory", embedding_function=EMBED)


# ---------------------------------------------------------------------------
# UnifiedMemory <-> ChromaDB helpers
#
# One ChromaDB row per memory:
#   - embedding   the vector of the memory's searchable text (_embedding_text);
#                 the text itself is not stored
#   - metadata    session_id, memory_type, compaction_source, timestamp and
#                 risk_score in the clear (for filtering and the admin list),
#                 plus encrypted=True, key_check (fingerprint of the session
#                 key) and envelope_enc: the full UnifiedMemory as JSON,
#                 encrypted with the session's key (server/session_crypto.py),
#                 and conflicts_with: the IDs of memories it contradicts,
#                 comma-separated (IDs only, never what they say)
# Rows stored before encryption keep a plain "envelope" and are not served
# until python -m server.backfill --apply encrypts them.
# ---------------------------------------------------------------------------
def _embedding_text(memory: UnifiedMemory) -> str:
    """Text that is embedded for semantic search, and counted for token budgets.

    The code graph (which function calls which) is not part of it: its lines
    made code memories 35% bigger and did not help search, because the
    embedding model reads only the first 256 word pieces and most files are
    longer (report section 5p). The graph stays in the memory, and
    run_memory_code's records carry it: in `calls` and in their `text`
    (_call_lines), where it costs no paging budget."""
    if memory.memory_type == "text":
        return memory.text_payload.factual_summary
    code = memory.code_payload
    parts = [code.file_path]
    if code.summary:
        parts.append(code.summary)
    parts.extend(symbol.signature for symbol in code.symbols)
    return "\n".join(parts)


def _call_lines(memory: UnifiedMemory) -> List[str]:
    """The code graph as text: "checkout calls load_cart, charge" (code memories only)."""
    code = memory.code_payload
    if memory.memory_type != "code" or code is None:
        return []
    return [f"{s.name} calls {', '.join(s.calls)}" for s in code.symbols if s.calls]


def _to_memory(chroma_id: str, envelope_json: str) -> UnifiedMemory:
    """Rebuild a UnifiedMemory from its stored JSON.

    Memories saved before risk scoring existed have no risk_score in their JSON
    and would load as "safe" (0). They are scored here on the way out, so the
    high-risk filters still apply to them; the score is not written back
    (python -m server.backfill does that)."""
    stored = json.loads(envelope_json)
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


def _session_where(session_id: str) -> dict:
    """ChromaDB filter for one session's encrypted rows."""
    return {"$and": [{"session_id": session_id}, {"encrypted": True}]}


def _authorize(session_id: str, session_key: str, action: str, ctx: Optional[Context]) -> str:
    """Check that the caller holds this session's key; return the key check.

    The first memory stored in a session fixes its key. A refused call is
    written to the audit log and raises SessionKeyError (a PermissionError)."""
    caller = _caller(ctx)
    if not session_id:
        _audit(action, "", caller, "refused: no session_id")
        raise SessionKeyError(f"{action} works on one session: give its session_id and session_key.")
    try:
        check = session_crypto.key_check(session_key, session_id)
    except SessionKeyError:
        _audit(action, "", caller, "refused: missing or malformed session key", session_id)
        raise SessionKeyError(f"{action} needs the session key of {session_id}.") from None
    existing = collection.get(where=_session_where(session_id), limit=1, include=["metadatas"])
    rows = existing.get("metadatas") or []
    if rows and not session_crypto.same_key_check(str(rows[0].get("key_check", "")), check):
        _audit(action, "", caller, "refused: wrong session key", session_id)
        raise SessionKeyError(f"Wrong session key for {session_id}.")
    return check


def _open_row(chroma_id: str, meta: Optional[dict], session_key: str) -> Optional[UnifiedMemory]:
    """Decrypt one row of the caller's session; None if it cannot be read."""
    if not meta or not meta.get("encrypted"):
        return None
    try:
        envelope = session_crypto.decrypt(meta["envelope_enc"], session_key, meta["session_id"], chroma_id)
    except (SessionKeyError, KeyError, ValueError) as exc:
        logger.warning("[Vault] Memory %s could not be decrypted (%s); skipped.", chroma_id, exc)
        return None
    return _to_memory(chroma_id, envelope)


def _session_memories(session_id: str, session_key: str) -> List[UnifiedMemory]:
    """Every readable memory of one session, high-risk ones included (call _authorize first)."""
    rows = collection.get(where=_session_where(session_id), include=["metadatas"])
    memories = []
    for chroma_id, meta in zip(rows.get("ids", []), rows.get("metadatas", [])):
        memory = _open_row(chroma_id, meta, session_key)
        if memory is not None:
            memories.append(memory)
    return memories


def _row_metadata(memory: UnifiedMemory, memory_id: str, session_key: str, check: str) -> dict:
    """The ChromaDB metadata of one encrypted memory."""
    return {
        "session_id": memory.session_id,
        "memory_type": memory.memory_type,
        "compaction_source": memory.compaction_source,
        "timestamp": memory.timestamp.isoformat(),
        "risk_score": float(memory.risk_score),
        "encrypted": True,
        "key_check": check,
        "envelope_enc": session_crypto.encrypt(memory.model_dump_json(), session_key, memory.session_id, memory_id),
    }


def _conflict_ids(meta: Optional[dict]) -> List[str]:
    return [i for i in str((meta or {}).get("conflicts_with", "")).split(",") if i]


def _find_conflicts(memory: UnifiedMemory, session_key: str) -> List[str]:
    """IDs of the older memories of the same session that this new text memory
    contradicts: its closest few (CONFLICT_CANDIDATES within CONFLICT_DISTANCE)
    are each checked by the conflict model (harness/conflicts.py). Code
    memories and high-risk ones are not checked; with the model off, none."""
    if memory.memory_type != "text" or conflicts.model_name() is None:
        return []
    existing = collection.get(where=_session_where(memory.session_id), limit=CONFLICT_CANDIDATES, include=[])
    count = len(existing.get("ids") or [])         # 0..CONFLICT_CANDIDATES: never ask for more than exist
    if count == 0:
        return []
    near = collection.query(query_embeddings=EMBED([_embedding_text(memory)]), n_results=min(CONFLICT_CANDIDATES, count),
                            where=_session_where(memory.session_id), include=["metadatas", "distances"])
    found = []
    for chroma_id, meta, distance in zip(near["ids"][0], near["metadatas"][0], near["distances"][0]):
        if distance > CONFLICT_DISTANCE:
            continue
        older = _open_row(chroma_id, meta, session_key)
        if older is None or older.memory_type != "text" or _is_high_risk(older):
            continue
        probability = conflicts.contradicts(older.text_payload.factual_summary, memory.text_payload.factual_summary)
        if probability is not None and probability >= conflicts.THRESHOLD:
            found.append(chroma_id)
    return found


def _mark_conflict(chroma_id: str, other_id: str) -> None:
    """Add other_id to one row's conflicts_with (ChromaDB merges metadata on update)."""
    rows = collection.get(ids=[chroma_id], include=["metadatas"])
    if not rows.get("ids"):
        return
    ids = _conflict_ids(rows["metadatas"][0])
    if other_id not in ids:
        ids = (ids + [other_id])[-MAX_CONFLICTS:]
        collection.update(ids=[chroma_id], metadatas=[{"conflicts_with": ",".join(ids)}])


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
def store_memory(memory: UnifiedMemory, session_key: str = "", ctx: Optional[Context] = None) -> UnifiedMemory:
    """
    Store a memory envelope into long-term memory, encrypted with its session's key.

    Args:
        memory: A UnifiedMemory (session_id, memory_type, compaction_source,
            raw_token_count and the matching code_payload or text_payload).
            Invalid input is rejected by the model's validation.
        session_key: The key of memory.session_id (64 hex characters). The
            first memory stored in a session fixes its key; after that only
            the same key is accepted.

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

    A text memory is also compared with its closest older memories of the
    same session (harness/conflicts.py). If it contradicts one, both are
    marked (conflicts_with), so search_vault and page_context return the
    mark and the chat model can show the user both versions instead of
    silently believing the newer one (fact poisoning, OWASP LLM04/LLM09).
    """
    check = _authorize(memory.session_id, session_key, "store_memory", ctx)
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
    found = _find_conflicts(memory, session_key)
    memory_id = f"mem_{uuid.uuid4().hex[:8]}"
    memory.vector_embedding_id = memory_id

    metadata = _row_metadata(memory, memory_id, session_key, check)
    if found:
        metadata["conflicts_with"] = ",".join(found)
    collection.add(
        ids=[memory_id],
        embeddings=EMBED([_embedding_text(memory)]),     # vectors only: the text is not stored
        metadatas=[metadata],
    )
    for older_id in found:
        _mark_conflict(older_id, memory_id)
    if found:
        logger.info("[store_memory] Memory %s contradicts %s; both are marked.", memory_id, ", ".join(found))
        _audit("store_memory", memory_id, _caller(ctx), f"conflict marked with {', '.join(found)}",
               memory.session_id)

    return memory


@mcp.tool()
def search_vault(query: str, top_k: int = 5, session_id: str = "", session_key: str = "",
                 ctx: Optional[Context] = None) -> List[RetrievedMemory]:
    """
    Search one session's long-term memory for content semantically related to the query.

    Args:
        query: The search query
        top_k: Maximum number of results to return (default: 5)
        session_id: The session to search (required)
        session_key: That session's key (required)
    Returns:
        List of RetrievedMemory records, best match first. Each holds the
        stored UnifiedMemory (memory), its vector distance to the query
        (distance, smaller means closer) and its token cost (token_cost).
        Empty list if no matches found.
        High-risk memories (risk_score >= HIGH_RISK_THRESHOLD) are never
        returned, whoever the caller is.
    """
    _authorize(session_id, session_key, "search_vault", ctx)
    return _search(query, top_k, session_id, session_key)


def _search(query: str, top_k: int, session_id: str, session_key: str) -> List[RetrievedMemory]:
    """search_vault without the key check (callers check it first)."""
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

    results = collection.query(
        query_texts=[query],
        n_results=effective_k,
        where=_session_where(session_id),
        include=["metadatas", "distances"],
    )

    matches = []
    hidden = 0
    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    for i in range(len(ids)):
        memory = _open_row(ids[i], metadatas[i], session_key)
        if memory is None:
            continue
        if _is_high_risk(memory):
            hidden += 1
            continue
        matches.append(RetrievedMemory(
            memory=memory,
            distance=distances[i],
            token_cost=len(encoder.encode(_embedding_text(memory))),
            conflicts_with=_conflict_ids(metadatas[i])[-MAX_CONFLICTS:],
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
def list_all_memories(admin_token: str = "", ctx: Optional[Context] = None) -> List[MemoryInfo]:
    """
    List every stored memory, from every session, without its content. Admin only.

    Args:
        admin_token: The vault's admin token (AMP_ADMIN_TOKEN, else the
            storage/.amp_admin_token file).

    Returns:
        One MemoryInfo per memory: ID, session, type, time, risk score and
        whether it is encrypted. The content stays encrypted: reading it needs
        the session's key (list_session_memories), not the admin token.
    """
    if not is_admin(admin_token):
        _audit("list_all_memories", "", _caller(ctx), "refused: missing or wrong admin token")
        raise PermissionError("list_all_memories needs the admin token.")
    rows = collection.get(include=["metadatas"])
    infos = []
    for chroma_id, meta in zip(rows.get("ids", []), rows.get("metadatas", [])):
        if not meta or meta.get("memory_type") not in ("code", "text"):
            continue
        encrypted = bool(meta.get("encrypted"))
        if encrypted:
            risk = meta.get("risk_score")
        else:                                   # an old plain row: its score (if any) is in the envelope
            risk = json.loads(meta["envelope"]).get("risk_score") if "envelope" in meta else None
        infos.append(MemoryInfo(memory_id=chroma_id, session_id=str(meta.get("session_id", "")),
                                memory_type=meta["memory_type"], compaction_source=meta.get("compaction_source", "rlm"),
                                timestamp=str(meta.get("timestamp", "")), risk_score=risk, encrypted=encrypted))
    return infos


@mcp.tool()
def list_sessions(admin_token: str = "", ctx: Optional[Context] = None) -> List[SessionInfo]:
    """
    Every chat session in the vault, most recently used first. Admin only.

    Args:
        admin_token: The vault's admin token (AMP_ADMIN_TOKEN, else the
            storage/.amp_admin_token file).

    Returns:
        One SessionInfo per session: its ID, how many memories it has and the
        times of its oldest and newest memory. No content: reading a session
        needs its key (list_session_memories).
    """
    if not is_admin(admin_token):
        _audit("list_sessions", "", _caller(ctx), "refused: missing or wrong admin token")
        raise PermissionError("list_sessions needs the admin token.")
    sessions: dict = {}
    for meta in collection.get(include=["metadatas"]).get("metadatas") or []:
        if not meta or not meta.get("session_id"):
            continue
        sid = str(meta["session_id"])
        stamp = str(meta.get("timestamp", ""))
        info = sessions.setdefault(sid, {"session_id": sid, "memories": 0, "first_time": stamp,
                                         "last_time": stamp, "encrypted": True})
        info["memories"] += 1
        info["first_time"] = min(info["first_time"], stamp) if info["first_time"] else stamp
        info["last_time"] = max(info["last_time"], stamp)
        info["encrypted"] = info["encrypted"] and bool(meta.get("encrypted"))
    ordered = sorted(sessions.values(), key=lambda info: info["last_time"], reverse=True)
    return [SessionInfo(**info) for info in ordered]


@mcp.tool()
def list_session_memories(session_id: str, session_key: str = "", limit: int = 0,
                          include_high_risk: bool = False, ctx: Optional[Context] = None) -> List[UnifiedMemory]:
    """
    One session's memories, newest first (for the chat app's vault view and its memory audit).

    Args:
        session_id: The session to list
        session_key: That session's key
        limit: At most this many memories (0 = all)
        include_high_risk: Also return high-risk memories. Leave False when the
            result goes to a model.

    Returns:
        The session's UnifiedMemory records, newest first.
    """
    _authorize(session_id, session_key, "list_session_memories", ctx)
    memories = [m for m in _session_memories(session_id, session_key) if include_high_risk or not _is_high_risk(m)]
    memories.sort(key=lambda m: m.timestamp, reverse=True)
    return memories[:limit] if limit > 0 else memories

def _page_max_distance(requested: Optional[float] = None, query: str = "") -> Optional[float]:
    """The relevance cut-off for page_context, or None for no cut-off:
      - the caller's max_distance if given (0 = none);
      - none when AMP_PAGE_MAX_DISTANCE is "off";
      - none when the question is about the user or the chat ("my", "we",
        "remember", "earlier", ...): their memories are not close in vector
        space ("What do you remember about me?"), so they are paged as before;
      - else AMP_PAGE_MAX_DISTANCE if it is a number, else DEFAULT_PAGE_MAX_DISTANCE."""
    if requested is not None:
        return requested if requested > 0 else None
    raw = os.environ.get(PAGE_MAX_DISTANCE_ENV, "").strip().lower()
    if raw in ("off", "none", "0"):
        return None
    if _ABOUT_THE_CHAT.search(query or ""):
        return None
    if raw:
        try:
            return float(raw)
        except ValueError:
            logger.warning("[page_context] %s=%r is not a number; using the default.", PAGE_MAX_DISTANCE_ENV, raw)
    return DEFAULT_PAGE_MAX_DISTANCE


@mcp.tool()
def page_context(query: str, total_budget: int = 1000, session_id: str = "", candidate_pool: int = 20,
                 session_key: str = "", max_distance: Optional[float] = None,
                 ctx: Optional[Context] = None) -> PageContextResult:
    """
    Retrieve context from one session's long-term memory for a query, limited by a token budget.

    Args:
        query: What the agent needs to recall
        total_budget: Maximum total tokens the returned memories may use (default: 1000)
        session_id: The session to recall from (required)
        candidate_pool: How many top matches to consider before applying the budget (default: 20)
        session_key: That session's key (required)
        max_distance: The relevance cut-off: matches farther from the query
            than this are not paged in (squared L2 distance; smaller is
            closer); 0 turns it off. By default 1.4 (AMP_PAGE_MAX_DISTANCE
            changes it, "off" turns it off), and none for questions about the
            user or the chat ("my", "we", "remember", "earlier", ...), whose
            memories are not close to the question in vector space.

    Returns:
        A PageContextResult with:
          memories: the selected RetrievedMemory records, best match first
          tokens_used: total token cost of the selected memories
          total_budget: the budget that was requested
          skipped: how many matches were left out because they did not fit
          too_far: how many matches were left out by the relevance cut-off
        A memory's token cost is the token count of its searchable text (the
        text summary, or for code the file path, summary and signatures).
        High-risk memories are left out before the budget is applied, so they
        never use up budget (search_vault drops them).
    """
    _authorize(session_id, session_key, "page_context", ctx)
    query = query.strip()
    if not query or total_budget <= 0:
        return PageContextResult(total_budget=total_budget)

    # Same search as search_vault: closest-first, high-risk memories already removed.
    candidates = _search(query, candidate_pool, session_id, session_key)
    if not candidates:
        return PageContextResult(total_budget=total_budget)

    cutoff = _page_max_distance(max_distance, query)
    selected = []
    used_budget = 0
    skipped = 0
    too_far = 0
    for candidate in candidates:
        if cutoff is not None and candidate.distance > cutoff:
            too_far += 1           # not about this query: it would only cost tokens and distract the model
            continue
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
        skipped=skipped,
        too_far=too_far,
    )
    

def _sandbox_record(memory: UnifiedMemory) -> dict:
    """One memory as the simple record that run_memory_code's code sees."""
    text = memory.text_payload
    code = memory.code_payload
    return {
        "id": memory.vector_embedding_id or str(memory.id),
        "time": memory.timestamp.isoformat(),
        "kind": memory.memory_type,
        "text": "\n".join([_embedding_text(memory), *_call_lines(memory)]),   # the graph too: no budget here
        "decisions": list(text.architectural_decisions) if text else [],
        "bugs": list(text.bugs_resolved) if text else [],
        "constraints": list(text.user_constraints) if text else [],
        "file": code.file_path if code else "",
        "calls": {s.name: list(s.calls) for s in code.symbols if s.calls} if code else {},
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
def run_memory_code(code: str, session_id: str = "", timeout: int = 5, session_key: str = "",
                    ctx: Optional[Context] = None) -> SandboxResult:
    """
    Run a short Python program over one session's stored memories, in the sandbox.

    For questions that need computing over many memories rather than reading
    a few: the latest decision about something, how many bugs were fixed,
    every constraint, memories in a time range.

    Args:
        code: Python that reads the list `memories` (already loaded; a top-level
            `memories = ...` line is ignored, other assignments are refused)
            and print()s its answer.
            Each memory is a dict: id, time (ISO string), kind ("text" or
            "code"), text, decisions, bugs, constraints (lists of strings) and
            file (code file path or ""), calls (for code: which function
            calls which, e.g. {"checkout": ["load_cart", "total"]}). They are
            ordered oldest first.
            Only json, re, math, statistics, collections, itertools, datetime
            and string can be imported; open, eval, exec and dunder names are
            blocked.
        session_id: The session whose memories are loaded (required).
        timeout: Seconds before the program is stopped (1 to 10, default 5).
        session_key: That session's key (required).

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
    _authorize(session_id, session_key, "run_memory_code", ctx)
    memories = [m for m in _session_memories(session_id, session_key) if not _is_high_risk(m)]
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