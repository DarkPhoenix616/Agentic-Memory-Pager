"""
harness/chat_session.py - One chat conversation on top of the AMP graph
=======================================================================
ChatSession.ask(text) runs one user turn through create_amp_graph():

    token monitor -> while the context is >= threshold: compact the oldest
    chunk (code -> AST compactor, text -> RLM summarizer) and store it in the
    vault -> page relevant memories back in -> local chat model -> reply

Paging out works like an operating system reclaiming memory:
    background reclaim  after each reply, if the context passed the trigger
                        (80%), a background thread compacts the oldest chunks
                        until it is under a lower mark (70%), so the user does
                        not wait for the summarizer (like Linux's kswapd);
    direct reclaim      if a new message still pushes the context past the
                        trigger, the graph compacts inside the turn.

Leaving a chat (a new chat, or opening another one) first saves what is still
in the context window: save_all() pages every chunk out to the vault, so the
chat can be reopened later. ChatSession(session_id=...).resume() reopens a
chat: its vault is there again, and the latest memories are shown as where
the chat left off. Only chats whose key is in the local keyring can be
reopened (their memories are encrypted with it); list_chats() lists them.

Auto-save: after every turn and every compaction, save_pending() writes what
is still only in the context window to an encrypted file (harness/chat_store.py).
So if the app stops without the chat being left (Ctrl+C, a crash), nothing but
the turn in progress is lost: resume() puts that part back into the context
window. flush() is the last save when the app shuts down.

Between turns it keeps:
    chunks      the active context (the "RAM"): recent messages, one chunk per
                prose part or code block, oldest first
    transcript  every message, for display; a message whose chunks were all
                compacted is marked as paged out
    events      what happened behind the scenes, turn by turn

The class has no UI code; app/server.py serves it to the web UI in app/static/.
"""
from __future__ import annotations

import ast
import json
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage

from harness import chat_store
from harness import graph as amp
from harness.chat_model import chat_model_name
from harness.local_keys import known_sessions

DEFAULT_TOKEN_LIMIT = 3072      # small on purpose, so compaction is visible after a few turns
MAX_TOKEN_LIMIT = 6144          # the chat model runs with an 8,192-token window
DEFAULT_THRESHOLD = 0.80
RESUME_NOTES = 5                # memories shown as "where we left off" when a chat is reopened
TITLE_CHARS = 70
PAGE_BUDGET = 1000              # same budget llm_reasoning_node uses
RECURSION_LIMIT = 200           # each compaction is 4 graph steps
RECLAIM_MARGIN = 0.10           # background compaction stops 10% of the window below the trigger
SHUTDOWN_WAIT = 20.0            # seconds a shutdown waits for a running answer, then for a background compaction

_FENCE_RE = re.compile(r"```[ \t]*([\w+#.-]*)[^\n]*\n(.*?)(?:```|\Z)", re.DOTALL)
_LANGUAGES = {"py": "python", "python": "python", "python3": "python",
              "js": "javascript", "javascript": "javascript", "ts": "typescript",
              "typescript": "typescript", "c": "c", "cpp": "cpp", "c++": "cpp",
              "java": "java", "go": "go", "rust": "rust", "rs": "rust", "sh": "bash",
              "bash": "bash"}
_EXTENSIONS = {"python": "py", "javascript": "js", "typescript": "ts", "c": "c", "cpp": "cpp",
               "java": "java", "go": "go", "rust": "rs", "bash": "sh"}


class MessageTooLong(ValueError):
    """The message alone would fill the context window past the threshold."""


@dataclass
class Event:
    """One thing that happened behind the scenes."""
    turn: int
    kind: str          # monitor | dispatch | compaction | search | page_in | reply | error
    title: str
    detail: str = ""
    seconds: float = 0.0
    data: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Messages -> context chunks
# ---------------------------------------------------------------------------

def _looks_like_python(text: str) -> bool:
    """A message that is a bare Python file (no ``` fences)."""
    if text.count("\n") < 2:
        return False
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return False
    return any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                                 ast.Import, ast.ImportFrom)) for node in tree.body)


def _guess_language(tag: str, code: str) -> str:
    tag = tag.strip().lower()
    if tag in _LANGUAGES:
        return _LANGUAGES[tag]
    if not tag:
        try:
            ast.parse(code)
            return "python"
        except (SyntaxError, ValueError):
            return "unknown"
    return tag


def split_message(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """(prose, [(language, code), ...]) for one message. Code blocks in the prose
    are replaced by a short "[code block N: language]" marker."""
    if _looks_like_python(text) and "```" not in text:
        return "", [("python", text)]
    prose: List[str] = []
    blocks: List[Tuple[str, str]] = []
    last = 0
    for match in _FENCE_RE.finditer(text):
        prose.append(text[last:match.start()])
        last = match.end()
        code = match.group(2)
        if code.strip():
            blocks.append((_guess_language(match.group(1), code), code))
            prose.append(f"[code block {len(blocks)}: {blocks[-1][0]}]")
    prose.append(text[last:])
    return "".join(prose).strip(), blocks


def _code_chunks(role: str, blocks: List[Tuple[str, str]], msg_id: str) -> List[dict]:
    return [{"role": role, "type": "code", "content": code, "language": language,
             "file_path": f"chat/{msg_id}_{n}.{_EXTENSIONS.get(language, 'txt')}", "msg_ids": [msg_id]}
            for n, (language, code) in enumerate(blocks, 1)]


def turn_to_chunks(user_text: str, user_id: str, reply_text: str, reply_id: str) -> List[dict]:
    """Context chunks for one finished turn, oldest first:

        user's code blocks  ->  the exchange (question + answer prose)  ->  reply's code blocks

    The exchange is one text chunk, "User: ...\n\nAssistant: ...", so the RLM
    summarizer compacts a question together with its answer. Each code block is
    its own chunk for the AST compactor. `reply_at` marks where the answer
    starts, so the chat model can see the two turns separately again."""
    user_prose, user_blocks = split_message(user_text)
    reply_prose, reply_blocks = split_message(reply_text)
    user_part = f"User: {user_prose or '[code only]'}"
    reply_part = f"Assistant: {reply_prose or '[code only]'}"
    exchange = {"role": "user", "type": "text", "content": f"{user_part}\n\n{reply_part}",
                "reply_at": len(user_part) + 2, "msg_ids": [user_id, reply_id]}
    return (_code_chunks("user", user_blocks, user_id) + [exchange]
            + _code_chunks("assistant", reply_blocks, reply_id))


def chunk_tokens(chunk: dict) -> int:
    """Tokens of one chunk, counted exactly as token_monitor_node counts them."""
    return amp._count_tokens(json.dumps(chunk))


def _memory_view(memory) -> Dict[str, Any]:
    """Short description of a stored UnifiedMemory for the UI."""
    if memory.memory_type == "code" and memory.code_payload:
        code = memory.code_payload
        text = "\n".join([code.file_path] + [s.signature for s in code.symbols[:6]])
        return {"engine": "AST", "label": code.file_path, "text": text,
                "tokens": amp._count_tokens(text), "id": memory.vector_embedding_id}
    payload = memory.text_payload
    text = payload.factual_summary if payload else ""
    lists = {}
    if payload:
        lists = {"decisions": payload.architectural_decisions, "bugs": payload.bugs_resolved,
                 "constraints": payload.user_constraints}
    return {"engine": "RLM", "label": payload.timeframe if payload else "text", "text": text,
            "tokens": amp._count_tokens(text), "id": memory.vector_embedding_id, **lists}


# ---------------------------------------------------------------------------
# The session
# ---------------------------------------------------------------------------

class ChatSession:
    def __init__(self, session_id: Optional[str] = None, token_limit: int = DEFAULT_TOKEN_LIMIT,
                 threshold: float = DEFAULT_THRESHOLD, graph: Any = None):
        self.session_id = session_id or f"chat-{uuid.uuid4().hex[:8]}"
        self.token_limit = token_limit
        self.threshold = threshold
        self.graph = graph or amp.create_amp_graph()
        self.chunks: List[dict] = []
        self.transcript: List[dict] = []
        self.events: List[Event] = []
        self.turn = 0
        self.compactions = 0
        self.tokens_evicted = 0
        self.tokens_stored = 0
        self.last_paged: List[Dict[str, Any]] = []
        self.resumed: List[Dict[str, Any]] = []      # the latest memories, when this is a reopened chat
        self.restored = 0               # unsaved chunks brought back when this chat was reopened
        self.background = True          # compact after the reply instead of before the next one
        self._reclaimer: Optional[threading.Thread] = None
        self._latest_ids: set = set()   # the last turn's messages: never paged out in the background
        self._ids = 0
        self._pending_lock = threading.Lock()
        self._pending_on_disk = False   # only a file this session wrote or read is ever removed

    # -- numbers for the UI ------------------------------------------------

    @property
    def context_tokens(self) -> int:
        return sum(chunk_tokens(c) for c in self.chunks)

    @property
    def trigger_tokens(self) -> int:
        return int(self.token_limit * self.threshold)

    @property
    def reclaim_target(self) -> int:
        """Background compaction runs until the context is at most this many tokens."""
        return int(self.token_limit * max(0.1, self.threshold - RECLAIM_MARGIN))

    @property
    def reclaiming(self) -> bool:
        return self._reclaimer is not None and self._reclaimer.is_alive()

    def message_status(self, msg_id: str) -> str:
        """'in context', 'paged out' or 'partly paged out' for one transcript message."""
        entry = next((e for e in self.transcript if e["id"] == msg_id), None)
        if entry is None:
            return "unknown"
        live = sum(1 for c in self.chunks if msg_id in c.get("msg_ids", ()))
        if live == 0:
            return "paged out"
        return "in context" if live >= entry.get("chunks", 1) else "partly paged out"

    def vault(self) -> List[Dict[str, Any]]:
        """This session's stored memories, newest first."""
        memories = amp._mcp_list_session_memories(self.session_id)      # newest first
        return [_memory_view(m) for m in memories]

    # -- one turn ------------------------------------------------------------

    def _new_id(self, role: str) -> str:
        self._ids += 1
        return f"{role[0]}{self._ids}"

    def _state(self, text: str) -> dict:
        return {
            "session_id": self.session_id,
            "current_token_count": 0,
            "token_limit": self.token_limit,
            "compaction_threshold": self.threshold,
            "context_chunks": list(self.chunks),
            "oldest_chunk_type": "text",
            "compaction_in_progress": False,
            "total_compactions": 0,
            "messages": [HumanMessage(content=text)],
            "needs_memory_search": False,
            "search_query": "",
            "code_request": "",
            "audit_requested": False,
        }

    def ask(self, text: str, on_event: Optional[Callable[[Event], None]] = None) -> str:
        """Run one turn and return the model's reply. Raises MessageTooLong, or
        ChatModelError when the local model cannot answer; compactions done
        before the error are kept."""
        text = text.strip()
        if not text:
            raise ValueError("Empty message.")
        needed = amp._count_tokens(f"human: {text}")
        if needed >= self.trigger_tokens:
            raise MessageTooLong(
                f"This message is {needed:,} tokens; the context window triggers compaction at "
                f"{self.trigger_tokens:,}. Raise the token limit or send a shorter message.")

        self.turn += 1
        emit = self._emitter(on_event)
        waited = self.wait_for_reclaim()
        if waited >= 0.05:
            emit(Event(self.turn, "monitor", f"Waited {waited:.1f}s for background compaction to finish"))
        state = self._state(text)
        values: Optional[dict] = None
        live = list(self.chunks)
        clock = time.monotonic()
        try:
            for mode, data in self.graph.stream(state, {"recursion_limit": RECURSION_LIMIT},
                                                stream_mode=["updates", "values"]):
                if mode == "values":
                    values = data
                    continue
                for node, update in data.items():
                    seconds = time.monotonic() - clock
                    clock = time.monotonic()
                    live = self._on_update(node, update or {}, live, seconds, emit)
        except Exception as exc:
            emit(Event(self.turn, "error", "The turn failed", str(exc)))
            raise
        finally:
            if values is not None:
                self.chunks = list(values["context_chunks"])    # keep compactions done so far
                self.save_pending()

        reply = next((m.content for m in reversed(values["messages"]) if isinstance(m, AIMessage)), "")
        user_id, reply_id = self._new_id("user"), self._new_id("assistant")
        new_chunks = turn_to_chunks(text, user_id, reply, reply_id)
        self.chunks += new_chunks
        self._latest_ids = {user_id, reply_id}
        for msg_id, role, content in ((user_id, "user", text), (reply_id, "assistant", reply)):
            owned = sum(1 for c in new_chunks if msg_id in c["msg_ids"])
            self.transcript.append({"id": msg_id, "role": role, "content": content,
                                    "chunks": owned, "turn": self.turn})
        self._record_page_in(text, emit)
        self.save_pending()
        if self.background and self.context_tokens >= self.trigger_tokens:
            self._reclaimer = threading.Thread(target=self._reclaim, args=(self.turn,),
                                               name="amp-reclaim", daemon=True)
            self._reclaimer.start()
        return reply

    # -- background reclaim ---------------------------------------------------

    def wait_for_reclaim(self) -> float:
        """Block until background compaction is done; returns the seconds waited."""
        if self._reclaimer is None:
            return 0.0
        started = time.monotonic()
        self._reclaimer.join()
        self._reclaimer = None
        return time.monotonic() - started

    def _reclaim(self, turn: int) -> None:
        """Compact the oldest chunks until the context is under reclaim_target.
        The turn that just ended is the working set and stays in context, even
        if it alone is over the target. Runs in a thread; ask() waits for it
        before touching the chunks."""
        emit = self._emitter(None)
        while self.chunks and self.context_tokens > self.reclaim_target:
            if self._latest_ids & set(self.chunks[0].get("msg_ids", ())):
                break
            if not self._compact_oldest(turn, emit, "in the background"):
                return

    def _compact_oldest(self, turn: int, emit, when: str) -> bool:
        """Page the oldest chunk out to the vault; False if that failed."""
        state = self._state("")
        state["messages"] = []
        evicted = self.chunks[0]
        node = amp.ast_compactor_node if evicted.get("type") == "code" else amp.rlm_summarizer_node
        started = time.monotonic()
        try:
            update = node(state)
        except Exception as exc:
            emit(Event(turn, "error", f"Compaction {when} failed", str(exc)))
            return False
        if "context_chunks" not in update:
            return False
        self.chunks = list(update["context_chunks"])
        self.save_pending()
        self._record_compaction(turn, evicted, time.monotonic() - started, emit, when=when)
        return True

    def save_all(self, on_event: Optional[Callable[[Event], None]] = None) -> bool:
        """Page every chunk still in the context window out to the vault, so
        nothing is lost when the user leaves this chat. True if all were saved."""
        self.wait_for_reclaim()
        emit = self._emitter(on_event)
        while self.chunks:
            if not self._compact_oldest(self.turn, emit, "before leaving the chat"):
                return False
        return True

    def resume(self, limit: int = RESUME_NOTES) -> List[Dict[str, Any]]:
        """Reopen this chat: keep its latest memories as "where we left off",
        and bring back what was still in its context window if the app stopped
        before the chat was left."""
        self.restored = self._restore_pending()
        memories = amp._mcp_list_session_memories(self.session_id)       # newest first
        self.resumed = [_memory_view(m) for m in memories[:limit]]
        title = f"Reopened this chat: {len(memories)} memor{'y' if len(memories) == 1 else 'ies'} in its vault"
        if self.restored:
            title += (f", and {self.restored} unsaved chunk{'' if self.restored == 1 else 's'} "
                      f"back in the context window")
        self.events.append(Event(0, "resume", title, ""))
        return self.resumed

    # -- auto-save: the unsaved part on disk ---------------------------------

    def save_pending(self) -> bool:
        """Write what is still only in the context window to this chat's
        encrypted file, or remove the file once nothing is left. Runs after
        every turn and every compaction. True if the file is up to date."""
        with self._pending_lock:
            chunks = list(self.chunks)
            try:
                if chunks:
                    chat_store.save_pending(self.session_id, {"turn": self.turn, "ids": self._ids,
                                                              "chunks": chunks})
                    self._pending_on_disk = True
                elif self._pending_on_disk:
                    chat_store.clear_pending(self.session_id)
                    self._pending_on_disk = False
                return True
            except Exception as exc:
                self.events.append(Event(self.turn, "error", "Could not keep the unsaved part of this chat "
                                         "on disk", str(exc)))
                return False

    def _restore_pending(self) -> int:
        """Put back the chunks the file holds; the number of chunks restored."""
        try:
            saved = chat_store.load_pending(self.session_id)
        except Exception as exc:                          # wrong key or damaged file: keep the file
            self.events.append(Event(0, "error", "Could not read the unsaved part of this chat", str(exc)))
            return 0
        if not saved:
            return 0
        self.chunks = list(saved["chunks"])
        self.turn = max(self.turn, int(saved.get("turn", 0)))
        self._ids = max(self._ids, int(saved.get("ids", 0)))    # new message IDs don't clash with old chunks
        self._pending_on_disk = True
        return len(self.chunks)

    def flush(self, timeout: float = SHUTDOWN_WAIT) -> bool:
        """The last save when the app shuts down: give a running background
        compaction up to `timeout` seconds, then make sure the file is current."""
        if self._reclaimer is not None:
            self._reclaimer.join(timeout)
        return self.save_pending()

    # -- behind the scenes ----------------------------------------------------

    def _emitter(self, on_event):
        def emit(event: Event) -> None:
            self.events.append(event)
            if on_event:
                on_event(event)
        return emit

    def _on_update(self, node: str, update: dict, live: List[dict], seconds: float, emit) -> List[dict]:
        if node == "token_monitor_node":
            tokens = update.get("current_token_count", 0)
            share = tokens / self.token_limit
            action = (f"over {self.threshold:.0%}, compacting the oldest chunk"
                      if share >= self.threshold else "room left, answering")
            emit(Event(self.turn, "monitor", f"Context {tokens:,} / {self.token_limit:,} tokens ({share:.0%})",
                       action, seconds, {"tokens": tokens}))
        elif node == "compaction_dispatch_node":
            kind = update.get("oldest_chunk_type", "text")
            engine = "AST code compactor" if kind == "code" else "RLM text summarizer"
            emit(Event(self.turn, "dispatch", f"Oldest chunk is {kind} → {engine}", "", seconds))
        elif node in ("ast_compactor_node", "rlm_summarizer_node") and "context_chunks" in update:
            self._record_compaction(self.turn, live[0] if live else {}, seconds, emit, when="before answering")
            return list(update["context_chunks"])
        elif node == "llm_reasoning_node":
            if update.get("code_request"):
                emit(Event(self.turn, "code", "Model wrote Python to run over the vault",
                           update["code_request"].strip()[:600], seconds))
            elif update.get("needs_memory_search"):
                emit(Event(self.turn, "search", f"Model asked to search the vault: “{update.get('search_query', '')}”",
                           "", seconds))
            elif update.get("messages"):
                emit(Event(self.turn, "reply", f"Answered by {chat_model_name()}", "", seconds))
        elif node == "memory_code_node" and update.get("messages"):
            emit(Event(self.turn, "code", "Sandbox ran it over the vault",
                       str(update["messages"][-1].content), seconds))
        elif node == "memory_search_node" and update.get("messages"):
            emit(Event(self.turn, "search", "Vault search results", str(update["messages"][-1].content), seconds))
        return live

    def _record_compaction(self, turn: int, evicted: dict, seconds: float, emit, when: str) -> None:
        raw = amp._count_tokens(evicted.get("content", ""))
        stored = self.vault()
        view = stored[0] if stored else {"engine": "?", "text": "", "tokens": 0, "label": ""}
        self.compactions += 1
        self.tokens_evicted += raw
        self.tokens_stored += view["tokens"]
        if evicted.get("reply_at"):
            what = "an exchange"
        elif evicted.get("type") == "code":
            what = f"a {evicted.get('language') or ''} code block from the {evicted.get('role', 'user')}"
        else:
            what = "a text chunk"
        emit(Event(turn, "compaction",
                   f"{view['engine']}: paged out {what} {when}, {raw:,} → {view['tokens']:,} tokens",
                   view["text"], seconds, {"memory": view, "evicted": evicted,
                                           "background": when == "in the background"}))

    def _record_page_in(self, text: str, emit) -> None:
        """Show which memories llm_reasoning_node paged in for this turn (same query and budget)."""
        try:
            paged = amp._mcp_page_context(query=text, total_budget=PAGE_BUDGET, session_id=self.session_id)
        except Exception as exc:
            emit(Event(self.turn, "error", "Could not read the paged-in memories", str(exc)))
            return
        self.last_paged = [dict(_memory_view(rm.memory), distance=rm.distance) for rm in paged.memories]
        if self.last_paged:
            emit(Event(self.turn, "page_in",
                       f"Paged in {len(self.last_paged)} memor{'y' if len(self.last_paged) == 1 else 'ies'} "
                       f"({paged.tokens_used:,} tokens) for this answer", "", 0.0,
                       {"memories": self.last_paged}))


# ---------------------------------------------------------------------------
# Chats that can be reopened
# ---------------------------------------------------------------------------

def _title(view: Dict[str, Any]) -> str:
    """A short title for a chat: the start of its oldest memory."""
    text = re.sub(r"^\s*(user|human)\s*:\s*", "", view.get("text", ""), flags=re.IGNORECASE)
    text = " ".join(text.split())
    return text if len(text) <= TITLE_CHARS else text[:TITLE_CHARS - 1].rstrip() + "…"


def _pending_title(session_id: str) -> str:
    """A title from the oldest unsaved chunk, for a chat with nothing in the vault yet."""
    try:
        saved = chat_store.load_pending(session_id) or {}
    except Exception:
        return ""
    text = next((c["content"] for c in saved.get("chunks", []) if c.get("type") == "text"), "")
    return _title({"text": text})


def list_chats(current_id: str = "") -> List[Dict[str, Any]]:
    """The chats this computer can reopen, most recently used first: every
    session in the vault, or with an unsaved part on disk (chat_store), whose
    key is in the local keyring. Each has its ID, a title, how many memories it
    has, whether an unsaved part is waiting, and when it was last used."""
    keys = known_sessions()
    pending = chat_store.pending_sessions()
    chats = []
    for info in amp._mcp_list_sessions():
        if info.session_id not in keys:
            continue                                   # another client's chat: cannot be read here
        memories = amp._mcp_list_session_memories(info.session_id)     # newest first
        chats.append({"session_id": info.session_id, "memories": info.memories,
                      "last_time": max(info.last_time, pending.get(info.session_id, "")),
                      "unsaved": info.session_id in pending, "current": info.session_id == current_id,
                      "title": _title(_memory_view(memories[-1])) if memories else ""})
    listed = {chat["session_id"] for chat in chats}
    for session_id, stamp in pending.items():
        if session_id in keys and session_id not in listed:
            chats.append({"session_id": session_id, "memories": 0, "last_time": stamp, "unsaved": True,
                          "current": session_id == current_id, "title": _pending_title(session_id)})
    return sorted(chats, key=lambda chat: chat["last_time"], reverse=True)
