"""
harness/mcp_client.py - The memory server, in-process or over MCP
=================================================================
    memory_server() -> an object with the memory server's six tools:
        store_memory(memory, session_key)                           -> UnifiedMemory
        search_vault(query, top_k, session_id, session_key)         -> List[RetrievedMemory]
        page_context(query, total_budget, session_id, session_key)  -> PageContextResult
        list_session_memories(session_id, session_key, ...)         -> List[UnifiedMemory]
        run_memory_code(code, session_id, timeout, session_key)     -> SandboxResult
        list_all_memories(admin_token)                              -> List[MemoryInfo]   (admin only)
        delete_memory(memory_id, admin_token)                       -> str                (admin only)

Every session tool needs that session's key; memories are stored encrypted with it.

Two ways to reach the server:

    over MCP     AMP_MCP_URL is set, e.g. http://localhost:8001/mcp. Calls go over
                 the Model Context Protocol (streamable HTTP) to a running memory
                 server: python server/amp_memory_server.py. Only that process opens
                 ChromaDB, and any other MCP client (Claude Code, Cursor, ...) can
                 use the same vault.
    in-process   AMP_MCP_URL is not set (the default). The server module is
                 imported and its tool functions are called directly. No second
                 process is needed; tests and quick runs use this.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import importlib
import os
import threading
from typing import Any, Dict, List, Optional

from harness.guardrails import MemoryInfo, PageContextResult, RetrievedMemory, SandboxResult, UnifiedMemory

MCP_URL_ENV = "AMP_MCP_URL"
CALL_TIMEOUT = 60.0
CONNECT_TIMEOUT = 15.0
_PING = "__ping__"
_STOP = "__stop__"


class MCPError(RuntimeError):
    """The MCP memory server could not be reached, or a tool call failed."""


class MCPMemoryClient:
    """A synchronous client for the AMP memory server over MCP.

    A background thread runs an event loop that owns one MCP session. Calls
    from any thread are queued to it and run one at a time. If the connection
    drops, the next call reconnects."""

    def __init__(self, url: str, *, call_timeout: float = CALL_TIMEOUT, connect_timeout: float = CONNECT_TIMEOUT):
        self.url = url
        self.call_timeout = call_timeout
        self.connect_timeout = connect_timeout
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._queue: Optional[asyncio.Queue] = None
        self._worker: Optional[concurrent.futures.Future] = None
        self._ready = threading.Event()
        self._error: Optional[BaseException] = None

    # -- the six tools ---------------------------------------------------------

    def store_memory(self, memory: UnifiedMemory, session_key: str = "") -> UnifiedMemory:
        return UnifiedMemory.model_validate(self._call(
            "store_memory", {"memory": memory.model_dump(mode="json"), "session_key": session_key}))

    def search_vault(self, query: str, top_k: int = 5, session_id: str = "",
                     session_key: str = "") -> List[RetrievedMemory]:
        data = self._call("search_vault", {"query": query, "top_k": top_k, "session_id": session_id,
                                           "session_key": session_key})
        return [RetrievedMemory.model_validate(item) for item in data.get("result", [])]

    def page_context(self, query: str, total_budget: int = 1000, session_id: str = "",
                     session_key: str = "") -> PageContextResult:
        return PageContextResult.model_validate(self._call(
            "page_context", {"query": query, "total_budget": total_budget, "session_id": session_id,
                             "session_key": session_key}))

    def list_session_memories(self, session_id: str, session_key: str = "", limit: int = 0,
                              include_high_risk: bool = False) -> List[UnifiedMemory]:
        data = self._call("list_session_memories", {"session_id": session_id, "session_key": session_key,
                                                    "limit": limit, "include_high_risk": include_high_risk})
        return [UnifiedMemory.model_validate(item) for item in data.get("result", [])]

    def list_all_memories(self, admin_token: str = "") -> List[MemoryInfo]:
        data = self._call("list_all_memories", {"admin_token": admin_token})
        return [MemoryInfo.model_validate(item) for item in data.get("result", [])]

    def delete_memory(self, memory_id: str, admin_token: str = "") -> str:
        return str(self._call("delete_memory", {"memory_id": memory_id, "admin_token": admin_token}).get("result", ""))

    def run_memory_code(self, code: str, session_id: str = "", timeout: int = 5,
                        session_key: str = "") -> SandboxResult:
        return SandboxResult.model_validate(self._call(
            "run_memory_code", {"code": code, "session_id": session_id, "timeout": timeout,
                                "session_key": session_key}))

    def ping(self) -> None:
        """Raise MCPError if the server cannot be reached."""
        self._call(_PING, {})

    def close(self) -> None:
        with self._lock:
            if self._loop is not None and self._queue is not None and self._worker and not self._worker.done():
                self._loop.call_soon_threadsafe(self._queue.put_nowait, (_STOP, {}, None))
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._loop.stop)
                self._loop = None

    # -- plumbing ----------------------------------------------------------------

    def _call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        self._connect()
        future: concurrent.futures.Future = concurrent.futures.Future()
        self._loop.call_soon_threadsafe(self._queue.put_nowait, (name, arguments, future))
        try:
            result = future.result(timeout=self.call_timeout)
        except concurrent.futures.TimeoutError as exc:
            raise MCPError(f"MCP call {name} to {self.url} timed out after {self.call_timeout:.0f}s") from exc
        if name == _PING:
            return {}
        if result.isError:
            text = " ".join(getattr(item, "text", "") for item in result.content)
            raise MCPError(f"MCP tool {name} failed: {text[:500]}")
        if not isinstance(result.structuredContent, dict):
            raise MCPError(f"MCP tool {name} returned no structured result (is the server's mcp package current?)")
        return result.structuredContent

    def _connect(self) -> None:
        with self._lock:
            if self._worker is None or self._worker.done():
                if self._loop is None:
                    self._loop = asyncio.new_event_loop()
                    threading.Thread(target=self._loop.run_forever, name="amp-mcp", daemon=True).start()
                self._ready.clear()
                self._error = None
                self._worker = asyncio.run_coroutine_threadsafe(self._serve(), self._loop)
            ready = self._ready.wait(self.connect_timeout)
            if not ready or self._error is not None:      # an error means the session is gone or going
                detail = f" ({type(self._error).__name__}: {self._error})" if self._error else ""
                raise MCPError(f"Cannot reach the MCP memory server at {self.url}{detail}. "
                               f"Start it with: python server/amp_memory_server.py")

    async def _serve(self) -> None:
        """Own one MCP session and run the queued calls on it, in order."""
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        queue: asyncio.Queue = asyncio.Queue()
        self._queue = queue
        try:
            async with streamablehttp_client(self.url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._ready.set()
                    while True:
                        name, arguments, future = await queue.get()
                        if name == _STOP:
                            return
                        try:
                            result = await (session.send_ping() if name == _PING else session.call_tool(name, arguments))
                        except Exception as exc:          # the connection broke: fail this call, reconnect next time
                            future.set_exception(MCPError(f"MCP call {name} to {self.url} failed: {exc}"))
                            raise
                        future.set_result(result)
        except BaseException as exc:
            self._error = exc
        finally:
            self._ready.set()                              # wake a caller waiting for the connection
            while not queue.empty():                       # and fail calls that will never run
                _, _, future = queue.get_nowait()
                if future is not None and not future.done():
                    future.set_exception(MCPError(f"The MCP connection to {self.url} closed"))


# ---------------------------------------------------------------------------
# Choosing the transport
# ---------------------------------------------------------------------------

_clients: Dict[str, MCPMemoryClient] = {}
_clients_lock = threading.Lock()


def mcp_url() -> Optional[str]:
    return os.environ.get(MCP_URL_ENV, "").strip() or None


def memory_server():
    """The MCP client when AMP_MCP_URL is set, else the server module itself (in-process)."""
    url = mcp_url()
    if url is None:
        return importlib.import_module("server.amp_memory_server")   # opens ChromaDB in this process
    with _clients_lock:
        if url not in _clients:
            _clients[url] = MCPMemoryClient(url)
        return _clients[url]


def vault_mode() -> Dict[str, Any]:
    """How the vault is reached, for status displays."""
    url = mcp_url()
    return {"mode": "mcp", "url": url} if url else {"mode": "in-process", "url": None}
