"""
app/server.py - AMP Chat web server
===================================
    python -m app.server                 # then open http://127.0.0.1:8000
    python -m app.server --mcp-url http://localhost:8001/mcp
                                         # memory over MCP (start server/amp_memory_server.py first)

Serves the web UI (app/static/) and a small JSON API around one ChatSession:

    GET  /api/status     is the local model ready, and how is the vault reached (MCP or in-process)?
    GET  /api/state      the chat: transcript, context meter, activity, vault, ...
    POST /api/chat       {"message": "..."} -> text/event-stream. One "event" per
                         behind-the-scenes step as it happens (token count,
                         compaction, memories paged in, reply), then "done" or "error".
    POST /api/settings   {"token_limit": int, "threshold": float, "background": bool}
    POST /api/new        start a new chat
    GET  /api/sessions   the chats that can be reopened (their key is in the local keyring)
    POST /api/open       {"session_id": "..."} reopen one of them

Leaving a chat (new or open) first saves what is still in its context window
to the vault; if that fails, the app stays on the current chat.

Auto-save: after every turn the unsaved part of the chat is also kept in an
encrypted file (harness/chat_store.py). When the app stops (Ctrl+C, SIGTERM),
it waits for a running answer or background compaction (up to 20 s each) and
writes that file one last time. Reopening the chat brings the part back.

Local only: binds to 127.0.0.1, accepts only localhost Host headers (no DNS
rebinding), and POST requests must be JSON from the same origin, so other
websites open in the browser cannot drive it.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import re
import sys
import threading
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from starlette.applications import Starlette  # noqa: E402
from starlette.middleware import Middleware  # noqa: E402
from starlette.middleware.trustedhost import TrustedHostMiddleware  # noqa: E402
from starlette.requests import Request  # noqa: E402
from starlette.responses import FileResponse, JSONResponse, StreamingResponse  # noqa: E402
from starlette.routing import Mount, Route  # noqa: E402
from starlette.staticfiles import StaticFiles  # noqa: E402

from harness.chat_model import ChatModelError, chat_model_name, model_status  # noqa: E402
from harness.local_keys import ensure_signing_key  # noqa: E402
from harness.mcp_client import MCP_URL_ENV, MCPError, memory_server, vault_mode  # noqa: E402
from harness.chat_session import (  # noqa: E402
    DEFAULT_THRESHOLD, DEFAULT_TOKEN_LIMIT, MAX_TOKEN_LIMIT, SHUTDOWN_WAIT, ChatSession, Event, MessageTooLong,
    list_chats,
)
from harness.local_keys import known_sessions  # noqa: E402

logger = logging.getLogger(__name__)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
LOCAL_HOSTS = {"127.0.0.1", "localhost"}
SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")      # same rule as UnifiedMemory.session_id
MIN_TOKEN_LIMIT = 1024
MAX_EVENTS = 300            # most recent activity entries sent to the browser


# ---------------------------------------------------------------------------
# JSON views of the session
# ---------------------------------------------------------------------------

def event_view(event: Event) -> Dict[str, Any]:
    data = {key: value for key, value in event.data.items() if key != "evicted"}   # raw chunk text stays server-side
    return {"turn": event.turn, "kind": event.kind, "title": event.title, "detail": event.detail,
            "seconds": round(event.seconds, 2), "data": data}


def state_view(session: ChatSession) -> Dict[str, Any]:
    code = sum(1 for chunk in session.chunks if chunk.get("type") == "code")
    return {
        "session_id": session.session_id,
        "model": chat_model_name(),
        "turn": session.turn,
        "settings": {"token_limit": session.token_limit, "threshold": session.threshold,
                     "background": session.background, "min_token_limit": MIN_TOKEN_LIMIT,
                     "max_token_limit": MAX_TOKEN_LIMIT},
        "context": {"tokens": session.context_tokens, "limit": session.token_limit,
                    "trigger": session.trigger_tokens, "reclaim_target": session.reclaim_target,
                    "chunks": len(session.chunks), "code_chunks": code},
        "reclaiming": session.reclaiming,
        "stats": {"compactions": session.compactions, "tokens_paged_out": session.tokens_evicted,
                  "tokens_stored": session.tokens_stored},
        "transcript": [dict(entry, status=session.message_status(entry["id"])) for entry in session.transcript],
        "last_paged": session.last_paged,
        "resumed": session.resumed,
        "restored": session.restored,
        "restored_messages": session.restored_messages,
        "events": [event_view(e) for e in session.events[-MAX_EVENTS:]],
        "vault": session.vault(),
    }


# ---------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------

class ChatApp:
    """Holds the one chat session of this local app and serialises its turns."""

    def __init__(self, graph: Any = None, shutdown_wait: float = SHUTDOWN_WAIT):
        self._graph = graph
        self.shutdown_wait = shutdown_wait
        self.turn_lock = threading.Lock()
        self.session = self._new_session(DEFAULT_TOKEN_LIMIT, DEFAULT_THRESHOLD, True)

    def _new_session(self, token_limit: int, threshold: float, background: bool,
                     session_id: Optional[str] = None) -> ChatSession:
        if self._graph is None:
            from harness.graph import create_amp_graph
            self._graph = create_amp_graph()
        session = ChatSession(session_id=session_id, token_limit=token_limit, threshold=threshold,
                              graph=self._graph)
        session.background = background
        return session

    def _leave(self) -> ChatSession:
        """Save what is still in the current chat's context window to the vault."""
        old = self.session
        if not old.save_all():
            raise SaveFailed("Could not save this chat to the vault (see Activity), so it stays open. "
                             "Check that the local model is running and try again.")
        return old

    def reset(self) -> None:
        old = self._leave()
        self.session = self._new_session(old.token_limit, old.threshold, old.background)

    def open(self, session_id: str) -> None:
        """Reopen a chat whose key is in the local keyring."""
        if session_id == self.session.session_id:
            return
        if session_id not in known_sessions():
            raise KeyError(session_id)
        old = self._leave()
        session = self._new_session(old.token_limit, old.threshold, old.background, session_id=session_id)
        session.resume()
        self.session = session

    def shutdown(self) -> bool:
        """When the app stops: let a running answer finish, then keep the chat's
        unsaved part on disk (ChatSession.flush). True if the file is current."""
        got = self.turn_lock.acquire(timeout=self.shutdown_wait)
        try:
            return self.session.flush(self.shutdown_wait)
        finally:
            if got:
                self.turn_lock.release()


class SaveFailed(RuntimeError):
    """The current chat could not be saved, so the app did not leave it."""


def vault_status() -> Dict[str, Any]:
    """How the vault is reached and whether it answers."""
    status = vault_mode()
    if status["mode"] == "mcp":
        try:
            memory_server().ping()
            status["ok"], status["message"] = True, f"Memory over MCP ({status['url']})"
        except MCPError as exc:
            status["ok"], status["message"] = False, str(exc)
    else:
        status["ok"], status["message"] = True, "Memory in-process (no MCP server)"
    return status


def _bad_request(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


def _check_post(request: Request) -> Optional[JSONResponse]:
    """Same-origin JSON only. A JSON content type forces browsers to send a CORS
    preflight for cross-site requests, which this server never approves."""
    origin = request.headers.get("origin")
    if origin and urlsplit(origin).hostname not in LOCAL_HOSTS:
        return _bad_request("Cross-origin requests are not allowed.", 403)
    if not request.headers.get("content-type", "").startswith("application/json"):
        return _bad_request("Send JSON (Content-Type: application/json).", 415)
    return None


async def _json_body(request: Request) -> Dict[str, Any]:
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return body if isinstance(body, dict) else {}


def create_app(graph: Any = None, shutdown_wait: float = SHUTDOWN_WAIT) -> Starlette:
    chat = ChatApp(graph, shutdown_wait)

    @contextlib.asynccontextmanager
    async def lifespan(app):
        yield
        saved = await asyncio.to_thread(chat.shutdown)
        if chat.session.chunks:
            print(f"AMP Chat stopped. {chat.session.session_id}: "
                  + ("its unsaved part is kept on disk; reopen the chat to continue." if saved
                     else "could not save its unsaved part (see the activity log)."))

    async def index(request: Request):
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

    async def status(request: Request):
        ready, message = await asyncio.to_thread(model_status)
        return JSONResponse({"ready": ready, "message": message, "model": chat_model_name(),
                             "vault": await asyncio.to_thread(vault_status)})

    async def state(request: Request):
        try:
            return JSONResponse(await asyncio.to_thread(state_view, chat.session))
        except MCPError as exc:
            return _bad_request(str(exc), 503)

    async def settings(request: Request):
        if (error := _check_post(request)) is not None:
            return error
        body = await _json_body(request)
        session = chat.session
        try:
            if "token_limit" in body:
                session.token_limit = max(MIN_TOKEN_LIMIT, min(MAX_TOKEN_LIMIT, int(body["token_limit"])))
            if "threshold" in body:
                session.threshold = max(0.5, min(0.95, float(body["threshold"])))
            if "background" in body:
                session.background = bool(body["background"])
        except (TypeError, ValueError):
            return _bad_request("token_limit must be a number of tokens and threshold a fraction such as 0.8.")
        return JSONResponse(await asyncio.to_thread(state_view, session))

    async def new_chat(request: Request):
        if (error := _check_post(request)) is not None:
            return error
        if not chat.turn_lock.acquire(blocking=False):
            return _bad_request("Wait for the current answer to finish.", 409)
        try:
            await asyncio.to_thread(chat.reset)
        except (SaveFailed, MCPError) as exc:
            return _bad_request(str(exc), 503)
        finally:
            chat.turn_lock.release()
        return JSONResponse(await asyncio.to_thread(state_view, chat.session))

    async def sessions(request: Request):
        try:
            chats = await asyncio.to_thread(list_chats, chat.session.session_id)
        except MCPError as exc:
            return _bad_request(str(exc), 503)
        return JSONResponse({"chats": chats})

    async def open_chat(request: Request):
        if (error := _check_post(request)) is not None:
            return error
        session_id = str((await _json_body(request)).get("session_id", ""))
        if not SESSION_ID_RE.match(session_id):
            return _bad_request("That is not a chat ID.")
        if not chat.turn_lock.acquire(blocking=False):
            return _bad_request("Wait for the current answer to finish.", 409)
        try:
            await asyncio.to_thread(chat.open, session_id)
        except KeyError:
            return _bad_request("No such chat on this computer (its key is not in the local keyring).", 404)
        except (SaveFailed, MCPError) as exc:
            return _bad_request(str(exc), 503)
        finally:
            chat.turn_lock.release()
        return JSONResponse(await asyncio.to_thread(state_view, chat.session))

    async def send(request: Request):
        if (error := _check_post(request)) is not None:
            return error
        message = str((await _json_body(request)).get("message", "")).strip()
        if not message:
            return _bad_request("The message is empty.")
        if not chat.turn_lock.acquire(blocking=False):
            return _bad_request("Wait for the current answer to finish.", 409)

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        session = chat.session

        def put(kind: str, payload: Dict[str, Any]) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, (kind, payload))

        def work() -> None:
            try:
                reply = session.ask(message, on_event=lambda e: put("event", event_view(e)))
                put("done", {"reply": reply})
            except (MessageTooLong, ChatModelError, MCPError, ValueError) as exc:
                put("error", {"message": str(exc)})
            except Exception as exc:                      # keep the server alive whatever happens
                logger.exception("Turn failed")
                put("error", {"message": f"Unexpected error: {type(exc).__name__}: {exc}"})
            finally:
                chat.turn_lock.release()

        threading.Thread(target=work, name="amp-turn", daemon=True).start()

        async def stream():
            while True:
                kind, payload = await queue.get()
                yield f"event: {kind}\ndata: {json.dumps(payload)}\n\n"
                if kind in ("done", "error"):
                    break

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    routes = [
        Route("/", index),
        Route("/api/status", status),
        Route("/api/state", state),
        Route("/api/settings", settings, methods=["POST"]),
        Route("/api/new", new_chat, methods=["POST"]),
        Route("/api/sessions", sessions),
        Route("/api/open", open_chat, methods=["POST"]),
        Route("/api/chat", send, methods=["POST"]),
        Mount("/static", StaticFiles(directory=STATIC_DIR), name="static"),
    ]
    middleware = [Middleware(TrustedHostMiddleware, allowed_hosts=[*LOCAL_HOSTS, "testserver"])]
    app = Starlette(routes=routes, middleware=middleware, lifespan=lifespan)
    app.state.chat = chat
    return app


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m app.server", description="AMP Chat web server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mcp-url", default=None,
                        help="reach the memory server over MCP, e.g. http://localhost:8001/mcp "
                             f"(default: ${MCP_URL_ENV}, else in-process)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="keep the default: the app has no login, so it should only be reachable from this machine")
    args = parser.parse_args(argv)
    if args.host not in LOCAL_HOSTS:
        parser.error("AMP Chat has no login; serve it on 127.0.0.1 only.")

    if args.mcp_url:
        os.environ[MCP_URL_ENV] = args.mcp_url
    key_source = ensure_signing_key()

    import uvicorn
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    print(f"AMP Chat on http://{args.host}:{args.port}  (model: {chat_model_name()}; "
          f"{vault_status()['message']}; signing key {key_source})")
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
