"""
integrations/claude_code/project.py - settings and saved progress
=================================================================
One Claude Code project folder = one AMP session:

    session ID   "cc-" + the first 20 hex characters of SHA-256(real path)
    session key  a random key in its own keyring, storage/.amp_cc_session_keys
                 (AMP_CC_KEYS_FILE overrides it). It is not AMP Chat's keyring,
                 so Claude Code projects do not show up as chats in AMP Chat.
    progress     storage/claude_code/<session>.json (AMP_CC_STATE_DIR overrides
                 the folder): how far each conversation file has been saved,
                 and which files were already stored as code memories.

The memory server must run as its own process (python server/amp_memory_server.py);
the hooks reach it over MCP at AMP_CC_MCP_URL (default http://localhost:8001/mcp).
They never open ChromaDB themselves: a hook is a short-lived process, and only
one process should own the vault.

Nothing here is allowed to break Claude Code: when AMP is not reachable the
hooks log it and do nothing. AMP_CC_DISABLE=1 turns every hook off;
AMP_CC_READ_ONLY=1 keeps paging in but saves nothing.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

DEFAULT_MCP_URL = "http://localhost:8001/mcp"
MCP_URL_ENV = "AMP_CC_MCP_URL"
KEYS_ENV = "AMP_CC_KEYS_FILE"
STATE_ENV = "AMP_CC_STATE_DIR"
DISABLE_ENV = "AMP_CC_DISABLE"
READ_ONLY_ENV = "AMP_CC_READ_ONLY"
KEYS_FILE = os.path.join(REPO, "storage", ".amp_cc_session_keys")
STATE_DIR = os.path.join(REPO, "storage", "claude_code")
SESSION_PREFIX = "cc-"


def disabled() -> bool:
    return os.environ.get(DISABLE_ENV, "").strip() == "1"


def read_only() -> bool:
    """AMP_CC_READ_ONLY=1: memories are paged in, nothing is saved (a sensitive
    session, or an experiment that must not change the vault)."""
    return os.environ.get(READ_ONLY_ENV, "").strip() == "1"


def mcp_url() -> str:
    return os.environ.get(MCP_URL_ENV) or DEFAULT_MCP_URL


def state_dir() -> str:
    return os.environ.get(STATE_ENV) or STATE_DIR


def setup_env() -> None:
    """Point this process's AMP code at the Claude Code keyring and the running
    memory server, and load the signing key (so stored memories are verified).
    Call once, before using harness.mcp_client or harness.local_keys."""
    os.environ["AMP_SESSION_KEYS_FILE"] = os.environ.get(KEYS_ENV) or KEYS_FILE
    os.environ["AMP_MCP_URL"] = mcp_url()
    from harness.local_keys import ensure_signing_key
    ensure_signing_key()


def project_dir(path: str) -> str:
    return os.path.realpath(os.path.expanduser(path or "."))


def session_id(path: str) -> str:
    """The AMP session of a project folder. Stable for the same folder."""
    digest = hashlib.sha256(project_dir(path).encode("utf-8")).hexdigest()
    return SESSION_PREFIX + digest[:20]


def session_key(session: str) -> str:
    """The project's key from the Claude Code keyring (made on first use)."""
    from harness.local_keys import session_key as keyring_key
    return keyring_key(session)


def memory_server(fast: bool = False):
    """The memory server's tools over MCP (harness/mcp_client.py). Tests replace this.
    fast=True is for hooks Claude Code waits on: short timeouts, so a stopped
    memory server costs the user a few seconds at most."""
    from harness.mcp_client import MCPMemoryClient, memory_server as connect
    if fast:
        return MCPMemoryClient(mcp_url(), call_timeout=8.0, connect_timeout=3.0)
    return connect()


# ---------------------------------------------------------------------------
# Saved progress (one small JSON file per project)
# ---------------------------------------------------------------------------

def _state_path(session: str) -> str:
    return os.path.join(state_dir(), f"{session}.json")


def load_state(session: str) -> Dict[str, Any]:
    try:
        with open(_state_path(session), encoding="utf-8") as handle:
            state = json.load(handle)
    except (FileNotFoundError, ValueError):
        state = {}
    state.setdefault("transcripts", {})      # conversation file -> bytes already saved
    state.setdefault("files", {})            # code file -> SHA-256 of the content stored
    state.setdefault("stored", {"text": 0, "code": 0})
    return state


def save_state(session: str, state: Dict[str, Any]) -> None:
    """Write atomically, owner-only."""
    folder = state_dir()
    os.makedirs(folder, mode=0o700, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{session}.", dir=folder)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=1, sort_keys=True)
        os.replace(tmp, _state_path(session))
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


@contextmanager
def project_lock(session: str, wait: bool = True) -> Iterator[bool]:
    """One page-out at a time per project. Yields False (without waiting) when
    wait=False and another page-out holds the lock."""
    folder = state_dir()
    os.makedirs(folder, mode=0o700, exist_ok=True)
    with open(os.path.join(folder, f".{session}.lock"), "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def log(message: str) -> None:
    """One line in storage/claude_code/amp_cc.log. Never raises."""
    try:
        folder = state_dir()
        os.makedirs(folder, mode=0o700, exist_ok=True)
        with open(os.path.join(folder, "amp_cc.log"), "a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{os.getpid()}] {message}\n")
    except OSError:
        pass
