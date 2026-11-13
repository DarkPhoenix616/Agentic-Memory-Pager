"""
harness/chat_store.py - the unsaved part of each chat, kept on disk
===================================================================
A chat's context window lives in memory until it is paged out to the vault.
If the app stops (Ctrl+C, a closed terminal, a crash, a power cut) before the
chat is left, that part would be lost. So after every turn and every
compaction, ChatSession writes what is still only in its context window here:

    storage/chats/<session_id>.pending      (AMP_CHATS_DIR overrides the folder)

Each file is encrypted with the chat's own session key (AES-256-GCM, the same
scheme as the vault, server/session_crypto.py), so it is as private as the
chat's memories. Files are written atomically and are owner-only (0600).

When the chat is left normally, everything is paged out to the vault and the
file is removed. When the chat is reopened, its file is read back into the
context window, so it continues exactly where it stopped.

A second file keeps the chat's messages for display, like the history of
ChatGPT or Claude (secrets scrubbed, same key, same protection):

    storage/chats/<session_id>.transcript

It stays after the chat is left; reopening the chat shows the old messages
again. It is never given to the model: the model sees its context window and
the memories paged in from the vault, as before.

No model is needed for any of this: a write is a few milliseconds.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from harness.local_keys import STORAGE_DIR, session_key
from server.session_crypto import decrypt, encrypt

CHATS_DIR_ENV = "AMP_CHATS_DIR"
CHATS_DIR = os.path.join(STORAGE_DIR, "chats")
SUFFIX = ".pending"
TRANSCRIPT_SUFFIX = ".transcript"
VERSION = 1
RECORD_ID = "pending-context"          # bound into the encryption, so the file can't be swapped with a memory
TRANSCRIPT_RECORD_ID = "transcript"    # ...or with the other file
_SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")   # same rule as UnifiedMemory.session_id
_lock = threading.Lock()


def chats_dir() -> str:
    return os.environ.get(CHATS_DIR_ENV) or CHATS_DIR


def _path(session_id: str, suffix: str = SUFFIX) -> str:
    if not _SESSION_ID_RE.match(session_id or ""):
        raise ValueError(f"Not a chat ID: {session_id!r}")
    return os.path.join(chats_dir(), session_id + suffix)


def save_pending(session_id: str, state: Dict[str, Any]) -> None:
    """Encrypt `state` (the chat's chunks, turn and message counter) with the
    chat's key and write it atomically. Creates the key if the chat has none yet."""
    _write(_path(session_id), session_id, RECORD_ID, state)


def _write(path: str, session_id: str, record_id: str, state: Dict[str, Any]) -> None:
    token = encrypt(json.dumps({"version": VERSION, **state}), session_key(session_id), session_id, record_id)
    folder = os.path.dirname(path)
    with _lock:
        os.makedirs(folder, mode=0o700, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{session_id}.", dir=folder)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="ascii") as handle:
                handle.write(token)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise


def load_pending(session_id: str) -> Optional[Dict[str, Any]]:
    """The saved state of the chat, or None if it has none. Raises
    SessionKeyError when the key on this computer does not open the file,
    and ValueError when the file is not a chat state."""
    path = _path(session_id)
    try:
        with open(path, encoding="ascii") as handle:
            token = handle.read().strip()
    except FileNotFoundError:
        return None
    state = json.loads(decrypt(token, session_key(session_id, create=False), session_id, RECORD_ID))
    chunks = state.get("chunks") if isinstance(state, dict) else None
    if state.get("version") != VERSION or not isinstance(chunks, list) \
            or not all(isinstance(c, dict) and isinstance(c.get("content"), str) for c in chunks):
        raise ValueError(f"{path} is not a saved chat state.")
    return state


def save_transcript(session_id: str, messages: List[Dict[str, Any]]) -> None:
    """Keep the chat's messages for display (call with secrets already scrubbed)."""
    _write(_path(session_id, TRANSCRIPT_SUFFIX), session_id, TRANSCRIPT_RECORD_ID, {"messages": messages})


def load_transcript(session_id: str) -> Optional[List[Dict[str, Any]]]:
    """The chat's saved messages, or None if it has none. Raises like load_pending."""
    path = _path(session_id, TRANSCRIPT_SUFFIX)
    try:
        with open(path, encoding="ascii") as handle:
            token = handle.read().strip()
    except FileNotFoundError:
        return None
    state = json.loads(decrypt(token, session_key(session_id, create=False), session_id, TRANSCRIPT_RECORD_ID))
    messages = state.get("messages") if isinstance(state, dict) else None
    if state.get("version") != VERSION or not isinstance(messages, list) or not all(
            isinstance(m, dict) and isinstance(m.get("id"), str) and m.get("role") in ("user", "assistant")
            and isinstance(m.get("content"), str) for m in messages):
        raise ValueError(f"{path} is not a saved chat transcript.")
    return messages


def clear_pending(session_id: str) -> None:
    """Remove the chat's file (everything is in the vault now)."""
    with _lock:
        try:
            os.unlink(_path(session_id))
        except FileNotFoundError:
            pass


def pending_sessions(suffix: str = SUFFIX) -> Dict[str, str]:
    """Every chat with an unsaved part on disk (or, with TRANSCRIPT_SUFFIX, a saved
    transcript), with the time it was written (ISO 8601, UTC)."""
    try:
        names = os.listdir(chats_dir())
    except FileNotFoundError:
        return {}
    found = {}
    for name in names:
        session_id = name[:-len(suffix)]
        if not (name.endswith(suffix) and _SESSION_ID_RE.match(session_id)):
            continue
        try:
            stamp = os.path.getmtime(os.path.join(chats_dir(), name))
        except FileNotFoundError:                     # removed meanwhile: the chat was just left
            continue
        found[session_id] = datetime.fromtimestamp(stamp, timezone.utc).isoformat()
    return found
