"""
server/session_crypto.py: per-session encryption of stored memories
===================================================================
Every chat session has its own secret session key (64 hex characters, made and
kept by the client, e.g. AMP Chat's keyring in harness/local_keys.py). The
server never stores it. From the key it derives, with HKDF-SHA256:

    the data key   AES-256-GCM key that encrypts each memory's JSON envelope
    the key check  a fingerprint stored next to each memory, so the server can
                   tell whether a caller's key is the right one for a session
                   (it cannot be turned back into the key)

Both derivations include the session ID, so one key gives different data keys
in different sessions. Each envelope is encrypted with a fresh 12-byte nonce,
and the session ID and memory ID are bound in as associated data: a ciphertext
copied to another row or session no longer decrypts.

Stored form:  "aesgcm:v1:" + base64(nonce + ciphertext)
"""

from __future__ import annotations

import base64
import hmac
import os
import re

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

PREFIX = "aesgcm:v1:"
_SALT = b"amp-session-vault-v1"
_KEY_RE = re.compile(r"^[0-9a-f]{64}$")


class SessionKeyError(PermissionError):
    """A missing, malformed or wrong session key."""


def _derive(session_key: str, session_id: str, purpose: bytes) -> bytes:
    if not isinstance(session_key, str) or not _KEY_RE.match(session_key):
        raise SessionKeyError("A session key is 64 hex characters.")
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=_SALT,
                info=purpose + b":" + session_id.encode("utf-8")).derive(bytes.fromhex(session_key))


def key_check(session_key: str, session_id: str) -> str:
    """The fingerprint stored with each memory of the session."""
    return _derive(session_key, session_id, b"check").hex()


def same_key_check(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("ascii"), b.encode("ascii"))


def _aad(session_id: str, memory_id: str) -> bytes:
    return f"{session_id}\x00{memory_id}".encode("utf-8")


def encrypt(plaintext: str, session_key: str, session_id: str, memory_id: str) -> str:
    nonce = os.urandom(12)
    data_key = _derive(session_key, session_id, b"data")
    sealed = AESGCM(data_key).encrypt(nonce, plaintext.encode("utf-8"), _aad(session_id, memory_id))
    return PREFIX + base64.b64encode(nonce + sealed).decode("ascii")


def decrypt(token: str, session_key: str, session_id: str, memory_id: str) -> str:
    """The plaintext, or SessionKeyError if the key, the session or the row is wrong."""
    if not token.startswith(PREFIX):
        raise SessionKeyError("Not an encrypted memory.")
    raw = base64.b64decode(token[len(PREFIX):])
    data_key = _derive(session_key, session_id, b"data")
    try:
        return AESGCM(data_key).decrypt(raw[:12], raw[12:], _aad(session_id, memory_id)).decode("utf-8")
    except InvalidTag as exc:
        raise SessionKeyError("This memory does not decrypt with that key.") from exc
