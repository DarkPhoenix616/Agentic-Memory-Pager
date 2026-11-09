"""
harness/local_keys.py: the two local secrets shared by AMP Chat and the memory server
====================================================================================
    AMP_SIGNING_KEY   HMAC key that seals every memory (harness/signing.py)
    AMP_ADMIN_TOKEN   password for the admin-only vault tools
                      (delete_memory, list_all_memories)

Each one comes from its environment variable when that is set. Otherwise it is
read from a private file under storage/, which is created once (owner-only,
0600) the first time it is needed. Because the chat app and the memory server
read the same files, they always agree on both values, also when the memory
server runs as its own process (AMP_MCP_URL).
"""

from __future__ import annotations

import os
import secrets
from typing import Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORAGE_DIR = os.path.join(ROOT, "storage")

SIGNING_KEY_ENV = "AMP_SIGNING_KEY"
SIGNING_KEY_FILE = os.path.join(STORAGE_DIR, ".amp_signing_key")
ADMIN_TOKEN_ENV = "AMP_ADMIN_TOKEN"
ADMIN_TOKEN_FILE = os.path.join(STORAGE_DIR, ".amp_admin_token")


def _load_or_create(env_name: str, path: str) -> Tuple[str, str]:
    """(value, where it came from). Env wins; else the file, created once if missing."""
    value = os.environ.get(env_name, "").strip()
    if value:
        return value, f"from {env_name}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    created = False
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)   # owner-only; fails if it exists
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd, "w") as handle:
            handle.write(secrets.token_hex(32))
        created = True
    with open(path, encoding="ascii") as handle:
        value = handle.read().strip()
    if not value:
        raise RuntimeError(f"{path} is empty; delete it and a new value is created.")
    where = os.path.relpath(path, ROOT) if path.startswith(ROOT) else path
    return value, f"{'created' if created else 'loaded'} {where}"


def ensure_signing_key(path: str = SIGNING_KEY_FILE) -> str:
    """Make sure AMP_SIGNING_KEY is set in this process; return where it came from.

    harness/signing.py reads the key from the environment, so it is exported
    there. Without a key every memory is unverified and therefore READ_ONLY,
    and Layer 4 then blocks tool calls (search_vault, run_memory_code) shaped
    by memories."""
    value, source = _load_or_create(SIGNING_KEY_ENV, path)
    os.environ[SIGNING_KEY_ENV] = value
    return source


def admin_token(path: str = ADMIN_TOKEN_FILE) -> str:
    """The admin token for delete_memory and list_all_memories."""
    return _load_or_create(ADMIN_TOKEN_ENV, path)[0]


def ensure_admin_token(path: str = ADMIN_TOKEN_FILE) -> str:
    """Create the admin token file if needed; return where the token comes from."""
    return _load_or_create(ADMIN_TOKEN_ENV, path)[1]


def is_admin(token: str) -> bool:
    """True only for the right admin token (constant-time comparison)."""
    return bool(token) and secrets.compare_digest(str(token).encode("utf-8"), admin_token().encode("utf-8"))
