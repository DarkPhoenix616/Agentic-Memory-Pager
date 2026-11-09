"""
server/backfill.py: re-score, encrypt (and optionally sign) the memories already in the vault
=============================================================================================
Memories stored before risk scoring have no risk_score (they would read as
"safe"), memories stored before signing have no signature (they read as
unverified, READ_ONLY), memories stored before session keys are not encrypted
(the server does not serve them), and a better risk model can raise old scores.
This script scores every stored memory again with score_payload(), encrypts the
plain ones, and writes the result back. Scores only ever go up.

    python -m server.backfill                          # dry run: report only, change nothing
    python -m server.backfill --apply                  # write the new scores, encrypt plain rows
    python -m server.backfill --apply --sign-unsigned  # also sign memories that have no signature
    python -m server.backfill --rebuild                # only the clean-up below

Encryption:
  - a plain row is encrypted with its session's key from the local keyring
    (storage/.amp_session_keys); a session with no key gets a new one, saved
    to the keyring (only with --apply);
  - an encrypted row is read with the keyring's key; a session whose key is
    not in the keyring cannot be read and is skipped;
  - the row keeps its search vector; its readable text is removed;
  - ChromaDB still holds the old text in its own write-ahead log
    (embeddings_queue) and in free pages of its database file. So after
    encrypting plain rows, --apply also rebuilds the vault: every row is copied
    into a fresh collection, the old collection and its log are dropped, and the
    database file is compacted (VACUUM). --rebuild runs only this step.

Signing:
  - a memory whose signature verifies is signed again when its score changes;
  - --sign-unsigned also signs memories that have no signature at all. That
    vouches for their content, so it is opt-in; their scores are updated
    first, so a high-risk one stays hidden either way;
  - a memory whose signature does NOT verify is never signed (it may have
    been edited); it is listed so it can be looked at or deleted.

--vault PATH works on another ChromaDB folder (default: storage/chroma_db, or AMP_CHROMA_PATH).
Run it while the chat app and the memory server are stopped: it writes to ChromaDB directly.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import sys
from collections import Counter
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def backfill(apply: bool = False, sign_unsigned: bool = False) -> Dict[str, object]:
    """Re-score and encrypt every stored memory. Returns counts and the IDs that need a look."""
    from harness.guardrails import UnifiedMemory
    from harness.local_keys import known_sessions, session_key
    from harness.signing import SIGNING_KEY_ENV, sign_memory, verify_memory
    from server import amp_memory_server as server
    from server import session_crypto
    from server.session_crypto import SessionKeyError

    have_key = bool(os.environ.get(SIGNING_KEY_ENV))
    data = server.collection.get(include=["metadatas", "embeddings", "documents"])
    keyring = known_sessions()
    new_keys: Dict[str, str] = {}           # sessions given a key by this run
    encrypted_sessions = {m.get("session_id") for m in data.get("metadatas") or [] if m and m.get("encrypted")}
    stats: Counter = Counter()
    not_verifying: List[str] = []
    now_high_risk: List[str] = []
    no_key: List[str] = []
    changes: List[str] = []

    rows = zip(data.get("ids", []), data.get("metadatas", []), data.get("embeddings", []), data.get("documents", []))
    for chroma_id, meta, embedding, document in rows:
        stats["memories"] += 1
        if not meta or not (meta.get("encrypted") or "envelope" in meta):
            stats["skipped: no envelope"] += 1
            continue
        session_id = str(meta.get("session_id", ""))
        key = keyring.get(session_id) or new_keys.get(session_id)
        was_encrypted = bool(meta.get("encrypted"))

        if was_encrypted:
            if not key:
                no_key.append(chroma_id)
                continue
            try:
                envelope = session_crypto.decrypt(meta["envelope_enc"], key, session_id, chroma_id)
            except SessionKeyError:
                no_key.append(chroma_id)
                continue
        else:
            envelope = meta["envelope"]
            if not key:
                if session_id in encrypted_sessions:        # its key exists somewhere else: never make a second one
                    no_key.append(chroma_id)
                    continue
                # a new key for this session: saved to the keyring only when applying
                key = session_key(session_id) if apply else secrets.token_hex(32)
                new_keys[session_id] = key

        stored = json.loads(envelope)
        memory = UnifiedMemory.model_validate(stored)
        memory.vector_embedding_id = chroma_id
        unscored = "risk_score" not in stored
        verified = verify_memory(memory)
        if memory.signature and have_key and not verified:
            not_verifying.append(chroma_id)

        old_score = memory.risk_score
        raised = server._rescore(memory)
        if unscored:
            stats["had no risk score"] += 1
        if raised:
            stats["score raised"] += 1
            changes.append(f"{chroma_id}: {'none' if unscored else old_score} -> {memory.risk_score} "
                           f"({memory.risk_level})")
        elif unscored:
            changes.append(f"{chroma_id}: none -> {memory.risk_score} ({memory.risk_level})")
        if raised and server._is_high_risk(memory):
            now_high_risk.append(chroma_id)

        sign = False
        if raised and verified:
            sign = True
            stats["signed again (score changed)"] += 1
        elif not memory.signature and sign_unsigned and have_key:
            sign = True
            stats["signed (had no signature)"] += 1
        if sign:
            memory.signature = sign_memory(memory)

        if not was_encrypted:
            stats["encrypted (were plain)" if apply else "to encrypt (plain now)"] += 1
        if not (unscored or raised or sign or not was_encrypted):
            continue
        stats["updated" if apply else "to update"] += 1
        if not apply:
            continue
        new_meta = server._row_metadata(memory, chroma_id, key, session_crypto.key_check(key, session_id))
        if was_encrypted:
            server.collection.update(ids=[chroma_id], metadatas=[new_meta])
        else:
            # ChromaDB keeps a row's text on update, so the plain row is replaced:
            # same ID and search vector, no text, encrypted envelope.
            server.collection.delete(ids=[chroma_id])
            try:
                server.collection.add(ids=[chroma_id], embeddings=[embedding], metadatas=[new_meta])
            except Exception:
                server.collection.add(ids=[chroma_id], embeddings=[embedding], metadatas=[meta],
                                      documents=[document] if document else None)
                raise

    stats["signature does not verify (left as is)"] = len(not_verifying)
    if no_key:
        stats["skipped: encrypted with a key not in the keyring"] = len(no_key)
    if new_keys:
        stats["new session keys" + ("" if apply else " (would be made)")] = len(new_keys)
    return {"counts": dict(stats), "changes": changes, "now_high_risk": now_high_risk,
            "not_verifying": not_verifying, "no_key": no_key, "new_key_sessions": sorted(new_keys),
            "applied": apply}


REBUILD_NAME = "amp_memory_rebuild"


def rebuild() -> int:
    """Copy every row into a fresh collection, drop the old one with its
    write-ahead log, and compact the database file. Returns the rows copied.
    Refuses while any row is still readable (run --apply first)."""
    import sqlite3
    from server import amp_memory_server as server

    client = server.chroma_client
    names = {getattr(c, "name", c) for c in client.list_collections()}
    if REBUILD_NAME in names:                  # left over from an interrupted run
        if server.collection.count() == 0:
            client.get_collection(REBUILD_NAME, embedding_function=server.EMBED).modify(name="amp_memory")
            server.collection = client.get_collection("amp_memory", embedding_function=server.EMBED)
        else:
            client.delete_collection(REBUILD_NAME)

    data = server.collection.get(include=["metadatas", "embeddings", "documents"])
    ids = list(data.get("ids", []))
    metas = list(data.get("metadatas") or [])
    if any(d is not None for d in data.get("documents") or []) or any(m and "envelope" in m for m in metas):
        raise RuntimeError("Some rows are still readable; run python -m server.backfill --apply first.")

    fresh = client.create_collection(REBUILD_NAME, embedding_function=server.EMBED)
    embeddings = list(data.get("embeddings") if data.get("embeddings") is not None else [])
    for start in range(0, len(ids), 256):
        fresh.add(ids=ids[start:start + 256], embeddings=embeddings[start:start + 256],
                  metadatas=metas[start:start + 256])
    copied = fresh.get(include=["metadatas"])
    if dict(zip(copied["ids"], copied["metadatas"])) != dict(zip(ids, metas)):
        client.delete_collection(REBUILD_NAME)
        raise RuntimeError("The copy does not match the vault; nothing was changed.")

    client.delete_collection("amp_memory")
    fresh.modify(name="amp_memory")
    server.collection = client.get_collection("amp_memory", embedding_function=server.EMBED)
    with sqlite3.connect(os.path.join(server.CHROMA_DB_PATH, "chroma.sqlite3")) as con:
        con.execute("VACUUM")
    return len(ids)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m server.backfill", description=__doc__.split("\n\n")[1])
    parser.add_argument("--apply", action="store_true", help="write the changes (default: dry run)")
    parser.add_argument("--sign-unsigned", action="store_true",
                        help="also sign memories that have no signature (vouches for their content)")
    parser.add_argument("--rebuild", action="store_true",
                        help="only rebuild the vault, so ChromaDB keeps no old copies of rows")
    parser.add_argument("--vault", help="ChromaDB folder (default: AMP_CHROMA_PATH, else storage/chroma_db)")
    args = parser.parse_args(argv)
    if args.vault:
        os.environ["AMP_CHROMA_PATH"] = os.path.abspath(args.vault)

    logging.getLogger("httpx").setLevel(logging.WARNING)     # one INFO line per model call otherwise
    from harness.local_keys import ensure_signing_key, keyring_path
    from harness.risk_model import model_name
    if args.rebuild:
        print(f"Rebuilt the vault: {rebuild()} rows copied; old copies and the database's free space removed.")
        return
    key_source = ensure_signing_key()
    print(f"Signing key {key_source}; session keyring {keyring_path()}; "
          f"risk model: {model_name() or 'off (heuristic only)'}")
    report = backfill(apply=args.apply, sign_unsigned=args.sign_unsigned)
    for name, value in report["counts"].items():
        print(f"  {name}: {value}")
    for line in report["changes"]:
        print(f"    score {line}")
    if report["new_key_sessions"]:
        print("  sessions given a new key: " + ", ".join(report["new_key_sessions"]))
    if report["now_high_risk"]:
        print("  now high risk (hidden from the model): " + ", ".join(report["now_high_risk"]))
    if report["not_verifying"]:
        print("  signature does not verify, look at these: " + ", ".join(report["not_verifying"]))
    if report["no_key"]:
        print("  cannot be read (no key in the keyring): " + ", ".join(report["no_key"]))
    if not args.apply:
        print("Dry run: nothing was written. Add --apply to write the changes.")
    elif report["counts"].get("encrypted (were plain)"):
        print(f"  rebuilt the vault ({rebuild()} rows), so ChromaDB keeps no readable copy of the old rows")


if __name__ == "__main__":
    main()
