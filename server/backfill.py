"""
server/backfill.py: re-score (and optionally sign) the memories already in the vault
===================================================================================
Memories stored before risk scoring have no risk_score (they would read as
"safe"), memories stored before signing have no signature (they read as
unverified, READ_ONLY), and a better risk model can raise old scores. This
script scores every stored memory again with score_payload() and writes the
result back. Scores only ever go up.

    python -m server.backfill                          # dry run: report only, change nothing
    python -m server.backfill --apply                  # write the new scores
    python -m server.backfill --apply --sign-unsigned  # also sign memories that have no signature

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
import sys
from collections import Counter
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def backfill(apply: bool = False, sign_unsigned: bool = False) -> Dict[str, object]:
    """Re-score every stored memory. Returns counts and the IDs that need a look."""
    from harness.guardrails import UnifiedMemory
    from harness.signing import SIGNING_KEY_ENV, sign_memory, verify_memory
    from server import amp_memory_server as server

    have_key = bool(os.environ.get(SIGNING_KEY_ENV))
    data = server.collection.get()
    stats: Counter = Counter()
    not_verifying: List[str] = []
    now_high_risk: List[str] = []
    changes: List[str] = []

    for chroma_id, meta in zip(data.get("ids", []), data.get("metadatas", [])):
        stats["memories"] += 1
        if not meta or "envelope" not in meta:
            stats["skipped: no envelope"] += 1
            continue
        stored = json.loads(meta["envelope"])
        memory = UnifiedMemory.model_validate(stored)
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

        if unscored or raised or sign:
            stats["to update" if not apply else "updated"] += 1
            if apply:
                server.collection.update(ids=[chroma_id], metadatas=[{**meta, "envelope": memory.model_dump_json()}])

    stats["signature does not verify (left as is)"] = len(not_verifying)
    return {"counts": dict(stats), "changes": changes, "now_high_risk": now_high_risk,
            "not_verifying": not_verifying, "applied": apply}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m server.backfill", description=__doc__.split("\n\n")[1])
    parser.add_argument("--apply", action="store_true", help="write the changes (default: dry run)")
    parser.add_argument("--sign-unsigned", action="store_true",
                        help="also sign memories that have no signature (vouches for their content)")
    parser.add_argument("--vault", help="ChromaDB folder (default: AMP_CHROMA_PATH, else storage/chroma_db)")
    args = parser.parse_args(argv)
    if args.vault:
        os.environ["AMP_CHROMA_PATH"] = os.path.abspath(args.vault)

    logging.getLogger("httpx").setLevel(logging.WARNING)     # one INFO line per model call otherwise
    from harness.local_keys import ensure_signing_key
    from harness.risk_model import model_name
    key_source = ensure_signing_key()
    print(f"Signing key {key_source}; risk model: {model_name() or 'off (heuristic only)'}")
    report = backfill(apply=args.apply, sign_unsigned=args.sign_unsigned)
    for name, value in report["counts"].items():
        print(f"  {name}: {value}")
    for line in report["changes"]:
        print(f"    score {line}")
    if report["now_high_risk"]:
        print("  now high risk (hidden from the model): " + ", ".join(report["now_high_risk"]))
    if report["not_verifying"]:
        print("  signature does not verify, look at these: " + ", ".join(report["not_verifying"]))
    if not args.apply:
        print("Dry run: nothing was written. Add --apply to write the changes.")


if __name__ == "__main__":
    main()
