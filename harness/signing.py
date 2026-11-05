"""
harness/signing.py: integrity signatures for stored memories (Phase 3, T4)
==========================================================================
A memory is signed when it is created and verified when it is retrieved, so an
edit made directly in ChromaDB (e.g. planting instructions in a stored summary,
or lowering risk_score) is detected.

Algorithm: HMAC-SHA256 keyed by the AMP_SIGNING_KEY environment variable. A
plain hash would not help: anyone who can edit a row can recompute it.

The signature covers: id, session_id, memory_type, compaction_source,
provenance, risk_score and the payload. It does NOT cover vector_embedding_id
(assigned after signing) or the signature itself.

Behaviour without a key: sign_memory() returns None (logged) and
verify_memory() returns False. Callers treat unverified memories as READ_ONLY;
nothing is raised, so a missing key never crashes the pipeline.

Any code that changes a signed field (e.g. T1 raising risk_score in
store_memory) must call sign_memory() again afterwards.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from typing import Optional

from harness.guardrails import UnifiedMemory

logger = logging.getLogger(__name__)

SIGNING_KEY_ENV = "AMP_SIGNING_KEY"
_PREFIX = "hmac-sha256:"


def _key() -> Optional[bytes]:
    key = os.environ.get(SIGNING_KEY_ENV, "")
    return key.encode("utf-8") if key else None


def _canonical_bytes(memory: UnifiedMemory) -> bytes:
    payload = memory.text_payload if memory.memory_type == "text" else memory.code_payload
    document = {
        "id": memory.id,
        "session_id": memory.session_id,
        "memory_type": memory.memory_type,
        "compaction_source": memory.compaction_source,
        "provenance": memory.provenance,
        "risk_score": memory.risk_score,
        "payload": payload.model_dump(mode="json") if payload is not None else None,
    }
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def sign_memory(memory: UnifiedMemory) -> Optional[str]:
    """Return the signature for `memory`, or None if no signing key is configured."""
    key = _key()
    if key is None:
        logger.warning("[Signing] %s is not set; memory %s left unsigned.", SIGNING_KEY_ENV, memory.id)
        return None
    return _PREFIX + hmac.new(key, _canonical_bytes(memory), hashlib.sha256).hexdigest()


def verify_memory(memory: UnifiedMemory) -> bool:
    """True only if the memory carries a signature that matches its current content."""
    key = _key()
    if key is None or not memory.signature:
        return False
    expected = _PREFIX + hmac.new(key, _canonical_bytes(memory), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, memory.signature)
