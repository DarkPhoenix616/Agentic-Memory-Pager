"""
Shared test setup.

No test touches the user's real vault (see the AMP_CHROMA_PATH line below) or
the real key files under storage/ (AMP_ADMIN_TOKEN is set; tests that sign set
AMP_SIGNING_KEY themselves), no test calls the risk model (AMP_RISK_MODEL=off;
tests/test_risk_model.py uses a fake Ollama client), and no test talks to a real
Ollama chat model unless it sets one up itself:
harness.chat_model's client is replaced by a stub that answers like the old
"[LLM Stub]" in llm_reasoning_node. Tests that need other replies patch
chat_model._client (or graph._invoke_llm) themselves; their patch wins.
"""
import os
import tempfile
from types import SimpleNamespace

import pytest

# Safety net: if any test reaches the real memory server in-process, it opens a
# throwaway vault, never the user's storage/chroma_db. (The server module reads
# AMP_CHROMA_PATH when it is first imported, so this must run before any test.)
os.environ["AMP_CHROMA_PATH"] = tempfile.mkdtemp(prefix="amp-test-vault-")
os.environ.pop("AMP_MCP_URL", None)     # tests choose the transport themselves
os.environ["AMP_RISK_MODEL"] = "off"    # heuristic-only scores, as the existing tests expect
os.environ["AMP_ADMIN_TOKEN"] = "test-admin-token"
os.environ["AMP_AUDIT_LOG"] = os.path.join(tempfile.mkdtemp(prefix="amp-test-audit-"), "amp_audit.log")
os.environ["AMP_SESSION_KEYS_FILE"] = os.path.join(tempfile.mkdtemp(prefix="amp-test-keys-"), ".amp_session_keys")


class _StubChatClient:
    def chat(self, model, messages, **kwargs):
        question = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return SimpleNamespace(message=SimpleNamespace(content=f"[LLM Stub] Responding to: '{question[:200]}'."))

    def list(self):
        return SimpleNamespace(models=[SimpleNamespace(model="llama3.1:8b")])


@pytest.fixture(autouse=True)
def _no_real_chat_model(monkeypatch):
    try:
        from harness import chat_model
    except ImportError:          # langchain-core not installed: nothing can reach the model anyway
        return
    monkeypatch.setattr(chat_model, "_client", lambda host: _StubChatClient())
