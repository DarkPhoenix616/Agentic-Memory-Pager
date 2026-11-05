"""
Shared test setup.

No test talks to a real Ollama chat model unless it sets one up itself:
harness.chat_model's client is replaced by a stub that answers like the old
"[LLM Stub]" in llm_reasoning_node. Tests that need other replies patch
chat_model._client (or graph._invoke_llm) themselves; their patch wins.
"""
from types import SimpleNamespace

import pytest


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
