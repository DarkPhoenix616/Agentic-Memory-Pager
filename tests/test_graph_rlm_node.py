"""
tests/test_graph_rlm_node.py — rlm_summarizer_node (harness/graph.py) with the
real RLM summarizer plugged in.

The MCP store and the Ollama model are replaced by fakes, so neither ChromaDB
nor a running model is needed. The test is skipped automatically if langgraph
is not installed.

Run with:
    pytest tests/test_graph_rlm_node.py -v
"""
import json

import pytest

pytest.importorskip("langgraph", reason="harness.graph needs langgraph")
graph = pytest.importorskip("harness.graph")

from compactor.rlm_text import recursive_summarizer as rs  # noqa: E402

TRANSCRIPT = "\n".join(
    [
        "User: We are building the memory pager. Never use SQLite for the vault.",
        "Assistant: I decided to keep the vault in ChromaDB because it runs locally.",
        "User: page_context() fails with KeyError: 'session_id' on a new session.",
        "Assistant: Fixed: page_context() now gives session_id a default value.",
    ] * 12
)


def fake_model(system, user, schema):
    """Answers like a model that read the transcript."""
    if "<transcript>" in user:
        return json.dumps({
            "summary": "The team chose ChromaDB for the vault and fixed a KeyError in page_context().",
            "architectural_decisions": ["Keep the vault in ChromaDB because it runs locally."],
            "bugs_resolved": ["page_context() KeyError 'session_id': no default -> default added"],
            "user_constraints": ["Never use SQLite for the vault."],
        })
    return json.dumps({"summary": "Merged: ChromaDB vault chosen, page_context() KeyError fixed."})


@pytest.fixture
def stored(monkeypatch):
    """Capture what the node sends to the MCP server instead of writing to ChromaDB."""
    saved = []

    def fake_store(memory):
        saved.append(memory)
        return memory.model_copy(update={"vector_embedding_id": "test-id"})

    monkeypatch.setattr(graph, "_mcp_store_memory", fake_store)
    return saved


def use_model(monkeypatch, model):
    monkeypatch.setattr(rs, "default_llm", lambda config: model)


def make_state(chunks):
    return {
        "session_id": "test-session",
        "context_chunks": chunks,
        "current_token_count": sum(graph._count_tokens(c["content"]) for c in chunks),
    }


def test_oldest_text_chunk_is_summarized_stored_and_evicted(stored, monkeypatch):
    use_model(monkeypatch, fake_model)
    text_chunk = {"role": "user", "type": "text", "content": TRANSCRIPT}
    code_chunk = {"role": "user", "type": "code", "content": "def f():\n    return 1\n"}
    state = make_state([text_chunk, code_chunk])

    result = graph.rlm_summarizer_node(state)

    # The raw chunk is evicted and its tokens are freed
    assert result["context_chunks"] == [code_chunk]
    assert result["current_token_count"] == state["current_token_count"] - graph._count_tokens(TRANSCRIPT)

    # Exactly one text memory was stored, built by the RLM summarizer
    assert len(stored) == 1
    memory = stored[0]
    assert memory.memory_type == "text" and memory.compaction_source == "rlm"
    payload = memory.text_payload
    assert "ChromaDB" in payload.factual_summary
    assert payload.architectural_decisions == ["Keep the vault in ChromaDB because it runs locally."]
    assert payload.user_constraints == ["Never use SQLite for the vault."]
    assert payload.bugs_resolved and payload.timeframe.startswith("messages 1-48")
    assert payload.original_token_count == graph._count_tokens(TRANSCRIPT)


def test_ollama_down_still_stores_a_memory(stored, monkeypatch):
    def down(system, user, schema):
        raise ConnectionError("Failed to connect to Ollama")

    use_model(monkeypatch, down)
    graph.rlm_summarizer_node(make_state([{"role": "user", "type": "text", "content": TRANSCRIPT}]))

    payload = stored[0].text_payload
    assert "Never use SQLite for the vault." in payload.user_constraints
    assert len(payload.factual_summary) >= 10


def test_single_short_message_is_stored_as_written(stored, monkeypatch):
    use_model(monkeypatch, None)
    graph.rlm_summarizer_node(make_state([{"role": "user", "type": "text", "content": "Please add caching."}]))
    payload = stored[0].text_payload
    assert payload.factual_summary == "Please add caching."
    assert payload.timeframe.startswith("1 user message")


def test_no_chunks_is_a_no_op(stored):
    assert graph.rlm_summarizer_node(make_state([])) == {}
    assert stored == []
