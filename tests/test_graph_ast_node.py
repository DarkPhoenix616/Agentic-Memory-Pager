"""
tests/test_graph_ast_node.py — ast_compactor_node (harness/graph.py) with the
real AST compactor plugged in.

The MCP store is replaced by a fake, so ChromaDB is not needed. The test is
skipped automatically if langgraph is not installed.

Run with:
    pytest tests/test_graph_ast_node.py -v
"""
import pytest

pytest.importorskip("langgraph", reason="harness.graph needs langgraph")
graph = pytest.importorskip("harness.graph")

CODE = '''import os


def load(path: str) -> str:
    """Read a file."""
    with open(path) as handle:
        return handle.read()


class Cache:
    def get(self, key: str) -> str:
        return ""
'''


@pytest.fixture
def stored(monkeypatch):
    """Capture what the node sends to the MCP server instead of writing to ChromaDB."""
    saved = []

    def fake_store(memory):
        saved.append(memory)
        return memory.model_copy(update={"vector_embedding_id": "test-id"})

    monkeypatch.setattr(graph, "_mcp_store_memory", fake_store)
    return saved


def make_state(chunks):
    return {
        "session_id": "test-session",
        "context_chunks": chunks,
        "current_token_count": sum(graph._count_tokens(c["content"]) for c in chunks),
    }


def test_oldest_code_chunk_is_compacted_stored_and_evicted(stored):
    code_chunk = {"role": "user", "type": "code", "content": CODE,
                  "file_path": "storage/loader.py", "language": "python"}
    text_chunk = {"role": "user", "type": "text", "content": "Please add caching."}
    state = make_state([code_chunk, text_chunk])

    result = graph.ast_compactor_node(state)

    # The raw chunk is evicted and its tokens are freed
    assert result["context_chunks"] == [text_chunk]
    assert result["current_token_count"] == state["current_token_count"] - graph._count_tokens(CODE)

    # Exactly one code memory was stored, built by the AST compactor
    assert len(stored) == 1
    memory = stored[0]
    assert memory.memory_type == "code" and memory.compaction_source == "ast"
    payload = memory.code_payload
    assert payload.file_path == "storage/loader.py"
    assert payload.imports == ["import os"]
    assert [(s.name, s.type) for s in payload.symbols] == [
        ("load", "function"), ("Cache", "class"), ("Cache.get", "method")]
    assert payload.original_token_count == graph._count_tokens(CODE)
    assert "handle.read()" not in payload.model_dump_json()      # bodies are not stored


def test_unsafe_path_and_broken_fragment_do_not_crash_the_graph(stored):
    chunk = {"role": "user", "type": "code", "content": "def broken(:\n    pass\n",
             "file_path": "/home/user/../secret/app.py"}
    graph.ast_compactor_node(make_state([chunk]))

    payload = stored[0].code_payload
    assert payload.file_path == "home/user/secret/app.py"
    assert [s.name for s in payload.symbols] == ["broken"]
    assert "line scan" in payload.summary


def test_no_chunks_is_a_no_op(stored):
    assert graph.ast_compactor_node(make_state([])) == {}
    assert stored == []
