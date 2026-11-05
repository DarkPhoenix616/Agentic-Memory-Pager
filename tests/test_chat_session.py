"""
tests/test_chat_session.py — the chat app's core (harness/chat_session.py and
harness/chat_model.py) running on the real AMP graph.

The vault (ChromaDB), the Ollama chat client and the RLM model are replaced
by fakes, so no database, model or GPU is needed.

Run with:
    pytest tests/test_chat_session.py -v
"""
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("langgraph", reason="harness.graph needs langgraph")

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from compactor.rlm_text import recursive_summarizer as rs  # noqa: E402
from harness import chat_model  # noqa: E402
from harness import graph as amp  # noqa: E402
from harness.chat_model import ChatModelError, build_messages  # noqa: E402
from harness.chat_session import (  # noqa: E402
    ChatSession, MessageTooLong, split_message, turn_to_chunks,
)
from harness.guardrails import PageContextResult, RetrievedMemory  # noqa: E402

LONG_REPLY = " ".join(f"word{i}" for i in range(220))


class FakeVault:
    """In-memory stand-in for the MCP memory server."""

    def __init__(self):
        self.memories = []

    def store(self, memory):
        memory = memory.model_copy(update={"vector_embedding_id": f"mem_{len(self.memories)}"})
        self.memories.append(memory)
        return memory

    def page(self, query, total_budget, session_id):
        chosen, used = [], 0
        for memory in reversed(self.memories):
            if memory.session_id != session_id:
                continue
            text = (memory.text_payload.factual_summary if memory.text_payload
                    else memory.code_payload.file_path)
            cost = amp._count_tokens(text)
            if used + cost <= total_budget:
                chosen.append(RetrievedMemory(memory=memory, distance=0.3, token_cost=cost))
                used += cost
        return PageContextResult(memories=chosen, tokens_used=used, total_budget=total_budget)


class FakeClient:
    """Stand-in for ollama.Client: records every chat call."""

    def __init__(self, reply=lambda messages: LONG_REPLY, fail=False):
        self.reply = reply
        self.fail = fail
        self.calls = []

    def chat(self, model, messages, options, keep_alive):
        self.calls.append(messages)
        if self.fail:
            raise ConnectionError("Failed to connect to Ollama")
        return SimpleNamespace(message=SimpleNamespace(content=self.reply(messages)))

    def list(self):
        return SimpleNamespace(models=[SimpleNamespace(model="llama3.1:8b")])


def fake_rlm(system, user, schema):
    words = user.split("<transcript>", 1)[-1].split()[:12]
    return json.dumps({"summary": "Summary: " + " ".join(words), "architectural_decisions": [],
                       "bugs_resolved": [], "user_constraints": []})


@pytest.fixture
def vault(monkeypatch):
    vault = FakeVault()
    monkeypatch.setattr(amp, "_mcp_store_memory", vault.store)
    monkeypatch.setattr(amp, "_mcp_list_all_memories", lambda: list(vault.memories))
    monkeypatch.setattr(amp, "_mcp_page_context",
                        lambda query, total_budget, session_id: vault.page(query, total_budget, session_id))
    monkeypatch.setattr(rs, "default_llm", lambda config: fake_rlm)
    return vault


@pytest.fixture
def client(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(chat_model, "_client", lambda host: client)
    monkeypatch.delenv("AMP_CHAT_MODEL", raising=False)
    monkeypatch.delenv("AMP_RLM_MODEL", raising=False)
    return client


def session(**kwargs):
    options = dict(session_id="test-chat", token_limit=800, threshold=0.8)
    options.update(kwargs)
    chat = ChatSession(**options)
    return chat


# ---------------------------------------------------------------------------
# Messages -> chunks -> what the model sees
# ---------------------------------------------------------------------------

def test_a_turn_becomes_code_chunks_around_one_exchange():
    user = "Here is my code:\n```py\ndef f(x):\n    return x\n```\nWhat does it do?"
    reply = "It returns x. Try it:\n```\nprint(f(1))\n```"
    chunks = turn_to_chunks(user, "u1", reply, "a2")
    assert [(c["role"], c["type"]) for c in chunks] == [
        ("user", "code"), ("user", "text"), ("assistant", "code")]
    assert chunks[0]["language"] == "python" and chunks[0]["file_path"] == "chat/u1_1.py"
    exchange = chunks[1]
    assert exchange["content"].startswith("User: Here is my code:\n[code block 1: python]")
    assert exchange["content"][exchange["reply_at"]:].startswith("Assistant: It returns x.")
    assert exchange["msg_ids"] == ["u1", "a2"]


def test_a_bare_python_file_is_one_code_chunk():
    source = "import os\n\n\ndef load(path):\n    return open(path).read()\n"
    assert split_message(source) == ("", [("python", source)])
    assert split_message("Plain question about lists?") == ("Plain question about lists?", [])


def test_the_model_sees_alternating_turns_with_code_and_search_results():
    chunks = turn_to_chunks("Look:\n```python\nx = 1\n```", "u1", "Nice. ```js\nlet y = 2\n```", "a2")
    payload, searched = build_messages(
        "SYSTEM PROMPT",
        [HumanMessage("And now?"), SystemMessage("[Vault Search: 'x'] No results.")],
        chunks)
    roles = [m["role"] for m in payload]
    assert roles == ["system", "user", "assistant", "user"]
    assert payload[1]["content"].startswith("```python\nx = 1\n```\n\nLook:")
    assert payload[2]["content"].endswith("```javascript\nlet y = 2\n```")
    assert "[Vault Search: 'x'] No results." in payload[0]["content"]
    assert searched and "already searched" in payload[0]["content"]


def test_a_second_search_request_and_thinking_are_removed_from_the_reply():
    assert chat_model._clean_reply('TOOL: search_vault("cat")', searched=True).startswith("I searched")
    assert chat_model._clean_reply('TOOL: search_vault("cat")\nPixel.', searched=True) == "Pixel."
    assert chat_model._clean_reply("<think>hmm</think>Hello", searched=False) == "Hello"
    assert chat_model._clean_reply('TOOL: search_vault("cat")', searched=False).startswith("TOOL:")


# ---------------------------------------------------------------------------
# Whole turns through the real graph
# ---------------------------------------------------------------------------

def test_old_exchanges_are_paged_out_and_paged_back_in(vault, client):
    chat = session()
    chat.background = False                       # compact inside the turn (direct reclaim)
    for n in range(6):
        reply = chat.ask(f"Question number {n}: tell me about topic {n}.")
        assert reply == LONG_REPLY

    assert chat.compactions >= 3 and len(vault.memories) == chat.compactions
    assert all(m.memory_type == "text" and m.compaction_source == "rlm" for m in vault.memories)
    assert vault.memories[0].text_payload.factual_summary.startswith("Summary: User: Question number 0")
    assert chat.message_status("u1") == "paged out" and chat.message_status("a2") == "paged out"
    assert chat.message_status(chat.transcript[-1]["id"]) == "in context"

    # The last call to the model carried paged-in memories in its system prompt
    assert "[Memory] Summary:" in client.calls[-1][0]["content"]
    assert any(e.kind == "page_in" for e in chat.events)
    kinds = [e.kind for e in chat.events if e.turn == chat.turn]
    assert kinds[0] == "monitor" and "reply" in kinds


def test_background_reclaim_keeps_the_next_turn_free_of_compaction(vault, client):
    chat = session()
    for n in range(6):
        chat.ask(f"Question number {n}: tell me about topic {n}.")
        chat.wait_for_reclaim()
        assert chat.context_tokens <= max(chat.reclaim_target, chat.trigger_tokens)
        # the turn that just ended is the working set: never paged out in the background
        assert all(chat.message_status(e["id"]) == "in context" for e in chat.transcript[-2:])

    compactions = [e for e in chat.events if e.kind == "compaction"]
    assert compactions and all(e.data["background"] for e in compactions)
    assert len(vault.memories) == chat.compactions


def test_code_blocks_are_compacted_by_the_ast_compactor(vault, client):
    chat = session(token_limit=600)
    chat.background = False
    chat.ask("Keep this:\n```python\nimport os\n\ndef load(path: str) -> str:\n    \"\"\"Read a file.\"\"\"\n"
             "    with open(path) as f:\n        return f.read()\n```")
    for n in range(4):
        chat.ask(f"Follow-up {n}")
    code = [m for m in vault.memories if m.memory_type == "code"]
    assert code and code[0].compaction_source == "ast"
    assert [s.name for s in code[0].code_payload.symbols] == ["load"]
    assert code[0].code_payload.file_path == "chat/u1_1.py"


def test_a_long_last_turn_is_kept_even_above_the_reclaim_target(vault, client):
    client.reply = lambda messages: " ".join(f"long{i}" for i in range(600))
    chat = session(token_limit=1024)
    chat.ask("Short question one")
    chat.ask("Short question two")
    chat.wait_for_reclaim()
    assert chat.context_tokens > chat.reclaim_target                  # one big turn left...
    assert {e["id"] for e in chat.transcript[-2:]} <= {i for c in chat.chunks for i in c["msg_ids"]}
    assert chat.message_status("u1") == "paged out"                  # ...and the older one paged out


def test_a_message_too_long_for_the_window_is_rejected(vault, client):
    chat = session(token_limit=1024)
    with pytest.raises(MessageTooLong):
        chat.ask("word " * 2000)
    assert chat.turn == 0 and chat.transcript == [] and client.calls == []


def test_a_model_error_keeps_the_compactions_already_done(vault, client):
    chat = session()
    chat.background = False
    for n in range(3):
        chat.ask(f"Question number {n}")
    transcript, chunks_before = list(chat.transcript), list(chat.chunks)

    client.fail = True
    with pytest.raises(ChatModelError):
        chat.ask("This one fails")
    assert chat.transcript == transcript                        # the failed turn left no message
    assert len(chat.chunks) < len(chunks_before)                # but its compactions stayed
    assert chat.events[-1].kind == "error"


def test_a_model_that_keeps_asking_to_search_cannot_loop(vault, client):
    client.reply = lambda messages: 'TOOL: search_vault("my cat")'
    chat = session()
    reply = chat.ask("What is my cat called?")
    assert reply.startswith("I searched my memory")
    assert [e.kind for e in chat.events].count("search") == 2   # request + results, once


def test_model_status_reports_what_is_missing(client, monkeypatch):
    assert chat_model.model_status() == (True, "llama3.1:8b (local, Ollama)")
    monkeypatch.setenv("AMP_CHAT_MODEL", "phi3:mini")
    ready, message = chat_model.model_status()
    assert not ready and "ollama pull phi3:mini" in message

    def unreachable(host):
        raise ConnectionError("refused")
    monkeypatch.setattr(chat_model, "_client", unreachable)
    ready, message = chat_model.model_status()
    assert not ready and "ollama serve" in message
