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
from harness.guardrails import PageContextResult, RetrievedMemory, SessionInfo  # noqa: E402

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

    def search(self, query, top_k, session_id):
        return self.page(query, 10_000, session_id).memories[:top_k]

    def delete(self, memory_id):
        self.memories = [m for m in self.memories if m.vector_embedding_id != memory_id]
        return f"Deleted {memory_id}"

    def session_memories(self, session_id):
        return sorted((m for m in self.memories if m.session_id == session_id),
                      key=lambda m: m.timestamp, reverse=True)

    def sessions(self):
        by_id = {}
        for m in self.memories:
            by_id.setdefault(m.session_id, []).append(m.timestamp.isoformat())
        infos = [SessionInfo(session_id=sid, memories=len(times), first_time=min(times), last_time=max(times),
                             encrypted=True) for sid, times in by_id.items()]
        return sorted(infos, key=lambda info: info.last_time, reverse=True)


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
    monkeypatch.setattr(amp, "_mcp_list_session_memories", vault.session_memories)
    monkeypatch.setattr(amp, "_mcp_list_sessions", vault.sessions)
    monkeypatch.setattr(amp, "_mcp_page_context",
                        lambda query, total_budget, session_id: vault.page(query, total_budget, session_id))
    monkeypatch.setattr(amp, "_mcp_search_vault",
                        lambda query, top_k, session_id: vault.search(query, top_k, session_id))
    monkeypatch.setattr(amp, "_mcp_delete_memory", vault.delete)
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
    # search results are user-role data, never system text
    assert payload[3]["content"] == "And now?\n\n[Vault Search: 'x'] No results."
    assert "[Vault Search" not in payload[0]["content"]
    assert searched and "already searched" in payload[0]["content"]


def test_search_results_sent_as_tagged_human_messages_are_recognised():
    card = HumanMessage("<UNTRUSTED_MEMORY>Pixel</UNTRUSTED_MEMORY>", additional_kwargs={"amp_source": "memory"})
    payload, searched = build_messages("SYSTEM PROMPT", [HumanMessage("My cat?"), card])
    assert searched and payload[-1] == {"role": "user", "content": "My cat?\n\n<UNTRUSTED_MEMORY>Pixel</UNTRUSTED_MEMORY>"}
    assert "Pixel" not in payload[0]["content"]


def test_a_second_search_request_and_thinking_are_removed_from_the_reply():
    assert chat_model._clean_reply('TOOL: search_vault("cat")', searched=True).startswith("I searched")
    assert chat_model._clean_reply('TOOL: search_vault("cat")\nPixel.', searched=True) == "Pixel."
    assert chat_model._clean_reply("<think>hmm</think>Hello", searched=False) == "Hello"
    assert chat_model._clean_reply('TOOL: search_vault("cat")', searched=False).startswith("TOOL:")


def test_memory_markup_is_removed_from_answers():
    reply = ('The memory entry is:\n<MEMORY_ENTRY id="chat-1-2" kind="text" verified="true">\n'
             'We use PostgreSQL.\n</MEMORY_ENTRY>')
    assert chat_model._clean_reply(reply, searched=False) == "The memory entry is:\n\nWe use PostgreSQL."
    assert "UNTRUSTED" not in chat_model._clean_reply('<UNTRUSTED_MEMORY source="x">hi</UNTRUSTED_MEMORY>', False)


def test_invented_tool_calls_are_removed_from_answers():
    clean = chat_model._clean_reply
    assert clean('Your lucky number is 47. TOOL: recall_fact("<lucky number> = 47")', searched=False) == \
        "Your lucky number is 47."
    assert clean("We use ChromaDB.\nTOOL: store_memory(x)\nAnything else?", searched=False) == \
        "We use ChromaDB.\n\nAnything else?"
    assert clean('TOOL: recall_fact("cat")', searched=False) == "(The model returned no answer.)"


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
    assert "Summary: User: Question number" in client.calls[-1][0]["content"]
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


# ---------------------------------------------------------------------------
# Leaving a chat and reopening it
# ---------------------------------------------------------------------------

def test_leaving_a_chat_saves_everything_still_in_its_window(vault, client):
    chat = session(token_limit=4000)                  # big window: nothing is compacted while chatting
    chat.ask("My cat is called Pixel.")
    chat.ask("And my codename is ZEBRAFISH.")
    assert chat.compactions == 0 and chat.chunks
    assert chat.save_all() is True
    assert chat.chunks == [] and chat.compactions == 2 and len(vault.memories) == 2
    assert all(chat.message_status(e["id"]) == "paged out" for e in chat.transcript)
    titles = [e.title for e in chat.events if e.kind == "compaction"]
    assert len(titles) == 2 and all("before leaving the chat" in t for t in titles)


def test_a_failed_save_keeps_the_chat_in_the_window(vault, client, monkeypatch):
    chat = session(token_limit=4000)
    chat.ask("My cat is called Pixel.")

    def broken(state):
        raise ConnectionError("Ollama is not running")
    monkeypatch.setattr(amp, "rlm_summarizer_node", broken)
    assert chat.save_all() is False
    assert chat.chunks and vault.memories == []
    assert any(e.kind == "error" and "before leaving the chat" in e.title for e in chat.events)


def test_a_reopened_chat_shows_where_it_left_off(vault, client):
    first = session(session_id="chat-old", token_limit=4000)
    for n in range(7):
        first.ask(f"Fact number {n}: remember topic {n}.")
    first.save_all()

    again = ChatSession(session_id="chat-old")
    notes = again.resume(limit=5)
    assert len(notes) == 5 and again.resumed == notes
    assert "Fact number 6" in notes[0]["text"] and "Fact number 2" in notes[-1]["text"]   # newest first
    assert len(again.vault()) == 7 and again.transcript == []
    assert again.events[-1].kind == "resume" and "7 memories" in again.events[-1].title
    again.ask("What was fact number 1?")
    assert "Fact number" in client.calls[-1][0]["content"]          # its memories are paged in again


def test_only_chats_with_a_key_on_this_computer_are_listed(vault, client):
    from harness.chat_session import list_chats
    from harness.local_keys import _write_keyring, keyring_path, known_sessions
    for sid in ("chat-mine", "chat-theirs"):
        chat = session(session_id=sid, token_limit=4000)
        chat.ask(f"User: Hello from {sid}, my dog is Bruno.")
        chat.save_all()
    keys = known_sessions()
    keys.pop("chat-theirs")                          # its key is on another computer, not in this keyring
    _write_keyring(keyring_path(), keys)
    chats = list_chats(current_id="chat-mine")
    assert [c["session_id"] for c in chats] == ["chat-mine"]
    assert chats[0]["current"] and chats[0]["memories"] == 1
    assert "Hello from chat-mine" in chats[0]["title"]


def test_a_chat_title_is_the_start_of_its_oldest_memory():
    from harness.chat_session import TITLE_CHARS, _title
    assert _title({"text": "User: Hi, my name is Akshay.\n\nAssistant: Nice to meet you."}) == \
        "Hi, my name is Akshay. Assistant: Nice to meet you."
    long = _title({"text": "The user asked " + "about many things " * 20})
    assert len(long) <= TITLE_CHARS and long.endswith("…")


# ---------------------------------------------------------------------------
# Auto-save: the unsaved part of a chat is kept on disk after every turn
# ---------------------------------------------------------------------------

def _pending_file(session_id):
    import os
    return os.path.join(os.environ["AMP_CHATS_DIR"], f"{session_id}.pending")


def test_after_every_turn_the_unsaved_part_is_on_disk_encrypted(vault, client):
    import os
    import stat
    from harness import chat_store
    chat = session(session_id="chat-autosave", token_limit=4000)       # nothing is compacted while chatting
    chat.ask("My cat is called Pixel.")
    path = _pending_file("chat-autosave")
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    raw = open(path, encoding="ascii").read()
    assert raw.startswith("aesgcm:v1:") and "Pixel" not in raw         # encrypted with the chat's key
    saved = chat_store.load_pending("chat-autosave")
    assert saved["chunks"] == chat.chunks and saved["turn"] == 1

    chat.ask("And my codename is ZEBRAFISH.")
    assert chat_store.load_pending("chat-autosave")["chunks"] == chat.chunks     # rewritten each turn
    assert chat.save_all() is True and not os.path.exists(path)         # left normally: all in the vault


def test_a_chat_whose_app_stopped_continues_where_it_was(vault, client):
    first = session(session_id="chat-crash", token_limit=4000)
    first.ask("My cat is called Pixel.")
    first.ask("And my codename is ZEBRAFISH.")
    # the app is killed here: no save_all(), nothing in the vault
    assert vault.memories == []

    again = ChatSession(session_id="chat-crash", token_limit=4000)
    again.resume()
    assert again.restored == 2 and again.chunks == first.chunks and again.turn == 2
    assert "2 unsaved chunks back in the context window" in again.events[-1].title
    again.ask("What is my cat called?")
    assert any("Pixel" in m["content"] for m in client.calls[-1])      # the model sees the old turns again
    assert again.transcript[0]["id"] == "u5"                           # new IDs don't clash with old chunks


def test_background_compaction_keeps_the_file_current(vault, client):
    from harness import chat_store
    chat = session(session_id="chat-reclaim")
    for n in range(4):
        chat.ask(f"Question number {n}: tell me about topic {n}.")
        chat.wait_for_reclaim()
    assert chat.compactions >= 1
    assert chat_store.load_pending("chat-reclaim")["chunks"] == chat.chunks


def test_a_chat_with_only_an_unsaved_part_is_listed(vault, client):
    from harness.chat_session import list_chats
    chat = session(session_id="chat-unsaved", token_limit=4000)
    chat.ask("User: Hi, my dog is called Bruno.")
    listed = next(c for c in list_chats() if c["session_id"] == "chat-unsaved")
    assert listed["memories"] == 0 and listed["unsaved"] and "Hi, my dog is called Bruno" in listed["title"]
    assert listed["last_time"]


def test_an_unreadable_file_is_kept_and_reported(vault, client):
    import os
    from harness.local_keys import _write_keyring, keyring_path, known_sessions
    chat = session(session_id="chat-locked", token_limit=4000)
    chat.ask("My cat is called Pixel.")
    keys = known_sessions()
    keys["chat-locked"] = "0" * 64                     # a different key than the one the file was written with
    _write_keyring(keyring_path(), keys)

    again = ChatSession(session_id="chat-locked")
    again.resume()
    assert again.restored == 0 and again.chunks == []
    assert any(e.kind == "error" and "unsaved part" in e.title for e in again.events)
    assert again.flush(0) is True and os.path.exists(_pending_file("chat-locked"))   # never deleted


def test_bad_chat_ids_never_become_paths():
    from harness import chat_store
    for bad in ("../etc/passwd", "a/b", "", "x" * 65):
        with pytest.raises(ValueError):
            chat_store.load_pending(bad)
