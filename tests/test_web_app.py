"""
tests/test_web_app.py — the AMP Chat web server (app/server.py) and its static UI.

Runs the real Starlette app and the real AMP graph with the fake vault, fake
Ollama client and fake RLM model from test_chat_session.py.

Run with:
    pytest tests/test_web_app.py -v
"""
import json
import shutil
import subprocess

import pytest

pytest.importorskip("langgraph", reason="harness.graph needs langgraph")
pytest.importorskip("httpx", reason="starlette's TestClient needs httpx")

from starlette.testclient import TestClient  # noqa: E402

from app.server import create_app  # noqa: E402
from tests.test_chat_session import LONG_REPLY, FakeClient, client, vault  # noqa: E402,F401  (fixtures)


@pytest.fixture
def web(vault, client):
    app = create_app(shutdown_wait=2)
    with TestClient(app) as test_client:
        test_client.amp = app.state.chat
        yield test_client


def stream_frames(response):
    """Parse a text/event-stream body into (event, data) pairs."""
    frames = []
    for block in response.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        frames.append((lines["event"], json.loads(lines["data"])))
    return frames


def chat(web, message, **headers):
    return web.post("/api/chat", json={"message": message}, headers=headers)


# ---------------------------------------------------------------------------
# Page and status
# ---------------------------------------------------------------------------

def test_the_page_and_its_files_are_served(web):
    page = web.get("/")
    assert page.status_code == 200 and "<title>AMP Chat</title>" in page.text
    assert web.get("/static/app.js").status_code == 200
    assert web.get("/static/style.css").headers["content-type"].startswith("text/css")


def test_status_reports_the_local_model_and_the_vault(web, monkeypatch):
    monkeypatch.delenv("AMP_MCP_URL", raising=False)
    assert web.get("/api/status").json() == {
        "ready": True, "message": "llama3.1:8b (local, Ollama)", "model": "llama3.1:8b",
        "vault": {"mode": "in-process", "url": None, "ok": True, "message": "Memory in-process (no MCP server)"}}


def test_status_reports_an_unreachable_mcp_server(web, monkeypatch):
    from harness import mcp_client
    monkeypatch.setenv("AMP_MCP_URL", "http://localhost:9/mcp")            # nothing listens on port 9
    monkeypatch.setattr(mcp_client, "_clients", {"http://localhost:9/mcp": mcp_client.MCPMemoryClient(
        "http://localhost:9/mcp", connect_timeout=5)})
    vault = web.get("/api/status").json()["vault"]
    assert vault["mode"] == "mcp" and vault["ok"] is False
    assert "Cannot reach the MCP memory server" in vault["message"]


# ---------------------------------------------------------------------------
# Chatting
# ---------------------------------------------------------------------------

def test_a_message_streams_its_steps_then_the_reply(web):
    response = chat(web, "Hello there")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = stream_frames(response)
    kinds = [kind for kind, _ in frames]
    assert kinds[-1] == "done" and frames[-1][1] == {"reply": LONG_REPLY}
    steps = [data["kind"] for kind, data in frames if kind == "event"]
    assert steps[0] == "monitor" and "reply" in steps

    state = web.get("/api/state").json()
    assert [m["role"] for m in state["transcript"]] == ["user", "assistant"]
    assert state["transcript"][0]["status"] == "in context"
    assert state["context"]["tokens"] > 0 and state["turn"] == 1


def test_long_chats_page_out_and_the_state_shows_it(web):
    web.post("/api/settings", json={"token_limit": 1024, "background": False})
    for n in range(5):
        chat(web, f"Question {n}")
    state = web.get("/api/state").json()
    assert state["stats"]["compactions"] > 0
    assert state["vault"] and state["vault"][0]["engine"] == "RLM"
    assert state["transcript"][0]["status"] == "paged out"
    compactions = [e for e in state["events"] if e["kind"] == "compaction"]
    assert compactions and "memory" in compactions[0]["data"]
    assert all("evicted" not in e["data"] for e in state["events"])     # raw chunk text stays on the server


def test_model_errors_arrive_as_an_error_event(web, client):
    client.fail = True
    frames = stream_frames(chat(web, "Hello"))
    assert frames[-1][0] == "error" and "ollama serve" in frames[-1][1]["message"]
    assert web.get("/api/state").json()["transcript"] == []


def test_a_message_too_long_for_the_window_is_refused(web):
    frames = stream_frames(chat(web, "word " * 3000))
    assert frames[-1][0] == "error" and "tokens" in frames[-1][1]["message"]


def test_one_turn_at_a_time(web):
    web.amp.turn_lock.acquire()
    try:
        response = chat(web, "Hello")
        assert response.status_code == 409
        assert web.post("/api/new", json={}).status_code == 409
    finally:
        web.amp.turn_lock.release()
    assert chat(web, "Hello").status_code == 200


def test_empty_messages_are_refused(web):
    assert chat(web, "   ").status_code == 400
    assert web.post("/api/chat", json={"text": "wrong key"}).status_code == 400


# ---------------------------------------------------------------------------
# Settings and new chat
# ---------------------------------------------------------------------------

def test_settings_are_applied_and_kept_in_range(web):
    state = web.post("/api/settings", json={"token_limit": 99_999, "threshold": 0.2, "background": False}).json()
    assert state["settings"]["token_limit"] == state["settings"]["max_token_limit"]
    assert state["settings"]["threshold"] == 0.5 and state["settings"]["background"] is False
    assert web.post("/api/settings", json={"token_limit": "lots"}).status_code == 400


def test_new_chat_starts_empty_with_the_same_settings(web):
    web.post("/api/settings", json={"token_limit": 2048})
    chat(web, "Hello")
    old_id = web.get("/api/state").json()["session_id"]
    state = web.post("/api/new", json={}).json()
    assert state["session_id"] != old_id and state["transcript"] == []
    assert state["settings"]["token_limit"] == 2048


# ---------------------------------------------------------------------------
# Old chats: saved when left, listed, reopened
# ---------------------------------------------------------------------------

def test_a_chat_is_saved_when_left_and_can_be_reopened(web):
    from harness.local_keys import session_key
    chat(web, "My cat is called Pixel.")
    old_id = web.get("/api/state").json()["session_id"]
    session_key(old_id)                                   # (the fake vault does not make keys itself)
    state = web.post("/api/new", json={}).json()
    assert state["session_id"] != old_id

    chats = web.get("/api/sessions").json()["chats"]
    saved = next(c for c in chats if c["session_id"] == old_id)
    assert saved["memories"] == 1 and not saved["current"] and "Pixel" in saved["title"]

    state = web.post("/api/open", json={"session_id": old_id}).json()
    assert state["session_id"] == old_id and state["restored_messages"] == 2
    assert [m["content"] for m in state["transcript"]][0] == "My cat is called Pixel."
    assert all(m["status"] == "paged out" for m in state["transcript"])
    assert state["resumed"] and "Pixel" in state["resumed"][0]["text"]
    assert next(c for c in web.get("/api/sessions").json()["chats"] if c["session_id"] == old_id)["current"]


def test_only_known_chats_can_be_opened(web):
    assert web.post("/api/open", json={"session_id": "chat-nobody"}).status_code == 404
    assert web.post("/api/open", json={"session_id": "../etc/passwd"}).status_code == 400
    assert web.post("/api/open", json={"session_id": "x"}, headers={"origin": "https://evil.example"}).status_code == 403


def test_no_switching_during_an_answer(web):
    assert web.amp.turn_lock.acquire()
    try:
        assert web.post("/api/open", json={"session_id": "chat-a"}).status_code == 409
        assert web.post("/api/new", json={}).status_code == 409
    finally:
        web.amp.turn_lock.release()


def test_a_chat_that_cannot_be_saved_stays_open(web, monkeypatch):
    from harness import graph as amp
    chat(web, "My cat is called Pixel.")
    before = web.get("/api/state").json()

    def broken(state):
        raise ConnectionError("Ollama is not running")
    monkeypatch.setattr(amp, "rlm_summarizer_node", broken)
    response = web.post("/api/new", json={})
    assert response.status_code == 503 and "stays open" in response.json()["error"]
    after = web.get("/api/state").json()
    assert after["session_id"] == before["session_id"] and after["context"]["chunks"] == before["context"]["chunks"]


def _pending_file(session_id):
    import os
    return os.path.join(os.environ["AMP_CHATS_DIR"], f"{session_id}.pending")


def test_stopping_the_app_waits_for_the_answer_then_saves(vault, client):
    import os
    import threading
    app = create_app(shutdown_wait=5)
    with TestClient(app) as web:
        chat(web, "My cat is called Pixel.")
        path = _pending_file(web.get("/api/state").json()["session_id"])
        os.remove(path)                                   # as if the last per-turn save had not happened
        assert app.state.chat.turn_lock.acquire()         # an answer is still running...
        threading.Timer(0.3, app.state.chat.turn_lock.release).start()     # ...and ends soon
    assert os.path.exists(path)                           # the shutdown waited for it, then saved


def test_a_chat_from_a_stopped_app_can_be_reopened_with_its_unsaved_part(vault, client):
    with TestClient(create_app(shutdown_wait=1)) as web:
        chat(web, "My cat is called Pixel.")
        chat(web, "And my codename is ZEBRAFISH.")
        before = web.get("/api/state").json()
    assert vault.memories == []                           # never left: nothing went to the vault

    with TestClient(create_app(shutdown_wait=1)) as web:  # the app is started again
        listed = next(c for c in web.get("/api/sessions").json()["chats"]
                      if c["session_id"] == before["session_id"])
        assert listed["unsaved"] and listed["memories"] == 0 and "Pixel" in listed["title"]
        state = web.post("/api/open", json={"session_id": before["session_id"]}).json()
        assert state["restored"] == before["context"]["chunks"] == state["context"]["chunks"] == 2
        assert state["context"]["tokens"] == before["context"]["tokens"]


# ---------------------------------------------------------------------------
# Local-only protections
# ---------------------------------------------------------------------------

def test_other_websites_cannot_drive_the_api(web):
    assert chat(web, "Hello", origin="https://evil.example").status_code == 403
    assert chat(web, "Hello", origin="http://localhost:8000").status_code == 200


def test_posts_must_be_json(web):
    response = web.post("/api/chat", content="message=hi", headers={"content-type": "text/plain"})
    assert response.status_code == 415


def test_unknown_host_headers_are_rejected(web):
    assert web.get("/api/state", headers={"host": "attacker.example"}).status_code == 400
    assert web.get("/api/state", headers={"host": "127.0.0.1:8000"}).status_code == 200


def test_the_server_refuses_to_listen_on_other_addresses():
    from app.server import main
    with pytest.raises(SystemExit):
        main(["--host", "0.0.0.0"])


NODE_CHECK = r"""
const fs = require("fs");
const src = fs.readFileSync("app/static/app.js", "utf8");
const start = src.indexOf("function escapeHtml");
const end = src.indexOf("// API");
eval(src.slice(start, src.lastIndexOf("\n", end)));
const cases = JSON.parse(process.argv[1]);
console.log(JSON.stringify(cases.map(markdown)));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js to run the page script")
def test_the_page_renders_model_output_safely():
    cases = [
        "<script>alert(1)</script>",
        "<img src=x onerror=alert(1)>",
        "**bold**, *italic* and `code <b>`",
        "```python\nx = '<b>'\n```",
        "- one\n- two\n\n1. first",
    ]
    out = json.loads(subprocess.run(["node", "-e", NODE_CHECK, json.dumps(cases)], capture_output=True,
                                    text=True, check=True, timeout=30).stdout)
    assert out[0] == "<p>&lt;script&gt;alert(1)&lt;/script&gt;</p>"
    assert "<img" not in out[1] and "&lt;img" in out[1]
    assert out[2] == "<p><strong>bold</strong>, <em>italic</em> and <code>code &lt;b&gt;</code></p>"
    assert out[3] == "<pre><code>x = &#39;&lt;b&gt;&#39;</code></pre>"
    assert out[4] == "<ul>\n<li>one</li>\n<li>two</li>\n</ul>\n<ol>\n<li>first</li>\n</ol>"
