"""
Live check of auto-save (final round, task 2) with the real app and llama3.1:8b.

Runs `python -m app.server` as a real process on throwaway folders (vault,
keyring, chats, keys), never the user's storage/:

  A  chat 2 turns, then stop the app with Ctrl+C (SIGINT)
  B  start it again, reopen the chat from the list, ask a recall question
  C  one more turn, then kill the app hard (SIGKILL, like a crash)
     start again, reopen, ask again
  D  control: the same chat with its saved file removed (= before task 2)

Run:  .venv/bin/python report_eval/auto_save/live_auto_save.py
Writes results.json next to this file.
"""
import json
import os
import secrets
import signal
import subprocess
import sys
import tempfile
import time

import httpx

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the repo root
PY = os.path.join(REPO, ".venv/bin/python")
PORT = 8011
BASE = f"http://127.0.0.1:{PORT}"
HERE = os.path.dirname(os.path.abspath(__file__))
FACTS = ("pixel", "teal", "zebrafish")
RECALL = "What is my cat's name, my favourite colour and my project codename?"

tmp = tempfile.mkdtemp(prefix="amp-live-autosave-")
env = dict(os.environ, AMP_CHROMA_PATH=f"{tmp}/vault", AMP_SESSION_KEYS_FILE=f"{tmp}/keys.json",
           AMP_CHATS_DIR=f"{tmp}/chats", AMP_SIGNING_KEY=secrets.token_hex(32),
           AMP_ADMIN_TOKEN=secrets.token_hex(16), AMP_AUDIT_LOG=f"{tmp}/audit.log")
env.pop("AMP_MCP_URL", None)
results = {"tmp": tmp, "model": "llama3.1:8b", "phases": {}}


def start():
    proc = subprocess.Popen([PY, "-m", "app.server", "--port", str(PORT)], cwd=REPO, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for _ in range(240):
        try:
            if httpx.get(f"{BASE}/api/state", timeout=2).status_code == 200:
                return proc
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    proc.kill()
    raise SystemExit("server did not start:\n" + proc.communicate()[0])


def stop(proc, sig):
    started = time.monotonic()
    proc.send_signal(sig)
    out = proc.communicate(timeout=120)[0]
    return round(time.monotonic() - started, 2), out.strip().splitlines()[-3:]


def ask(message):
    reply, error, kinds = "", "", []
    with httpx.stream("POST", f"{BASE}/api/chat", json={"message": message}, timeout=600) as response:
        event = None
        for line in response.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
                if event == "done":
                    reply = data["reply"]
                elif event == "error":
                    error = data["message"]
                elif event == "event":
                    kinds.append(data["kind"])
    return {"message": message, "reply": reply, "error": error, "compactions": kinds.count("compaction")}


def state():
    return httpx.get(f"{BASE}/api/state", timeout=60).json()


def recalled(reply):
    text = reply.lower()
    return [fact for fact in FACTS if fact in text]


def pending_path(sid):
    return os.path.join(env["AMP_CHATS_DIR"], f"{sid}.pending")


def reopen(sid):
    listed = next((c for c in httpx.get(f"{BASE}/api/sessions", timeout=60).json()["chats"]
                   if c["session_id"] == sid), None)
    opened = httpx.post(f"{BASE}/api/open", json={"session_id": sid}, timeout=600)
    body = opened.json()
    return listed, opened.status_code, body


# A: chat, then Ctrl+C -------------------------------------------------------
proc = start()
turns = [ask("Hi! My cat is called Pixel and my favourite colour is teal."),
         ask("My project codename is ZEBRAFISH. Please remember it.")]
st = state()
sid = st["session_id"]
seconds, tail = stop(proc, signal.SIGINT)
raw = open(pending_path(sid), encoding="ascii").read() if os.path.exists(pending_path(sid)) else ""
results["phases"]["A_ctrl_c"] = {
    "turns": turns, "session_id": sid, "chunks": st["context"]["chunks"], "tokens": st["context"]["tokens"],
    "vault_memories": len(st["vault"]), "shutdown_seconds": seconds, "shutdown_output": tail,
    "file_exists": bool(raw), "file_mode": oct(os.stat(pending_path(sid)).st_mode & 0o777) if raw else "",
    "plaintext_in_file": any(f in raw.lower() for f in FACTS)}

# B: start again, reopen, recall --------------------------------------------
proc = start()
listed, code, body = reopen(sid)
answer = ask(RECALL)
st = state()
results["phases"]["B_reopen_after_ctrl_c"] = {
    "listed": listed, "open_status": code, "restored": body.get("restored"),
    "context_chunks_after_open": body.get("context", {}).get("chunks"), "recall": answer,
    "facts_recalled": recalled(answer["reply"])}

# C: one more turn, crash, start again, reopen, recall -----------------------
extra = ask("One more thing: my dog is called Bruno.")
before = state()["context"]["chunks"]
seconds, _ = stop(proc, signal.SIGKILL)
proc = start()
listed, code, body = reopen(sid)
answer = ask(RECALL + " And my dog's name?")
results["phases"]["C_reopen_after_kill_9"] = {
    "extra_turn": extra, "chunks_before_kill": before, "listed": listed, "open_status": code,
    "restored": body.get("restored"), "recall": answer,
    "facts_recalled": recalled(answer["reply"]) + (["bruno"] if "bruno" in answer["reply"].lower() else [])}
stop(proc, signal.SIGINT)

# D: control, the saved file removed (what happened before task 2) ------------
os.remove(pending_path(sid))
proc = start()
listed, code, body = reopen(sid)
answer = ask(RECALL)
results["phases"]["D_control_no_file"] = {
    "listed": listed, "open_status": code, "restored": body.get("restored"), "recall": answer,
    "facts_recalled": recalled(answer["reply"])}
stop(proc, signal.SIGINT)

# How long one save takes (in-process, same folders) --------------------------
sys.path.insert(0, REPO)
os.environ.update({k: env[k] for k in ("AMP_SESSION_KEYS_FILE", "AMP_CHATS_DIR")})
from harness import chat_store  # noqa: E402
chunks = [{"role": "user", "type": "text", "content": "User: hello " * 200 + "\n\nAssistant: hi " * 200,
           "msg_ids": [f"u{n}", f"a{n}"]} for n in range(6)]
started = time.perf_counter()
for _ in range(100):
    chat_store.save_pending("chat-timing", {"turn": 6, "ids": 12, "chunks": chunks})
results["save_ms"] = round((time.perf_counter() - started) * 1000 / 100, 2)
results["save_bytes"] = os.path.getsize(chat_store._path("chat-timing"))

with open(os.path.join(HERE, "results.json"), "w") as handle:
    json.dump(results, handle, indent=2)
print(json.dumps({phase: {k: v for k, v in data.items() if k not in ("turns", "recall", "extra_turn")}
                  for phase, data in results["phases"].items()}, indent=1))
for phase in ("B_reopen_after_ctrl_c", "C_reopen_after_kill_9", "D_control_no_file"):
    print(f"\n{phase}: {results['phases'][phase]['recall']['reply'][:400]}")
print("\nsave:", results["save_ms"], "ms,", results["save_bytes"], "bytes")
