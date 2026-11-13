"""
Final check (final round, task 5): a fresh clone of akshaybranch with a fresh
venv from requirements.txt, run the way a new user would, with its own empty
storage/. Drives the real web API with the real llama3.1:8b:

  A  in-process:  python -m app.server
     facts + a code block with a small window (compaction: RLM and AST),
     recall, a vault search and run_memory_code, new chat, reopen, recall,
     encryption on disk, Ctrl+C
  B  over MCP:    python server/amp_memory_server.py  +  python -m app.server --mcp-url ...
     two turns, recall after compaction, Ctrl+C on both

Run:  <clone>/.venv/bin/python final_check.py <clone dir>
Writes results.json next to this file.
"""
import json
import os
import signal
import stat
import subprocess
import sys
import time

import httpx

CLONE = os.path.abspath(sys.argv[1])
PY = os.path.join(CLONE, ".venv/bin/python")
HERE = os.path.dirname(os.path.abspath(__file__))
ENV = {k: v for k, v in os.environ.items() if not k.startswith("AMP_")}
CODE = '''```python
def cart_total(items, tax_rate=0.18):
    """Sum of price * quantity for every item, plus tax."""
    subtotal = sum(item["price"] * item["qty"] for item in items)
    return round(subtotal * (1 + tax_rate), 2)
```'''
results = {"clone": CLONE, "commit": subprocess.run(["git", "-C", CLONE, "log", "--oneline", "-1"],
                                                     capture_output=True, text=True).stdout.strip()}


def start(args, port, env=None):
    proc = subprocess.Popen([PY, *args], cwd=CLONE, env=env or ENV, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    url = f"http://127.0.0.1:{port}"
    for _ in range(240):
        try:
            httpx.get(url + ("/api/state" if "app.server" in args else "/mcp"), timeout=2)
            return proc, url
        except httpx.HTTPError:
            time.sleep(0.5)
    proc.kill()
    raise SystemExit(f"{args} did not start:\n" + proc.communicate()[0])


def stop(proc):
    started = time.monotonic()
    proc.send_signal(signal.SIGINT)
    out = proc.communicate(timeout=120)[0]
    return round(time.monotonic() - started, 2), [line for line in out.splitlines() if line.strip()][-3:]


def ask(url, message):
    reply, error, kinds, started = "", "", [], time.monotonic()
    with httpx.stream("POST", f"{url}/api/chat", json={"message": message}, timeout=900) as response:
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
                    kinds.append((data["kind"], data["title"][:110]))
    return {"message": message[:80], "reply": reply, "error": error, "seconds": round(time.monotonic() - started, 1),
            "steps": kinds}


def state(url):
    return httpx.get(f"{url}/api/state", timeout=120).json()


def post(url, path, body):
    return httpx.post(f"{url}{path}", json=body, timeout=900)


def has(text, *words):
    return {w: w.lower() in text.lower() for w in words}


def mode(path):
    return oct(stat.S_IMODE(os.stat(path).st_mode)) if os.path.exists(path) else "missing"


def chat_flow(url, label):
    """The same conversation for both modes; returns its record."""
    out = {}
    post(url, "/api/settings", {"token_limit": 1024, "threshold": 0.8})
    turns = [ask(url, "Hi! My name is Akshay. My cat is called Pixel and my project codename is ZEBRAFISH. "
                      "Please remember these."),
             ask(url, f"Here is the function we use for the shop cart:\n{CODE}\nCan you explain what it does?"),
             ask(url, "Explain in a few short paragraphs how a context window works in a language model, and "
                      "why long chats run out of space."),
             ask(url, "What is my cat's name, my project codename, and what does my cart_total function do?")]
    st = state(url)
    out["turns"] = turns
    out["compactions"] = st["stats"]["compactions"]
    out["engines"] = sorted({m["engine"] for m in st["vault"]})
    out["vault_memories"] = len(st["vault"])
    out["recall"] = has(turns[-1]["reply"], "pixel", "zebrafish", "tax")
    out["session_id"] = st["session_id"]
    return out


report = {}

# A: in-process ------------------------------------------------------------------
proc, url = start(["-m", "app.server", "--port", "8021"], 8021)
status = httpx.get(f"{url}/api/status", timeout=60).json()
page = httpx.get(url + "/", timeout=30)
js = httpx.get(url + "/static/app.js", timeout=30)
a = {"status": status, "page_ok": page.status_code == 200 and "<html" in page.text.lower(),
     "app_js_ok": js.status_code == 200}
a.update(chat_flow(url, "A"))
a["search"] = ask(url, "Use search_vault to look up what my project codename is, then tell me.")
a["code"] = ask(url, "Use run_memory_code to count how many memories mention Pixel, then tell me the number.")
first = a["session_id"]
new = post(url, "/api/new", {})
a["new_chat_status"] = new.status_code
chats = httpx.get(f"{url}/api/sessions", timeout=60).json()["chats"]
a["listed"] = next((c for c in chats if c["session_id"] == first), None)
opened = post(url, "/api/open", {"session_id": first})
a["reopen_status"] = opened.status_code
a["resumed_notes"] = len(opened.json().get("resumed", []))
a["after_reopen"] = ask(url, "What is my cat called and what is my project codename?")
a["recall_after_reopen"] = has(a["after_reopen"]["reply"], "pixel", "zebrafish")
a["shutdown_seconds"], a["shutdown_output"] = stop(proc)
storage = os.path.join(CLONE, "storage")
vault_bytes = b""
for root, _, files in os.walk(os.path.join(storage, "chroma_db")):
    for name in files:
        with open(os.path.join(root, name), "rb") as handle:
            vault_bytes += handle.read()
a["plaintext_in_vault"] = {w: vault_bytes.count(w.encode()) for w in ("ZEBRAFISH", "Pixel", "cart_total")}
a["files"] = {name: mode(os.path.join(storage, name)) for name in
              (".amp_signing_key", ".amp_session_keys", ".amp_admin_token", "amp_audit.log")}
a["chats_dir"] = sorted(os.listdir(os.path.join(storage, "chats"))) if os.path.isdir(os.path.join(storage, "chats")) else []
report["A_in_process"] = a

# B: over MCP ---------------------------------------------------------------------
env = dict(ENV, AMP_MCP_PORT="8031")
server, _ = start(["server/amp_memory_server.py"], 8031, env)
proc, url = start(["-m", "app.server", "--port", "8022", "--mcp-url", "http://localhost:8031/mcp"], 8022)
b = {"status": httpx.get(f"{url}/api/status", timeout=60).json()}
b.update(chat_flow(url, "B"))
b["shutdown_seconds"], b["shutdown_output"] = stop(proc)
b["server_shutdown_seconds"], _ = stop(server)
report["B_over_mcp"] = b

results["report"] = report
with open(os.path.join(HERE, "results.json"), "w") as handle:
    json.dump(results, handle, indent=1)

for key, data in report.items():
    print(f"\n== {key}")
    for field in ("compactions", "engines", "vault_memories", "recall", "recall_after_reopen", "new_chat_status",
                  "reopen_status", "resumed_notes", "plaintext_in_vault", "files", "chats_dir", "shutdown_seconds",
                  "shutdown_output", "page_ok", "app_js_ok"):
        if field in data:
            print(f"{field}: {data[field]}")
    print("status:", data["status"].get("message"), "|", data["status"].get("vault", {}).get("message"))
    for turn in data["turns"] + [data[k] for k in ("search", "code", "after_reopen") if k in data]:
        print(f"  [{turn['seconds']:5.1f}s] {turn['message']!r} -> {turn['reply'][:160]!r} {turn['error']}")
        print("        steps:", [k for k, _ in turn["steps"]])
