"""
Does the page switch chats? Drives a real headless Chrome (DevTools protocol)
through AMP Chat on a throwaway instance (port 8030, temporary folders, the real
llama3.1:8b): send a message, press New chat, reopen the old chat, and count
the messages the page shows after each step.

Run from the repo root:  .venv/bin/python -I report_eval/ui_check/new_chat_ui.py <label>
"""
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import httpx
from websockets.sync.client import connect

REPO = os.getcwd()
LABEL = sys.argv[1] if len(sys.argv) > 1 else "run"
tmp = tempfile.mkdtemp(prefix="amp-ui-")
env = dict({k: v for k, v in os.environ.items() if not k.startswith("AMP_")},
           AMP_CHROMA_PATH=f"{tmp}/vault", AMP_SESSION_KEYS_FILE=f"{tmp}/keys", AMP_CHATS_DIR=f"{tmp}/chats",
           AMP_SIGNING_KEY=secrets.token_hex(32), AMP_ADMIN_TOKEN=secrets.token_hex(16), AMP_AUDIT_LOG=f"{tmp}/audit.log")
app = subprocess.Popen([os.path.join(REPO, ".venv/bin/python"), "-m", "app.server", "--port", "8030"], cwd=REPO,
                       env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
profile = tempfile.mkdtemp(prefix="amp-chrome-")
chrome = subprocess.Popen(["google-chrome", "--headless=new", "--disable-gpu", f"--user-data-dir={profile}",
                           "--remote-debugging-port=9333", "about:blank"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_for(url):
    for _ in range(120):
        try:
            return httpx.get(url, timeout=2)
        except httpx.HTTPError:
            time.sleep(0.5)
    raise SystemExit(f"{url} never answered")


results = {"label": LABEL}
try:
    wait_for("http://127.0.0.1:8030/api/state")
    page = next(t for t in wait_for("http://127.0.0.1:9333/json/list").json() if t["type"] == "page")
    with connect(page["webSocketDebuggerUrl"], max_size=None) as ws:
        ids = iter(range(1, 10_000))

        def call(method, **params):
            n = next(ids)
            ws.send(json.dumps({"id": n, "method": method, "params": params}))
            while True:
                reply = json.loads(ws.recv(timeout=900))
                if reply.get("id") == n:
                    return reply

        def js(expression):
            reply = call("Runtime.evaluate", expression=expression, awaitPromise=True, returnByValue=True)
            if "exceptionDetails" in reply.get("result", {}):
                raise RuntimeError(reply["result"]["exceptionDetails"])
            return reply["result"]["result"].get("value")

        shown = "document.querySelectorAll('#messages .msg').length"
        greeting = "document.getElementById('empty').hidden ? '' : document.getElementById('empty').innerText"
        call("Page.navigate", url="http://127.0.0.1:8030/")
        for _ in range(120):
            if js("typeof state !== 'undefined' && state !== null && ready === true"):
                break
            time.sleep(0.5)
        results["fresh_page"] = {"shown": js(shown), "greeting": js(greeting)}
        js('send("Hi! My cat is called Pixel.")')
        results["after_send"] = {"shown": js(shown), "session": js("state.session_id")}
        old = results["after_send"]["session"]
        js('leave("/api/new", {}, "")')
        results["after_new_chat"] = {"shown": js(shown), "session": js("state.session_id"),
                                     "empty_hint_visible": js("!document.getElementById('empty').hidden"),
                                     "greeting": js(greeting)}
        js(f'openChat("{old}")')
        results["after_reopen"] = {"shown": js(shown), "session": js("state.session_id"), "greeting": js(greeting)}
        # an older chat, from before chats saved their messages: its transcript file does not exist
        js('leave("/api/new", {}, "")')
        os.remove(os.path.join(tmp, "chats", f"{old}.transcript"))
        js(f'openChat("{old}")')
        results["old_chat_without_saved_messages"] = {
            "summaries_shown": js("document.querySelectorAll('#messages .msg.summary').length"),
            "shown": js(shown), "greeting": js(greeting),
            "info_box_present": js("document.getElementById('resume') !== null")}
        results["cache_control"] = httpx.get("http://127.0.0.1:8030/static/app.js").headers.get("cache-control")
finally:
    chrome.send_signal(signal.SIGTERM)
    app.send_signal(signal.SIGINT)
    chrome.wait(timeout=30)
    app.wait(timeout=60)
    shutil.rmtree(profile, ignore_errors=True)

ok = (results["after_send"]["shown"] == 2 and results["after_new_chat"]["shown"] == 0
      and results["after_new_chat"]["session"] != results["after_send"]["session"]
      and results["after_reopen"]["shown"] == 2)
if "fresh_page" in results:
    ok = ok and results["fresh_page"]["greeting"].startswith("Hello") and \
        results["after_new_chat"]["greeting"].startswith("Hello") and not results["after_reopen"]["greeting"] and \
        results["old_chat_without_saved_messages"]["summaries_shown"] >= 1 and \
        not results["old_chat_without_saved_messages"]["greeting"] and results["cache_control"] == "no-cache" and \
        not results["old_chat_without_saved_messages"].get("info_box_present", False)
results["page_switches_chats"] = ok
print(json.dumps(results, indent=1))
json.dump(results, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{LABEL}.json"), "w"), indent=1)
