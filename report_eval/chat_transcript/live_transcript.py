"""
Live check of saved messages (final round, task 6) with the real app and llama3.1:8b.

The app runs as its own process on throwaway folders. A chat of 3 turns with a
small window (so older messages are paged out), then New chat, then the old
chat is reopened: its messages must come back for display, greyed where they
were paged out, and recall must still work through the vault. A headless
Chrome screenshot shows the reopened chat as the user sees it.

Run:  .venv/bin/python report_eval/chat_transcript/live_transcript.py
Writes results.json and reopened_chat.png next to this file.
"""
import json
import os
import secrets
import shutil
import signal
import subprocess
import tempfile
import time

import httpx

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the repo root
PY = os.path.join(REPO, ".venv/bin/python")
PORT = 8024
URL = f"http://127.0.0.1:{PORT}"
HERE = os.path.dirname(os.path.abspath(__file__))
tmp = tempfile.mkdtemp(prefix="amp-live-transcript-")
env = dict({k: v for k, v in os.environ.items() if not k.startswith("AMP_")},
           AMP_CHROMA_PATH=f"{tmp}/vault", AMP_SESSION_KEYS_FILE=f"{tmp}/keys.json", AMP_CHATS_DIR=f"{tmp}/chats",
           AMP_SIGNING_KEY=secrets.token_hex(32), AMP_ADMIN_TOKEN=secrets.token_hex(16), AMP_AUDIT_LOG=f"{tmp}/audit.log")


def ask(message):
    reply = ""
    with httpx.stream("POST", f"{URL}/api/chat", json={"message": message}, timeout=900) as response:
        event = None
        for line in response.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: ") and event == "done":
                reply = json.loads(line[6:])["reply"]
    return reply


def wait_reclaim():
    for _ in range(600):
        st = httpx.get(f"{URL}/api/state", timeout=120).json()
        if not st["reclaiming"]:
            return st
        time.sleep(0.5)


proc = subprocess.Popen([PY, "-m", "app.server", "--port", str(PORT)], cwd=REPO, env=env,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for _ in range(240):
    try:
        httpx.get(f"{URL}/api/state", timeout=2)
        break
    except httpx.HTTPError:
        time.sleep(0.5)

httpx.post(f"{URL}/api/settings", json={"token_limit": 1024}, timeout=60)
turns = ["Hi! My cat is called Pixel and my project codename is ZEBRAFISH.",
         "Here is our helper:\n```python\ndef cart_total(items, tax_rate=0.18):\n"
         "    return round(sum(i['price'] * i['qty'] for i in items) * (1 + tax_rate), 2)\n```\nWhat does it do?",
         "Explain in a few short paragraphs how a context window works in a language model."]
for message in turns:
    ask(message)
before = wait_reclaim()
old_id = before["session_id"]
httpx.post(f"{URL}/api/new", json={}, timeout=900)
chats = httpx.get(f"{URL}/api/sessions", timeout=120).json()["chats"]
opened = httpx.post(f"{URL}/api/open", json={"session_id": old_id}, timeout=900).json()
recall = ask("What is my cat called, and what is my project codename?")
after = httpx.get(f"{URL}/api/state", timeout=120).json()

shot = os.path.join(HERE, "reopened_chat.png")
profile = tempfile.mkdtemp(prefix="amp-chrome-")
subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", f"--user-data-dir={profile}",
                "--window-size=1400,1700", "--virtual-time-budget=10000", f"--screenshot={shot}", URL + "/"],
               capture_output=True, timeout=120)
shutil.rmtree(profile, ignore_errors=True)

proc.send_signal(signal.SIGINT)
proc.communicate(timeout=120)
chats_dir = f"{tmp}/chats"
stored = b"".join(open(os.path.join(chats_dir, n), "rb").read() for n in os.listdir(chats_dir))

results = {
    "before_leaving": {"compactions": before["stats"]["compactions"],
                       "statuses": [m["status"] for m in before["transcript"]]},
    "listed": next((c for c in chats if c["session_id"] == old_id), None),
    "after_reopen": {"restored_messages": opened["restored_messages"],
                     "statuses": [m["status"] for m in opened["transcript"]],
                     "first_message": opened["transcript"][0]["content"] if opened["transcript"] else "",
                     "resumed_notes": len(opened["resumed"])},
    "recall_reply": recall,
    "recall": {w: w in recall.lower() for w in ("pixel", "zebrafish")},
    "transcript_after_recall": len(after["transcript"]),
    "ids_after_recall": [m["id"] for m in after["transcript"]],
    "plaintext_in_chat_files": {w: stored.count(w.encode()) for w in ("ZEBRAFISH", "Pixel", "cart_total")},
    "chat_files": sorted(os.listdir(chats_dir)),
    "screenshot": shot if os.path.exists(shot) else "",
}
json.dump(results, open(os.path.join(HERE, "results.json"), "w"), indent=1)
print(json.dumps(results, indent=1))
