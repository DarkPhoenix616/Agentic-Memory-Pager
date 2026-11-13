"""Live check of saving, listing and reopening chats, through the real AMP Chat
web server and llama3.1:8b, on a temporary vault and keyring.

Run from the repo root:  .venv/bin/python -I report_eval/live_chat_history.py
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

PORT = 8011
BASE = f"http://127.0.0.1:{PORT}"
HEAD = {"Content-Type": "application/json", "Origin": BASE}


def call(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, headers=HEAD if body is not None else {})
    with urllib.request.urlopen(req, timeout=600) as res:
        text = res.read().decode()
    if path == "/api/chat":
        frames = [b for b in text.strip().split("\n\n") if b]
        last = json.loads(frames[-1].split("data: ", 1)[1])
        return last.get("reply") or last
    return json.loads(text)


def main():
    work = tempfile.mkdtemp(prefix="amp-live-history-")
    env = dict(os.environ, AMP_CHROMA_PATH=f"{work}/vault", AMP_SESSION_KEYS_FILE=f"{work}/keyring", AMP_CHATS_DIR=f"{work}/chats",
               AMP_SIGNING_KEY="live-sign", AMP_ADMIN_TOKEN="live-admin", AMP_AUDIT_LOG=f"{work}/audit.log")
    env.pop("AMP_MCP_URL", None)
    app = subprocess.Popen([sys.executable, "-m", "app.server", "--port", str(PORT)], env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            try:
                call("/api/status")
                break
            except OSError:
                time.sleep(0.5)
        first = call("/api/state")["session_id"]
        call("/api/chat", {"message": "My cat is called Pixel and my codename is ZEBRAFISH. "
                                      "Explain in about 150 words what a vector database is."})
        call("/api/chat", {"message": "Thanks. In two lines, what is an embedding?"})
        t0 = time.monotonic()
        new = call("/api/new", {})
        print(f"left chat {first}: saved in {time.monotonic() - t0:.1f}s; new chat {new['session_id']}")
        chats = call("/api/sessions")["chats"]
        print("listed:", [(c["session_id"], c["memories"], c["title"][:50]) for c in chats])
        opened = call("/api/open", {"session_id": first})
        print("reopened:", opened["session_id"], "| resumed notes:", len(opened["resumed"]),
              "| transcript:", len(opened["transcript"]))
        reply = call("/api/chat", {"message": "What is my cat called and what is my codename?"})
        print("answer:", reply[:200])
        print("recalled Pixel:", "pixel" in reply.lower(), "| recalled ZEBRAFISH:", "zebrafish" in reply.lower())
    finally:
        app.terminate()
        app.wait(timeout=20)
    found = subprocess.run(["grep", "-r", "-a", "-c", "ZEBRAFISH", f"{work}/vault"], capture_output=True, text=True).stdout
    print("ZEBRAFISH in vault files:", sum(int(line.rsplit(":", 1)[1]) for line in found.split()))
    subprocess.run(["rm", "-rf", work])


if __name__ == "__main__":
    main()
