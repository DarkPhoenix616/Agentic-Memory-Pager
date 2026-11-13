"""Supplement to final_check.py: reopen chat A's first chat in the clone and list its vault."""
import json, os, signal, subprocess, sys, time
import httpx
clone, sid = sys.argv[1], sys.argv[2]
env = {k: v for k, v in os.environ.items() if not k.startswith("AMP_")}
proc = subprocess.Popen([os.path.join(clone, ".venv/bin/python"), "-m", "app.server", "--port", "8023"], cwd=clone,
                        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
url = "http://127.0.0.1:8023"
for _ in range(240):
    try:
        httpx.get(url + "/api/state", timeout=2); break
    except httpx.HTTPError:
        time.sleep(0.5)
st = httpx.post(url + "/api/open", json={"session_id": sid}, timeout=600).json()
vault = st["vault"]
out = {"session_id": sid, "memories": len(vault), "by_engine": {}, "labels": [m.get("label", "")[:80] for m in vault]}
for m in vault:
    out["by_engine"][m["engine"]] = out["by_engine"].get(m["engine"], 0) + 1
code = [m for m in vault if m["engine"] == "AST"]
out["ast_text"] = code[0]["text"][:400] if code else ""
proc.send_signal(signal.SIGINT); proc.communicate(timeout=60)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vault_check.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
