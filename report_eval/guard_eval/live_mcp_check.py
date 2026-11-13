"""Live check: the memory server as its own process (temp vault, real risk
model), an outside MCP client storing attacks and real notes that all claim
risk_score=0. Prints what the server stored and what search gives back.

Run from the repo root:  .venv/bin/python -I <this file>
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_guard import ATTACKS, BENIGN  # noqa: E402
from harness.guardrails import TextMemorySchema, UnifiedMemory  # noqa: E402
from harness.mcp_client import MCPMemoryClient  # noqa: E402


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def note(session, text):
    return UnifiedMemory(session_id=session, memory_type="text", compaction_source="rlm", raw_token_count=30,
                         risk_score=0.0,
                         text_payload=TextMemorySchema(chunk_index=0, total_chunks=1, timeframe="1 message",
                                                       factual_summary=text, original_token_count=30))


def main():
    work = tempfile.mkdtemp(prefix="amp-live-")
    port = free_port()
    env = dict(os.environ, AMP_CHROMA_PATH=os.path.join(work, "vault"), AMP_MCP_PORT=str(port),
               AMP_SIGNING_KEY="live-check-key", AMP_ADMIN_TOKEN="live-check-admin",
               AMP_AUDIT_LOG=os.path.join(work, "audit.log"))
    env.pop("AMP_RISK_MODEL", None)                       # the default: the chat model
    server = subprocess.Popen([sys.executable, "server/amp_memory_server.py"], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for _ in range(300):
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.3)
        client = MCPMemoryClient(f"http://localhost:{port}/mcp", call_timeout=120)
        rows = []
        for kind, texts in (("attack", ATTACKS), ("normal", BENIGN)):
            for text in texts:
                t0 = time.perf_counter()
                stored = client.store_memory(note("live", text))
                rows.append({"kind": kind, "text": text, "id": stored.vector_embedding_id,
                             "score": stored.risk_score, "level": stored.risk_level,
                             "ms": round((time.perf_counter() - t0) * 1000)})
        visible = set()
        for text in ATTACKS + BENIGN:                    # search with each note's own words: best case for recall
            for hit in client.search_vault(text, top_k=5, session_id="live"):
                visible.add(hit.memory.vector_embedding_id)
        for r in rows:
            r["returned_by_search"] = r["id"] in visible
        refused = client.delete_memory("mem_whatever")
        with open(env["AMP_AUDIT_LOG"]) as handle:
            audit = [json.loads(line) for line in handle]
        client.close()
    finally:
        server.terminate()
        server.wait(timeout=10)
    for kind in ("attack", "normal"):
        mine = [r for r in rows if r["kind"] == kind]
        levels = {lv: sum(r["level"] == lv for r in mine) for lv in ("high_risk", "suspicious", "safe")}
        print(f"{kind}: {len(mine)} stored claiming risk 0 -> {levels}; returned by search: "
              f"{sum(r['returned_by_search'] for r in mine)}")
    ms = sorted(r["ms"] for r in rows)
    print(f"store_memory over MCP with the model: median {ms[len(ms)//2]} ms, max {ms[-1]} ms")
    for r in rows:
        flag = "SHOWN " if r["returned_by_search"] else "hidden"
        print(f"  {r['kind']:6} {flag} {r['score']:5.1f} {r['level']:10} {r['text'][:70]!r}")
    print("delete without token:", refused)
    print("audit:", audit)
    json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_mcp.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
