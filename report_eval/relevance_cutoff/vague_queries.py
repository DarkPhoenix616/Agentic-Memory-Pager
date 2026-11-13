"""Distances of vague questions to run A's vault (no model): which memories would a cut-off keep?

Run from the repo root:  .venv/bin/python -I report_eval/relevance_cutoff/vague_queries.py <run A temp dir>
"""
import json
import os
import sys

W = sys.argv[1]
os.environ.update(AMP_CHROMA_PATH=f"{W}/vault", AMP_SESSION_KEYS_FILE=f"{W}/keys", AMP_CHATS_DIR=f"{W}/chats",
                  AMP_ADMIN_TOKEN="x", AMP_AUDIT_LOG=f"{W}/audit.log", AMP_RISK_MODEL="off", AMP_CONFLICT_MODEL="off")
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import server.amp_memory_server as server  # noqa: E402
from chat_recall_eval import FACTS, normalise  # noqa: E402
from harness.local_keys import known_sessions  # noqa: E402

sid = next(s for s in known_sessions() if s.startswith("cutoff-"))
key = known_sessions()[sid]
KEYS = [normalise(k) for _, _, k in FACTS]
VAGUE = ["What do you remember about me?", "What did we talk about earlier?",
         "Remind me what I told you about myself.", "What facts did I ask you to remember?",
         "Summarize what you know about my project.", "What did I say about the project's tools and models?",
         "Tell me everything you know about our setup."]
out = []
for q in VAGUE:
    found = server._search(q, 20, sid, key)
    rows = [(round(m.distance, 3), [k for k in KEYS if k in normalise(m.memory.text_payload.factual_summary
                                                                      if m.memory.text_payload else "")])
            for m in found]
    fact_d = [d for d, ks in rows if ks]
    print(f"{q!r}: fact memories at {sorted(fact_d)[:6]} ... | closest overall {rows[0][0]}")
    for T in (1.4, 1.5):
        print(f"    T={T}: {sum(d <= T for d, _ in rows)} memories kept, "
              f"{len({k for d, ks in rows if d <= T for k in ks})} distinct facts")
    out.append({"query": q, "rows": rows})
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vague_queries.json"), "w"), indent=1)
