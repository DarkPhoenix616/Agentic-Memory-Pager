"""
Relevance cut-off for paging (final round, task 7): the 16-fact recall chat of
chat_recall_eval.py (report 5.3b), AMP only, recording on EVERY turn which
memories were paged in, at what distance and how many tokens.

  probes   16 recall questions: the memory holding the asked fact is relevant,
           the rest is noise
  fillers  16 general questions ("Explain the CAP theorem"): every paged
           memory is noise
  facts    16 statements of new facts

Run A (no cut-off) gives the distance of relevant memories vs noise; run B
repeats the chat with AMP_PAGE_MAX_DISTANCE set, to check that recall stays
16/16 and to measure the tokens saved.

Temporary vault, keyring, chats and audit log: never the user's storage/.

Run from the repo root:
  .venv/bin/python -I report_eval/relevance_cutoff/recall_cutoff_eval.py <label> [token_limit]
Writes <label>.json next to this file.
"""
import json
import os
import secrets
import sys
import tempfile
import time

W = tempfile.mkdtemp(prefix="amp-cutoff-")
os.environ.update(AMP_CHROMA_PATH=f"{W}/vault", AMP_SESSION_KEYS_FILE=f"{W}/keys", AMP_CHATS_DIR=f"{W}/chats",
                  AMP_AUDIT_LOG=f"{W}/audit.log", AMP_ADMIN_TOKEN=secrets.token_hex(16),
                  AMP_SIGNING_KEY=secrets.token_hex(32))
os.environ.pop("AMP_MCP_URL", None)
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from chat_recall_eval import FACTS, FILLERS, normalise  # noqa: E402
from harness import chat_model  # noqa: E402
from harness.chat_session import ChatSession  # noqa: E402

LABEL = sys.argv[1] if len(sys.argv) > 1 else "run"
TOKEN_LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{LABEL}.json")
KEYWORDS = [normalise(k) for _, _, k in FACTS]

real = chat_model._client(None)
calls = []


class Recording:
    def chat(self, **kwargs):
        reply = real.chat(**kwargs)
        calls.append((reply.prompt_eval_count or 0, reply.eval_count or 0))
        return reply

    def list(self):
        return real.list()


chat_model._client = lambda host: Recording()


def turn(chat, kind, message, keyword=None):
    calls.clear()
    started = time.monotonic()
    answer = chat.ask(message)
    paged = [{"distance": round(m["distance"], 3), "tokens": m["tokens"],
              "facts": [k for k in KEYWORDS if k in normalise(m["text"])]} for m in chat.last_paged]
    row = {"kind": kind, "message": message, "answer": answer, "seconds": round(time.monotonic() - started, 1),
           "paged": paged, "paged_tokens": sum(m["tokens"] for m in paged),
           "prompt_tokens": sum(c[0] for c in calls), "reply_tokens": sum(c[1] for c in calls),
           "says_memories": "according to the memories" in answer.lower()}
    if keyword is not None:
        key = normalise(keyword)
        row.update(keyword=keyword, recalled=key in normalise(answer),
                   relevant_distances=[m["distance"] for m in paged if key in m["facts"]])
    return row


chat = ChatSession(session_id=f"cutoff-{LABEL}", token_limit=TOKEN_LIMIT)
rows = []
started = time.monotonic()
for i, (statement, _, _) in enumerate(FACTS):
    rows.append(turn(chat, "fact", statement))
    rows.append(turn(chat, "filler", FILLERS[i]))
    print(f"[{LABEL}] planted {i + 1}/16 | compactions {chat.compactions} | paged {rows[-1]['paged_tokens']} tokens",
          flush=True)
for statement, question, keyword in FACTS:
    rows.append(turn(chat, "probe", question, keyword))
    r = rows[-1]
    print(f"[{LABEL}] {'OK ' if r['recalled'] else 'MISS'} {question} | relevant at {r['relevant_distances']} | "
          f"paged {len(r['paged'])} ({r['paged_tokens']} tokens)", flush=True)
chat.wait_for_reclaim()
probes = [r for r in rows if r["kind"] == "probe"]
result = {"label": LABEL, "token_limit": TOKEN_LIMIT, "max_distance": os.environ.get("AMP_PAGE_MAX_DISTANCE", "none"),
          "recalled": sum(r["recalled"] for r in probes), "compactions": chat.compactions,
          "vault_size": len(chat.vault()), "seconds": round(time.monotonic() - started, 1), "rows": rows}
json.dump(result, open(OUT, "w"), indent=1)
print(f"\n[{LABEL}] recalled {result['recalled']}/16 | compactions {result['compactions']} | {result['seconds']} s")
for kind in ("fact", "filler", "probe"):
    part = [r for r in rows if r["kind"] == kind]
    print(f"  {kind:6s}: paged {sum(r['paged_tokens'] for r in part) / len(part):6.0f} tokens/turn, "
          f"prompt {sum(r['prompt_tokens'] for r in part) / len(part):6.0f} tokens/turn, "
          f"{sum(r['seconds'] for r in part) / len(part):5.1f} s/turn, 'according to the memories' "
          f"{sum(r['says_memories'] for r in part)}/{len(part)}")
