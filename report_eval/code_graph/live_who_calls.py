"""
Live check for task 8: with the call lines out of the search text, can the real
model still answer call-graph questions through run_memory_code's `calls`?

Two code memories (shop.py and payments.py, compacted by the AST compactor,
signed, stored through the memory server) on a temporary vault. Questions are
asked RUNS times each in fresh chats.

Run from the repo root:  .venv/bin/python -I report_eval/code_graph/live_who_calls.py <out name> <runs>
"""
import json
import os
import secrets
import sys
import tempfile

W = tempfile.mkdtemp(prefix="amp-graph-")
os.environ.update(AMP_CHROMA_PATH=f"{W}/vault", AMP_SESSION_KEYS_FILE=f"{W}/keys", AMP_CHATS_DIR=f"{W}/chats",
                  AMP_AUDIT_LOG=f"{W}/audit.log", AMP_ADMIN_TOKEN=secrets.token_hex(16),
                  AMP_SIGNING_KEY=secrets.token_hex(32))
os.environ.pop("AMP_MCP_URL", None)
sys.path.insert(0, os.getcwd())

import server.amp_memory_server as server  # noqa: E402
from compactor.ast_code.code_parser import compact_code  # noqa: E402
from harness.chat_session import ChatSession  # noqa: E402
from harness.guardrails import UnifiedMemory  # noqa: E402
from harness.local_keys import session_key  # noqa: E402
from harness.signing import sign_memory  # noqa: E402

RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
OUT = sys.argv[1] if len(sys.argv) > 1 else "live_who_calls"
SHOP = '''from payments import charge, refund

def load_cart(user_id):
    """Read the user's cart from the database."""
    return []

def checkout(user_id):
    """Pay for everything in the cart."""
    cart = load_cart(user_id)
    return charge(user_id, sum(item["price"] for item in cart))

def cancel_order(user_id, order_id):
    """Give the money back for an order."""
    return refund(user_id, order_id)
'''
PAYMENTS = '''def charge(user_id, amount):
    """Take money from the user's card."""
    return {"user": user_id, "amount": amount}

def refund(user_id, order_id):
    """Return the money for an order."""
    return {"user": user_id, "order": order_id}
'''
QUESTIONS = [
    ("tool asked", "Use run_memory_code to find which function in my code calls charge.", "checkout"),
    ("natural", "Which function in my shop code calls refund?", "cancel_order"),
    ("callees", "Use run_memory_code to list what checkout calls.", "charge"),
]


def plant(sid):
    for source, path in ((SHOP, "shop.py"), (PAYMENTS, "payments.py")):
        memory = UnifiedMemory(session_id=sid, memory_type="code", compaction_source="ast", raw_token_count=80,
                               code_payload=compact_code(source, file_path=path), provenance="user_input")
        memory.signature = sign_memory(memory)
        server.store_memory(memory, session_key=session_key(sid))


rows = []
for label, question, expected in QUESTIONS:
    for n in range(1, RUNS + 1):
        sid = f"graph-{label.replace(' ', '-')}-{n}"
        plant(sid)
        chat = ChatSession(session_id=sid, token_limit=6144)
        chat.background = False
        reply = chat.ask(question)
        code = [e.detail for e in chat.events if e.kind == "code" and "wrote Python" in e.title]
        ran = [e.detail for e in chat.events if e.kind == "code" and "Sandbox ran" in e.title]
        blocked = [e.title for e in chat.events if e.kind == "blocked"]
        right = expected in reply.lower()
        rows.append({"label": label, "run": n, "question": question, "expected": expected, "right": right,
                     "used_calls": any("calls" in c for c in code), "code": code, "sandbox": ran,
                     "blocked": blocked, "reply": reply})
        print(f"{label:10s} run {n}: {'RIGHT' if right else 'WRONG'} | used calls: {rows[-1]['used_calls']} | "
              f"blocked: {len(blocked)} | {reply[:110]!r}", flush=True)
json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{OUT}.json"), "w"), indent=1)
print(f"{OUT}: right {sum(r['right'] for r in rows)}/{len(rows)}, used calls {sum(r['used_calls'] for r in rows)}")
