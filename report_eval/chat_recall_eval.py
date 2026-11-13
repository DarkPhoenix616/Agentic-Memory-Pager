"""
AMP vs a plain sliding-window chatbot: can the bot recall facts from early in a long chat?

Both systems run the same code, model, prompts and context window. The only
difference is the vault:
    amp       compacted chunks are stored in ChromaDB and paged back in
    baseline  compacted chunks are thrown away (plain chatbot that drops old messages)

Script: 16 fact turns, each followed by a filler question with a long answer
(this pushes the facts out of the window), then 16 probe questions, one per
fact. A probe counts as recalled if the answer contains the expected keyword.

Usage (from the repo):
    cd BTP/AgenticMemoryPager
    PYTHONPATH=$PWD .venv/bin/python report_eval/chat_recall_eval.py [token_limit] [out.json]
"""
import json
import re
import sys
import time

from compactor.rlm_text import recursive_summarizer as rs
from harness import chat_model
from harness import graph as amp
from harness.chat_session import ChatSession
from harness.guardrails import PageContextResult

FACTS = [  # (statement, probe question, expected keyword)
    ("Please remember: my cat is called Pixel.", "What is my cat's name?", "pixel"),
    ("Note this: our project codename is BLUEHERON.", "What is our project codename?", "blueheron"),
    ("Rule for this project: the vault must use ChromaDB, never SQLite.", "Which database did I say we must never use?", "sqlite"),
    ("FYI, my teammate Aman owns the guardrails module.", "Who owns the guardrails module?", "aman"),
    ("Remember that our final demo is on 14 December.", "On which date is our final demo?", "december"),
    ("Note: the memory server runs on port 8001.", "Which port does the memory server run on?", "8001"),
    ("Our laptop GPU is an RTX 3050 with 6 GB.", "Which GPU model does our laptop have?", "3050"),
    ("The bug in pager_v3.py was an off-by-one error in the eviction loop.", "What was the bug in pager_v3.py?", "off by one"),
    ("We chose LangGraph for orchestration.", "Which library did we choose for orchestration?", "langgraph"),
    ("My project supervisor is Dr. Rao.", "Who is my project supervisor?", "rao"),
    ("My favourite editor is Neovim.", "What is my favourite editor?", "neovim"),
    ("My lucky number is 47.", "What is my lucky number?", "47"),
    ("The embedding model we use is all-MiniLM-L6-v2.", "Which embedding model do we use?", "minilm"),
    ("Constraint: the final report must be under 30 pages.", "What page limit did I give for the final report?", "30"),
    ("I live in Jodhpur.", "Which city do I live in?", "jodhpur"),
    ("Our backup model is qwen3:8b.", "What is our backup model?", "qwen3"),
]
FILLERS = [
    "Explain in detail how hash tables handle collisions.",
    "Explain the CAP theorem with examples.",
    "Explain in detail how TCP congestion control works.",
    "Explain Python decorators with an example.",
    "What is the difference between processes and threads? Explain in detail.",
    "Explain in detail how a B-tree index works.",
    "Explain in detail how garbage collection works in Java.",
    "Explain the transformer attention mechanism in detail.",
    "Explain the SOLID principles with examples.",
    "Explain in detail how virtual memory and paging work in an operating system.",
    "Explain in detail how HTTPS establishes a secure connection.",
    "Explain dynamic programming using the knapsack problem.",
    "Explain eventual consistency with examples.",
    "Explain in detail how Git stores commits internally.",
    "Explain in detail how vector databases find nearest neighbours.",
    "Explain MapReduce with an example.",
]


def normalise(text):
    text = text.lower().replace(",", "").replace("-", " ")
    return re.sub(r"\s+", " ", text)


def run(system, token_limit, calls):
    patches = {}
    if system == "baseline":                      # no vault: compacted chunks are discarded
        patches = {
            (amp, "_mcp_store_memory"): lambda memory: memory,
            (amp, "_mcp_page_context"): lambda **kw: PageContextResult(total_budget=kw.get("total_budget", 1000)),
            (amp, "_mcp_search_vault"): lambda **kw: [],
            (amp, "_mcp_list_all_memories"): lambda: [],
            (rs, "default_llm"): lambda config: None,   # the summary is thrown away, so skip the model
        }
    saved = {key: getattr(*key) for key in patches}
    for (module, name), value in patches.items():
        setattr(module, name, value)
    try:
        chat = ChatSession(session_id=f"eval-{system}-{int(time.time())}", token_limit=token_limit)
        started = time.monotonic()
        for i, (statement, _, _) in enumerate(FACTS):
            chat.ask(statement)
            chat.ask(FILLERS[i])
            print(f"[{system}] planted {i + 1}/16 | context {chat.context_tokens} | compactions {chat.compactions}", flush=True)
        probes = []
        for statement, question, keyword in FACTS:
            calls.clear()
            t = time.monotonic()
            answer = chat.ask(question)
            hit = normalise(keyword) in normalise(answer)
            paged = chat.last_paged
            probes.append({
                "question": question, "keyword": keyword, "answer": answer, "recalled": hit,
                "seconds": round(time.monotonic() - t, 1),
                "memories_paged_in": len(paged),
                "fact_in_paged_memories": any(normalise(keyword) in normalise(m["text"]) for m in paged),
                "fact_in_context": any(normalise(keyword) in normalise(c.get("content", "")) for c in chat.chunks[:-3]),
                "prompt_tokens": sum(c[0] for c in calls), "reply_tokens": sum(c[1] for c in calls),
            })
            print(f"[{system}] {'OK ' if hit else 'MISS'} {question} -> {answer[:90]!r}", flush=True)
        chat.wait_for_reclaim()
        vault = chat.vault()
        result = {
            "system": system, "token_limit": token_limit, "recalled": sum(p["recalled"] for p in probes),
            "probes": probes, "compactions": chat.compactions, "tokens_paged_out": chat.tokens_evicted,
            "tokens_stored": chat.tokens_stored, "vault_size": len(vault), "seconds": round(time.monotonic() - started, 1),
            "turn_seconds": [round(e.seconds, 1) for e in chat.events if e.kind == "reply"],
        }
        for memory in vault:
            amp._mcp_delete_memory(memory["id"])
        return result
    finally:
        for (module, name), value in saved.items():
            setattr(module, name, value)


def main():
    token_limit = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    out = sys.argv[2] if len(sys.argv) > 2 else "chat_recall_eval.json"
    real = chat_model._client(None)
    calls = []

    class Recording:                            # records prompt/reply token counts from Ollama
        def chat(self, **kwargs):
            reply = real.chat(**kwargs)
            calls.append((reply.prompt_eval_count or 0, reply.eval_count or 0))
            return reply

        def list(self):
            return real.list()

    chat_model._client = lambda host: Recording()
    results = [run("amp", token_limit, calls), run("baseline", token_limit, calls)]
    with open(out, "w") as handle:
        json.dump(results, handle, indent=1)
    for r in results:
        print(f"{r['system']:9s} recalled {r['recalled']}/16 | compactions {r['compactions']} | "
              f"{r['tokens_paged_out']} -> {r['tokens_stored']} tokens | {r['seconds']}s total")


if __name__ == "__main__":
    main()
