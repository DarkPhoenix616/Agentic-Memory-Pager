"""Do user-stated facts survive the RLM summarizer?

For each saved exchange (exchanges.json) the chunk "User: ...\\n\\nAssistant: ..."
goes through compactor.rlm_text.distill (llama3.1:8b, temperature 0, seed 0),
and every planted fact is looked for in the stored factual_summary (case
insensitive). The 12 exchanges are also joined into one long chat, so the
multi-piece path (map + reduce) is tested too.

Run from the repo root:  .venv/bin/python -I report_eval/user_facts_eval/eval_user_facts.py LABEL
Writes results_LABEL.json next to this file.
"""
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.getcwd())
logging.disable(logging.INFO)
from compactor.ast_code import count_tokens  # noqa: E402
from compactor.rlm_text.recursive_summarizer import RLMConfig, distill  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def found(fact, text):
    return fact.lower() in text.lower()


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    cases = json.load(open(os.path.join(HERE, "exchanges.json")))
    config = RLMConfig()
    rows, kept, total = [], 0, 0
    started = time.monotonic()
    for case in cases:
        chunk = f"User: {case['question']}\n\nAssistant: {case['answer']}"
        memory = distill(chunk, config=config).memory
        hits = {f: found(f, memory.factual_summary) for f in case["facts"]}
        kept += sum(hits.values())
        total += len(hits)
        rows.append({"question": case["question"], "facts": hits, "summary": memory.factual_summary,
                     "in_tokens": case["tokens"], "out_tokens": count_tokens(memory.factual_summary)})
        lost = [f for f, ok in hits.items() if not ok]
        print(f"{sum(hits.values())}/{len(hits)} kept  {case['tokens']:3d}->{rows[-1]['out_tokens']:3d} tokens"
              f"  lost={lost}  {case['question'][:50]!r}")

    joined = "\n\n".join(f"User: {c['question']}\n\nAssistant: {c['answer']}" for c in cases)
    result = distill(joined, config=config)
    all_facts = [f for c in cases for f in c["facts"]]
    long_hits = {f: found(f, result.memory.factual_summary) for f in all_facts}
    seconds = time.monotonic() - started

    summary = {
        "label": label,
        "single_exchanges": {"kept": kept, "total": total},
        "long_chat": {"kept": sum(long_hits.values()), "total": len(all_facts), "pieces": len(result.pieces),
                      "reduce_levels": result.reduce_levels, "in_tokens": count_tokens(joined),
                      "out_tokens": result.summary_tokens,
                      "lost": [f for f, ok in long_hits.items() if not ok]},
        "out_tokens_single_total": sum(r["out_tokens"] for r in rows),
        "seconds": round(seconds, 1),
    }
    print(json.dumps(summary, indent=1))
    json.dump({"summary": summary, "rows": rows, "long_chat_summary": result.memory.factual_summary},
              open(os.path.join(HERE, f"results_{label}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
