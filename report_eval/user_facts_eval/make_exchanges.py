"""Make the test set for the "summary drops user facts" problem.

12 exchanges: the user states small facts about themselves or the project and
asks for a ~150-word explanation; llama3.1:8b writes the answer (as in AMP Chat:
temperature 0.3, num_ctx 8192). Answers are saved, so every summarizer version
is tested on exactly the same text.

Run from the repo root:  .venv/bin/python -I report_eval/user_facts_eval/make_exchanges.py
Writes exchanges.json next to this file.
"""
import json
import os
import sys

import ollama

sys.path.insert(0, os.getcwd())
from compactor.ast_code import count_tokens  # noqa: E402

CASES = [
    ("My cat is called Pixel and my codename is ZEBRAFISH. Explain in about 150 words what a vector database is.",
     ["Pixel", "ZEBRAFISH"]),
    ("I'm Akshay from IIT Jodhpur, and my teammate Aman wrote the guardrails. Explain in about 150 words how HNSW indexing works.",
     ["Akshay", "Jodhpur", "Aman"]),
    ("Our demo is on 14 December in room LHC-104. Explain in about 150 words what an embedding is.",
     ["December", "LHC-104"]),
    ("My laptop is a Lenovo LOQ with an RTX 3050. Explain in about 150 words what quantization of an LLM means.",
     ["Lenovo", "LOQ", "RTX 3050"]),
    ("The project's internal name is Orion and the vault server runs on port 8001. Explain in about 150 words what JSON-RPC is.",
     ["Orion", "8001"]),
    ("Dr. Mehta wants the final report in IEEE format. Explain in about 150 words what a context window is.",
     ["Mehta", "IEEE"]),
    ("Our backup model is qwen3:8b and the main one is llama3.1:8b. Explain in about 150 words what temperature does when sampling.",
     ["qwen3:8b", "llama3.1:8b"]),
    ("My dog Bruno chewed my notes, so I lost the slides for Friday. Explain in about 150 words what a transformer layer does.",
     ["Bruno", "Friday"]),
    ("We keep the backups in a bucket called amp-backups-2026 in Mumbai. Explain in about 150 words what cosine similarity is.",
     ["amp-backups-2026", "Mumbai"]),
    ("Sanya from the design team picked teal as the main colour of the UI. Explain in about 150 words what a REST API is.",
     ["Sanya", "teal"]),
    ("My Wi-Fi network is called AMP_LAB_5G. Explain in about 150 words how tokenization works.",
     ["AMP_LAB_5G"]),
    ("Remember that our client is Tata Motors and the contract ends in March. Explain in about 150 words what a knowledge graph is.",
     ["Tata Motors", "March"]),
]


def main():
    client = ollama.Client(timeout=300)
    out = []
    for question, facts in CASES:
        for attempt in range(3):
            reply = client.chat(model="llama3.1:8b", messages=[{"role": "user", "content": question}],
                                options={"temperature": 0.3, "num_ctx": 8192, "num_predict": 500},
                                keep_alive="10m").message.content.strip()
            chunk = f"User: {question}\n\nAssistant: {reply}"
            if count_tokens(chunk) > 150:          # longer than the verbatim limit, so the summarizer runs
                break
        out.append({"question": question, "answer": reply, "facts": facts, "tokens": count_tokens(chunk)})
        print(f"{count_tokens(chunk):4d} tokens  {question[:60]!r}")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exchanges.json")
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
