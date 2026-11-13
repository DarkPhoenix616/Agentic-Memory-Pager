"""Build a ~50k-token developer/assistant chat with planted facts; write chat.txt and facts.json."""
import json, random
random.seed(7)
comps = ["pager", "vault", "router", "tokenizer", "scheduler", "indexer", "auditor", "signer", "sandbox", "dashboard"]
techs = ["ChromaDB", "FAISS", "LangGraph", "FastMCP", "tiktoken", "Pydantic", "Streamlit", "Ollama", "SQLAlchemy", "Redis",
         "msgpack", "orjson", "uvicorn", "httpx", "pytest-xdist", "tree-sitter", "Langfuse", "LiteLLM", "zstandard", "watchdog"]
errors = ["KeyError", "TypeError", "ValueError", "IndexError", "TimeoutError", "RecursionError", "UnicodeDecodeError",
          "AttributeError", "ConnectionResetError", "JSONDecodeError"]
filler_q = ["Can you explain how the {c} talks to the rest of the system?",
            "What would happen to the {c} if the context window filled up during a long session?",
            "Walk me through the data flow of the {c} step by step.",
            "How should we test the {c} without a running model?"]
para = ("The {c} receives chunks from the token monitor, checks their type, and either keeps them in the active window "
        "or hands them to the compaction stage. Each chunk carries its role, its content and a type flag, and the {c} "
        "never edits content in place; it only moves whole chunks so the history stays reproducible. When the budget is "
        "close to the limit, the oldest chunk is selected first, which keeps recent turns verbatim for the model. ")
code = "```python\ndef handle_{c}(state):\n    chunks = state['context_chunks']\n    if not chunks:\n        return {{}}\n    oldest = chunks[0]\n    return {{'context_chunks': chunks[1:], 'last': oldest['type']}}\n```\n"
facts = {"decisions": [], "bugs": [], "constraints": []}
turns = []
def filler(i):
    c = comps[i % len(comps)]
    turns.append("User: " + random.choice(filler_q).format(c=c))
    turns.append("Assistant: " + (para.format(c=c) * 3) + "\n" + code.format(c=c) + para.format(c=c))
for i in range(130):
    filler(i)
    if i % 6 == 1 and len(facts["decisions"]) < 20:
        n = len(facts["decisions"]); t = techs[n]; c = comps[n % len(comps)]
        turns.append(f"User: For the {c}, should we use {t} or write our own?")
        turns.append(f"Assistant: We decided to use {t} for the {c}, because it keeps the {c} under {100 + n} lines and the team already knows it.")
        facts["decisions"].append(t)
    if i % 6 == 3 and len(facts["bugs"]) < 20:
        n = len(facts["bugs"]); e = errors[n % len(errors)]; f = f"{comps[n % len(comps)]}_v{n}.py"
        turns.append(f"User: The tests crash with {e} in {f} when the session is empty.")
        turns.append(f"Assistant: Fixed it. The {e} in {f} came from reading chunk {n} before the list was filled; I added a guard and test_{n} now passes.")
        facts["bugs"].append(f)
    if i % 13 == 5 and len(facts["constraints"]) < 10:
        n = len(facts["constraints"])
        turns.append(f"User: From now on, every {comps[n]} call must finish in under {200 + 10 * n} ms. Don't relax that.")
        turns.append(f"Assistant: Understood, I will keep the {comps[n]} under {200 + 10 * n} ms.")
        facts["constraints"].append(f"{200 + 10 * n} ms")
open("chat.txt", "w").write("\n\n".join(turns))
json.dump(facts, open("facts.json", "w"), indent=1)
print(len(turns), "turns", {k: len(v) for k, v in facts.items()})
