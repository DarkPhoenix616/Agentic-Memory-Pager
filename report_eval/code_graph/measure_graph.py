"""
Lighter code graph (final round, task 8): what do the call lines in a code
memory's search text cost, and what do they buy?

No model and no vault: every Python file of the repo (not tests) is compacted
with the AST compactor, and its search text is built four ways:

  none   file path, summary, signatures (no call lines)
  full   + one line per function: "ChatSession.save_all calls ChatSession.wait_for_reclaim, ..."   (today)
  short  + the same lines, without the caller's own class in front of its methods
  file   + one line for the whole file: "Calls: a, b, c" (each name once)

For each: total tokens (what page_context charges against the budget) and
search quality with ChromaDB's embedding model (all-MiniLM-L6-v2, squared L2):

  who-calls   "Which function calls compact_code?" -> a file with a caller
              (only callees the caller's file does not define: the case the
              graph is for; a callee defined in the same file is findable by
              its own signature)
  where       "Where is the function save_all defined?" -> its file (no-harm check)

Run from the repo root:  .venv/bin/python -I report_eval/code_graph/measure_graph.py
Writes results.json next to this file.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np  # noqa: E402
import tiktoken  # noqa: E402
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction  # noqa: E402
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from compactor.ast_code.code_parser import compact_code  # noqa: E402

ENC = tiktoken.get_encoding("cl100k_base")
EMBED = DefaultEmbeddingFunction()
MAX_EMBED_TOKENS = 256           # all-MiniLM-L6-v2 reads at most this many word pieces
_model = ONNXMiniLM_L6_V2()
_model(["warm up"])
WORDPIECES = Tokenizer.from_str(_model.tokenizer.to_str())   # a copy, so the model's own truncation is untouched
WORDPIECES.no_truncation()
WORDPIECES.no_padding()


def own_prefix(caller: str, callee: str) -> str:
    owner = caller.rsplit(".", 1)[0] + "." if "." in caller else None
    return callee[len(owner):] if owner and callee.startswith(owner) else callee


def search_text(schema, variant: str) -> str:
    parts = [schema.file_path]
    if schema.summary:
        parts.append(schema.summary)
    parts.extend(s.signature for s in schema.symbols)
    linked = [s for s in schema.symbols if s.calls]
    if variant == "full":
        parts.extend(f"{s.name} calls {', '.join(s.calls)}" for s in linked)
    elif variant == "short":
        parts.extend(f"{s.name} calls {', '.join(own_prefix(s.name, c) for c in s.calls)}" for s in linked)
    elif variant == "file":
        names = list(dict.fromkeys(c for s in linked for c in s.calls))
        if names:
            parts.append("Calls: " + ", ".join(names))
    return "\n".join(parts)


files = sorted(f for pattern in ("app/*.py", "compactor/**/*.py", "harness/*.py", "server/*.py")
               for f in glob.glob(pattern, recursive=True) if not f.endswith("__init__.py"))
schemas = {}
for path in files:
    with open(path, encoding="utf-8") as handle:
        schemas[path] = compact_code(handle.read(), file_path=path, language="python")

defined = {path: {s.name for s in schema.symbols} | {s.name.rsplit(".", 1)[-1] for s in schema.symbols}
           for path, schema in schemas.items()}
callers_of = {}
for path, schema in schemas.items():
    for symbol in schema.symbols:
        for callee in symbol.calls:
            if callee not in defined[path] and callee.rsplit(".", 1)[-1] not in defined[path]:
                callers_of.setdefault(callee.rsplit(".", 1)[-1], set()).add(path)
who_calls = [(f"Which function calls {name}?", targets) for name, targets in sorted(callers_of.items())]
where = []
for path, schema in schemas.items():
    funcs = [s.name for s in schema.symbols if s.type in ("function", "method")][:2]
    where += [(f"Where is the function {name.rsplit('.', 1)[-1]} defined?", {path}) for name in funcs]

paths = list(schemas)
result = {"files": len(paths), "symbols": sum(len(s.symbols) for s in schemas.values()),
          "links": sum(len(sym.calls) for s in schemas.values() for sym in s.symbols),
          "who_calls_queries": len(who_calls), "where_queries": len(where), "variants": {}}
query_vecs = {q: np.array(v) for q, v in zip([q for q, _ in who_calls + where],
                                                  EMBED([q for q, _ in who_calls + where]))}
for variant in ("none", "full", "short", "file"):
    texts = [search_text(schemas[p], variant) for p in paths]
    tokens = [len(ENC.encode(t)) for t in texts]
    vecs = np.array(EMBED(texts))

    def hits(queries):
        top1 = top3 = 0
        for question, targets in queries:
            d = ((vecs - query_vecs[question]) ** 2).sum(axis=1)
            order = [paths[i] for i in np.argsort(d)]
            top1 += order[0] in targets
            top3 += any(p in targets for p in order[:3])
        return top1, top3

    w1, w3 = hits(who_calls)
    h1, h3 = hits(where)
    over = sum(1 for t in texts if len(WORDPIECES.encode(t).ids) > MAX_EMBED_TOKENS)
    result["variants"][variant] = {"tokens": sum(tokens), "who_calls_top1": w1, "who_calls_top3": w3,
                                   "where_top1": h1, "where_top3": h3, "texts_longer_than_embedding_window": over}
    print(f"{variant:6s} tokens {sum(tokens):6d} | who-calls top1 {w1}/{len(who_calls)} top3 {w3}/{len(who_calls)} | "
          f"where top1 {h1}/{len(where)} top3 {h3}/{len(where)} | over the embedding window: {over}")
base = result["variants"]["none"]["tokens"]
for variant, row in result["variants"].items():
    row["growth_vs_none"] = round(100 * (row["tokens"] - base) / base, 1)
print("growth vs none:", {v: f"{r['growth_vs_none']}%" for v, r in result["variants"].items()})
print(f"{result['files']} files, {result['symbols']} symbols, {result['links']} call links")
json.dump(result, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json"), "w"), indent=1)

# What the model is shown when a code memory is paged in: the first 5 signatures, and since
# task 8 also their calls (harness.graph._memory_view). Measured per file, before and after.
from harness import graph as amp_graph  # noqa: E402
from harness.guardrails import UnifiedMemory  # noqa: E402

shown = {"signatures_only": 0, "with_calls": 0}
for path, schema in schemas.items():
    memory = UnifiedMemory(session_id="m", memory_type="code", compaction_source="ast", raw_token_count=1,
                           code_payload=schema)
    with_calls = amp_graph._memory_view(memory)[1]
    shown["with_calls"] += len(ENC.encode(with_calls))
    shown["signatures_only"] += len(ENC.encode(with_calls.split("\nCalls: ")[0]))
shown["files"] = len(schemas)
result["shown_to_model"] = shown
print(f"shown to the model per paged code memory: {shown['signatures_only'] / len(schemas):.0f} tokens -> "
      f"{shown['with_calls'] / len(schemas):.0f} with the calls line")
json.dump(result, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json"), "w"), indent=1)
