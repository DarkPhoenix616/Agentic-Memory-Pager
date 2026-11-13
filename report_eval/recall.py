"""Usage: recall.py facts.json memory.json -> how many planted facts survived, per list and anywhere."""
import json, sys
facts = json.load(open(sys.argv[1])); m = json.load(open(sys.argv[2]))
lists = {"decisions": m["architectural_decisions"], "bugs": m["bugs_resolved"], "constraints": m["user_constraints"]}
everything = json.dumps(m).lower()
for kind, anchors in facts.items():
    in_list = [a for a in anchors if any(a.lower() in item.lower() for item in lists[kind])]
    anywhere = [a for a in anchors if a.lower() in everything]
    missing = [a for a in anchors if a.lower() not in everything]
    print(f"{kind:12s} in its list {len(in_list):2d}/{len(anchors)} | anywhere {len(anywhere):2d}/{len(anchors)} | missing {missing}")
print("list sizes:", {k: len(v) for k, v in lists.items()}, "| summary chars:", len(m["factual_summary"]))
