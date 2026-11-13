"""Usage: recall_text.py facts.json run.txt -> facts.json-style recall from the CLI's text output."""
import json, re, sys
facts = json.load(open(sys.argv[1])); text = open(sys.argv[2]).read()
main = text.split("\n--- Piece")[0]
def section(title):
    m = re.search(rf"^{title} \(\d+\)\n((?:  - .*\n)*)", main, re.M)
    return [l[4:] for l in m.group(1).splitlines()] if m else []
summary = re.search(r"^Summary\n  (.*)$", main, re.M).group(1)
lists = {"decisions": section("Architectural decisions"), "bugs": section("Bugs resolved"), "constraints": section("User constraints")}
everything = main.lower()
for kind, anchors in facts.items():
    in_list = [a for a in anchors if any(a.lower() in i.lower() for i in lists[kind])]
    in_summary = [a for a in anchors if a.lower() in summary.lower()]
    missing = [a for a in anchors if a.lower() not in everything]
    print(f"{kind:12s} in its list {len(in_list):2d}/{len(anchors)} | in summary {len(in_summary):2d}/{len(anchors)} | missing anywhere: {missing}")
print("list sizes:", {k: len(v) for k, v in lists.items()}, "| summary chars:", len(summary))
