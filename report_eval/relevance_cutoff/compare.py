"""Side-by-side numbers for runs of recall_cutoff_eval.py.

    python3 compare.py A_no_cutoff.json B_cutoff_1p4.json
"""
import json
import sys

runs = [json.load(open(path)) for path in sys.argv[1:]]
print("run".ljust(28), *[r["label"].rjust(16) for r in runs])
print("recalled (of 16)".ljust(28), *[str(r["recalled"]).rjust(16) for r in runs])
print("compactions".ljust(28), *[str(r["compactions"]).rjust(16) for r in runs])
print("total minutes".ljust(28), *[f"{r['seconds'] / 60:.1f}".rjust(16) for r in runs])
for kind in ("fact", "filler", "probe"):
    for name, value in (("memories paged / turn", lambda rows: sum(len(r["paged"]) for r in rows) / len(rows)),
                        ("paged tokens / turn", lambda rows: sum(r["paged_tokens"] for r in rows) / len(rows)),
                        ("prompt tokens / turn", lambda rows: sum(r["prompt_tokens"] for r in rows) / len(rows)),
                        ("seconds / turn", lambda rows: sum(r["seconds"] for r in rows) / len(rows)),
                        ("'according to the memories'", lambda rows: sum(r["says_memories"] for r in rows))):
        cells = []
        for r in runs:
            rows = [x for x in r["rows"] if x["kind"] == kind]
            cells.append(f"{value(rows):.1f}".rjust(16))
        print(f"{kind:6s} {name}".ljust(28), *cells)
for r in runs:
    misses = [x["message"] for x in r["rows"] if x["kind"] == "probe" and not x["recalled"]]
    print(f"{r['label']}: misses {misses}")
