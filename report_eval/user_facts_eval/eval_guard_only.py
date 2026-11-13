"""The user-fact guard on its own: the new code, but with the OLD prompts (as on
akshaybranch before the fix), so any fact that survives was put back by the guard.

Run from the repo root:  .venv/bin/python -I report_eval/user_facts_eval/eval_guard_only.py
Writes results_guard_only.json next to this file.
"""
import os
import re
import subprocess
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compactor.rlm_text import recursive_summarizer as rs  # noqa: E402

old = subprocess.run(["git", "show", "8700ea7:compactor/rlm_text/recursive_summarizer.py"],
                     capture_output=True, text=True, check=True).stdout
for name in ("_MAP_SYSTEM", "_REDUCE_SYSTEM"):
    setattr(rs, name, re.search(rf'^{name} = """(.*?)"""', old, re.S | re.M).group(1))
assert "facts the user stated" not in rs._MAP_SYSTEM

import eval_user_facts  # noqa: E402

sys.argv = [sys.argv[0], "guard_only"]
eval_user_facts.main()
