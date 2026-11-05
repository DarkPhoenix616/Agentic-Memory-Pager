import sys
import json

payload = json.load(sys.stdin)
code = payload["code"]
memories = payload["memories"]

# `memories` is a list of UnifiedMemory records as plain JSON dicts with keys:
#   id, timestamp (ISO string), session_id, memory_type ("text" or "code"),
#   compaction_source ("rlm" or "ast"), raw_token_count, vector_embedding_id,
#   text_payload (dict or None), code_payload (dict or None)
if not isinstance(memories, list):
    print("TypeError: memories must be a list", file=sys.stderr)
    sys.exit(1)

namespace = {"memories": memories}

try:
    exec(compile(code, "<sandbox>", "exec"), namespace)
except Exception as e:
    tb = e.__traceback__
    lineno = None
    while tb:
        if tb.tb_frame.f_code.co_filename == "<sandbox>":
            lineno = tb.tb_lineno
        tb = tb.tb_next
    where = f" (line {lineno})" if lineno else ""
    print(f"{type(e).__name__}: {e}{where}", file=sys.stderr)
    sys.exit(1)