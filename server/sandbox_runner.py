import sys
import json

payload = json.load(sys.stdin)
code = payload["code"]
namespace = {"memories": payload["memories"]}

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