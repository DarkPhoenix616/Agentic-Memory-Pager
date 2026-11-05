import sys
import json

payload = json.load(sys.stdin)
code = payload["code"]
memories = payload["memories"]

# `memories` is a list of JSON dicts. The memory server's run_memory_code tool
# passes one simple record per memory, oldest first:
#   id, time (ISO string), kind ("text" or "code"), text (the searchable text),
#   decisions, bugs, constraints (lists of strings), file (code file path or "")
if not isinstance(memories, list):
    print("TypeError: memories must be a list", file=sys.stderr)
    sys.exit(1)


def _limit_resources(cpu_seconds):
    """Linux/macOS: cap CPU time, memory and file size, and forbid new processes,
    before any model-written code runs. The operating system enforces these even
    if code slips past check_code. (Windows has no `resource` module.)"""
    try:
        import resource
    except ImportError:
        return
    limits = {
        "RLIMIT_CPU": cpu_seconds,           # seconds of CPU time
        "RLIMIT_AS": 512 * 1024 * 1024,      # bytes of memory
        "RLIMIT_FSIZE": 1024 * 1024,         # bytes per written file
        "RLIMIT_NPROC": 0,                   # no new processes or threads
        "RLIMIT_CORE": 0,                    # no core dumps
    }
    for name, value in limits.items():
        limit = getattr(resource, name, None)
        if limit is None:
            continue
        try:
            resource.setrlimit(limit, (value, value))
        except (ValueError, OSError):
            pass                             # e.g. the hard limit is already lower


_limit_resources(int(payload.get("cpu_seconds", 6)))
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
