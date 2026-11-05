#A sandbox will help you test your tools and memory functions without affecting the main memory database and prevents any python injection attack possibilities.
import os
import signal
import subprocess
import sys
import tempfile
import ast
import json

RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox_runner.py")

DEFAULT_TIMEOUT = 5  # seconds
MAX_PAYLOAD_BYTES = 5_000_000  # hard ceiling on the code + memories sent to the child


def _child_env():
    """Give the child almost no environment, so it cannot read API keys."""
    env = {}
    if os.name == "nt":
        # Python on Windows may fail to start without this variable
        env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
    return env

MAX_OUTPUT_CHARS = 4000  # per stream (output and error)

ALLOWED_IMPORTS = {
    "json", "re", "math", "statistics", "collections",
    "itertools", "datetime", "string",
}
BLOCKED_NAMES = {
    "open", "exec", "eval", "compile", "input", "breakpoint",
    "globals", "locals", "vars", "getattr", "setattr", "delattr",
    "exit", "quit", "help",
}


def check_code(code):
    """Return a problem description if the code looks unsafe, else None."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e.msg} (line {e.lineno})"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    return f"import of '{root}' is not allowed"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level or root not in ALLOWED_IMPORTS:
                return f"import from '{node.module}' is not allowed"
        elif isinstance(node, ast.Name):
            if node.id in BLOCKED_NAMES or node.id.startswith("__"):
                return f"use of '{node.id}' is not allowed"
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                return f"access to '{node.attr}' is not allowed"
    return None


def _truncate(text, limit=MAX_OUTPUT_CHARS):
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated, {len(text) - limit} more characters]"


def _result(ok, output, error, loaded, scrub=None):
    """
    Build the result dict. Its keys match the SandboxResult model in
    guardrails.py (ok, output, error, memories_loaded), so the caller can do
    SandboxResult(**result). Do not add other keys without updating that model.

    If a scrub function is given (for example the guardrails secret scrubber),
    it runs BEFORE truncation, so a secret cannot be cut in half and leak.
    """
    if scrub is not None:
        output = scrub(output)
        error = scrub(error)
    return {
        "ok": ok,
        "output": _truncate(output),
        "error": _truncate(error),
        "memories_loaded": loaded,
    }


def _new_process_group():
    """Start the child in its own process group, so a timeout can kill it with
    every process it started."""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _kill_tree(proc):
    """Kill the child and everything it started, then reap it."""
    if os.name == "nt":
        # taskkill /T also kills grandchild processes; capture its own output
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)    # the child leads its own group (start_new_session)
        except ProcessLookupError:
            pass                                    # it already exited
    try:
        proc.communicate(timeout=5)                 # reap it and close the pipes
    except subprocess.TimeoutExpired:
        proc.kill()


def run_code(code, memories=None, timeout=DEFAULT_TIMEOUT, scrub=None):
    """
    Run model-written Python in a separate, restricted child process.

    memories: list of JSON-serializable records (dicts). They reach the code as
        the variable `memories`. The memory server's run_memory_code tool passes
        simple records (id, time, kind, text, decisions, bugs, constraints, file).
        This module does not import pydantic or guardrails.
    scrub: optional function(str) -> str applied to the output and error text.

    Returns {"ok", "output", "error", "memories_loaded"}.

    Besides check_code, the child runs with no environment, in an empty temporary
    folder, under resource limits (CPU, memory, file size, no new processes; set by
    sandbox_runner.py on Linux/macOS), and the whole process tree is killed when it
    runs past `timeout`.
    """
    memories = memories or []
    loaded = len(memories)

    problem = check_code(code)
    if problem:
        return _result(False, "", f"Blocked: {problem}", loaded, scrub)

    try:
        payload = json.dumps({"code": code, "memories": memories, "cpu_seconds": int(timeout) + 1})
    except (TypeError, ValueError) as e:
        return _result(False, "", f"memories must be JSON-serializable: {e}", 0, scrub)

    size = len(payload.encode("utf-8"))
    if size > MAX_PAYLOAD_BYTES:
        return _result(
            False, "",
            f"Too much data for the sandbox ({size} bytes, limit {MAX_PAYLOAD_BYTES}). Pass fewer memories.",
            0, scrub,
        )

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        proc = subprocess.Popen(
                # -X utf8 makes the child read and print UTF-8 on any OS, so
                # non-English text in memories cannot crash it on Windows
                # (-I would ignore the PYTHONIOENCODING variable).
            [sys.executable, "-I", "-X", "utf8", RUNNER],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=tmp,
            env=_child_env(),
            **_new_process_group(),
        )
        try:
            stdout, stderr = proc.communicate(
                input=payload.encode("utf-8"),
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            return _result(False, "", f"Timed out after {timeout} seconds", loaded, scrub)
        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")
    return _result(proc.returncode == 0, stdout_text, stderr_text, loaded, scrub)