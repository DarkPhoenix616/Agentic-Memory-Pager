#A sandbox will help you test your tools and memory functions without affecting the main memory database and prevents any python injection attack possibilities.
import os
import subprocess
import sys
import tempfile
import ast
import json

RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox_runner.py")

DEFAULT_TIMEOUT = 5  # seconds


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

def run_code(code, memories=None, timeout=DEFAULT_TIMEOUT):
    problem = check_code(code)
    if problem:
        return {"ok": False, "output": "", "error": f"Blocked: {problem}"}

    payload = json.dumps({"code": code, "memories": memories or []})

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", RUNNER],
                input=payload,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmp,
                env=_child_env(),
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "output": "",
                    "error": f"Timed out after {timeout} seconds"}

    return {
        "ok": proc.returncode == 0,
        "output": _truncate(proc.stdout),
        "error": _truncate(proc.stderr),
    }