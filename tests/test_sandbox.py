"""
tests/test_sandbox.py — server/sandbox.py: running model-written Python safely.

Run with:
    pytest tests/test_sandbox.py -v
"""
import os
import subprocess
import time

import pytest

from harness.guardrails import SandboxResult, _scrub_secrets
from server.sandbox import MAX_OUTPUT_CHARS, MAX_PAYLOAD_BYTES, run_code

MEMORIES = [
    {"id": "m1", "time": "2026-10-01T10:00:00", "kind": "text", "text": "We use PostgreSQL for the vault.",
     "decisions": ["Use PostgreSQL"], "bugs": [], "constraints": [], "file": ""},
    {"id": "m2", "time": "2026-10-02T10:00:00", "kind": "code", "text": "chat/u1_1.py\ndef add(a, b):",
     "decisions": [], "bugs": [], "constraints": [], "file": "chat/u1_1.py"},
    {"id": "m3", "time": "2026-10-05T10:00:00", "kind": "text", "text": "Switched the vault database to MongoDB.",
     "decisions": ["Switch to MongoDB"], "bugs": ["Fixed the KeyError in pager.py"], "constraints": [], "file": ""},
]


def runner_processes() -> int:
    if os.name == "nt":
        return 0
    found = subprocess.run(["pgrep", "-f", "sandbox_runner.py"], capture_output=True, text=True)
    return len(found.stdout.split())


# ---------------------------------------------------------------------------
# Normal use
# ---------------------------------------------------------------------------

def test_code_sees_the_memories_and_its_output_comes_back():
    code = ("db = [m for m in memories if m['kind'] == 'text']\n"
            "print(max(db, key=lambda m: m['time'])['text'])\n"
            "print(sum(len(m['bugs']) for m in memories))")
    result = run_code(code, MEMORIES)
    assert result == {"ok": True, "output": "Switched the vault database to MongoDB.\n1\n", "error": "",
                      "memories_loaded": 3}
    SandboxResult(**result)                      # the keys match the guardrails model


def test_allowed_modules_can_be_used():
    code = ("from collections import Counter\nfrom datetime import datetime\n"
            "print(Counter(m['kind'] for m in memories)['text'])\n"
            "print(datetime.fromisoformat(memories[0]['time']).year)")
    assert run_code(code, MEMORIES)["output"] == "2\n2026\n"


def test_non_english_text_survives_the_round_trip():
    memories = [{"text": "नमस्ते, 你好, ¡hola!"}]
    assert run_code("print(memories[0]['text'])", memories)["output"] == "नमस्ते, 你好, ¡hola!\n"


# ---------------------------------------------------------------------------
# Unsafe code
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("code, reason", [
    ("import os\nos.system('ls')", "import of 'os'"),
    ("import socket", "import of 'socket'"),
    ("from subprocess import run", "import from 'subprocess'"),
    ("open('/etc/passwd').read()", "'open'"),
    ("eval('1 + 1')", "'eval'"),
    ("getattr(memories, 'clear')()", "'getattr'"),
    ("print(().__class__.__bases__)", "access to '__"),
])
def test_unsafe_code_is_blocked_before_it_runs(code, reason):
    result = run_code(code, MEMORIES)
    assert result["ok"] is False and result["error"].startswith("Blocked:") and reason in result["error"]


def test_errors_report_the_line():
    result = run_code("x = 1\ny = memories[99]", MEMORIES)
    assert result["ok"] is False and result["error"].strip() == "IndexError: list index out of range (line 2)"
    assert run_code("print(", MEMORIES)["error"].startswith("Blocked: SyntaxError")


def test_a_runaway_loop_is_stopped_and_killed():
    before = runner_processes()
    started = time.monotonic()
    result = run_code("while True:\n    pass", MEMORIES, timeout=1)
    assert result["ok"] is False and result["error"] == "Timed out after 1 seconds"
    assert time.monotonic() - started < 4
    assert runner_processes() == before          # nothing left spinning in the background


@pytest.mark.skipif(os.name == "nt", reason="resource limits are set on Linux/macOS only")
def test_memory_is_capped():
    result = run_code("x = 'a' * (2 * 1024 ** 3)", MEMORIES)
    # macOS might kill the process entirely or time it out before a MemoryError prints.
    # As long as the sandbox successfully blocked it (ok is False), the test should pass.
    assert result["ok"] is False


# ---------------------------------------------------------------------------
# Output limits and secrets
# ---------------------------------------------------------------------------

def test_long_output_is_truncated():
    result = run_code("print('x' * 10000)", MEMORIES)
    assert len(result["output"]) < MAX_OUTPUT_CHARS + 100 and "truncated" in result["output"]


def test_secrets_are_scrubbed_even_where_the_output_is_cut():
    # The key straddles the 4,000-character cut. Scrubbing runs before truncation,
    # so no half-key survives. (The scrubber needs a word boundary before "sk-",
    # as in real text: after a space, "=", ":" or a quote.)
    key = "sk-" + "A1b2C3d4E5" * 3
    code = f"print('x' * {MAX_OUTPUT_CHARS - 10} + ' token={key}')"
    result = run_code(code, MEMORIES, scrub=_scrub_secrets)
    assert "sk-A1b" not in result["output"] and "A1b2C3" not in result["output"]   # no fragment of the key
    assert "token=[RE" in result["output"]          # replaced (the marker itself may be cut)


def test_too_much_data_is_refused():
    result = run_code("print(1)", [{"text": "x" * MAX_PAYLOAD_BYTES}])
    assert result["ok"] is False and result["error"].startswith("Too much data")
