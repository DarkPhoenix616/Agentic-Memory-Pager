"""
report_eval/claude_code/compact_recall.py - does a fact survive Claude Code's /compact?
=======================================================================================
    .venv/bin/python -I report_eval/claude_code/compact_recall.py [--model haiku] [--port 8012]

A real Claude Code session, with and without AMP:

  1. A demo project gets AMP (integrations/claude_code/install.py), with its own
     memory server on --port, a temporary vault, keyring and progress folder.
  2. Eight turns: each pastes a long document (deploy log, meeting notes,
     config, ...) and asks for its 3 key points. One small fact is buried in
     the middle of each document; the user never asks about it.
  3. /compact. The session is built and compacted as plain Claude Code (AMP
     hooks off), so nothing of AMP's can reach Claude Code's own summary or
     the compacted session that both conditions fork from. Then AMP pages the
     conversation file out, exactly as its PreCompact hook does (store.py on
     the same file). The first run had the hooks on during the build; their
     notes leaked into the session both conditions forked from
     (compact_recall_results_run1_contaminated.json).
  4. For each fact, a question in a fresh fork of the compacted session, twice:
        A  with AMP      (hooks page in, recall tool allowed; read-only, so
                          one question's answer is never saved for the next)
        B  without AMP   (AMP_CC_DISABLE=1, recall tool blocked)
     File, shell and web tools are blocked in both, so the answer can only
     come from what the session remembers.

Also measured: which facts are still in Claude Code's own compaction summary,
and which ones AMP's page-in puts into the context for the question.

Needs the claude CLI (CLAUDE_BIN, or `claude` on PATH, or the VS Code
extension's binary), a Claude login, and Ollama with llama3.1:8b for the RLM.
Uses Claude usage: about 26 short calls with the chosen model.
"""
from __future__ import annotations

import argparse
import fcntl
import glob
import json
import os
import random
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
PYTHON = os.path.join(REPO, ".venv", "bin", "python")
OUT = os.path.dirname(os.path.abspath(__file__))
NO_TOOLS = ["Bash", "Read", "Grep", "Glob", "WebFetch", "WebSearch", "Task", "Agent", "Edit", "Write"]
RECALL = "mcp__amp-memory__recall"

R = random.Random(7)
PEOPLE = ["Arjun", "Priya", "Kiran", "Sana", "Rahul", "Divya", "Vikram", "Anita", "Rohan", "Isha"]
SERVICES = ["auth", "orders", "search", "inventory", "notifications", "reports", "catalog", "shipping"]
THINGS = ["dashboard", "retry policy", "cache layer", "alert rules", "load test", "backup job", "rate limiter",
          "audit trail", "release notes", "onboarding guide", "error budget", "queue consumer"]
VERBS = ["review", "update", "finish", "draft", "test", "clean up", "document", "benchmark"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Friday"]


def num(lo, hi):
    while True:
        n = R.randint(lo, hi)
        if n not in (25, 7421, 77, 30):
            return n


def ci_log(n):
    steps = ["checkout", "install deps", "lint", "unit tests", "build image", "push image", "migrate", "smoke test"]
    return [f"2026-11-{num(10, 12)} 0{num(1, 5)}:{num(10, 59)}:{num(10, 59)} [{R.choice(SERVICES)}] step "
            f"{R.randint(1, 8)}/8 {R.choice(steps)}: {R.choice(['ok', 'ok', 'ok', 'retrying', 'WARN slow', 'FAILED'])} "
            f"({num(2, 300)}s)" for _ in range(n)]


def meeting(n):
    return [f"- {R.choice(PEOPLE)} will {R.choice(VERBS)} the {R.choice(SERVICES)} {R.choice(THINGS)} by "
            f"{R.choice(DAYS)}; {R.choice(['blocked on review', 'on track', 'needs a second pair of eyes', 'done soon'])}."
            for _ in range(n)]


def incident(n):
    return [f"{num(10, 23)}:{num(10, 59)} {R.choice(PEOPLE)} {R.choice(['checked', 'restarted', 'scaled', 'paused', 'rolled back'])} "
            f"the {R.choice(SERVICES)} {R.choice(THINGS)}; error rate {num(1, 9)}.{num(10, 99)}%." for _ in range(n)]


def config(n):
    return [f"  {R.choice(SERVICES)}_{R.choice(['timeout_ms', 'workers', 'retries', 'batch_size', 'log_level', 'pool'])}"
            f"_{i}: {R.choice([str(num(2, 900)), 'true', 'false', 'info', 'debug', 'warn'])}" for i in range(n)]


def emails(n):
    return [f"{R.choice(PEOPLE)}: {R.choice(['Thanks,', 'Quick update:', 'Following up:', 'FYI,'])} the "
            f"{R.choice(SERVICES)} {R.choice(THINGS)} {R.choice(['looks fine now', 'needs another pass', 'is scheduled', 'moved to next sprint'])}."
            for _ in range(n)]


def api_spec(n):
    return [f"GET /v1/{R.choice(SERVICES)}/{R.choice(['items', 'events', 'jobs', 'users', 'stats'])}/{{id}} - returns one "
            f"record; rate limit {num(40, 900)} requests per minute; cached for {num(2, 60)} s." for _ in range(n)]


def review(n):
    return [f"- {R.choice(SERVICES)}/{R.choice(['handlers', 'models', 'client', 'utils'])}.py:{num(10, 400)} "
            f"{R.choice(['rename this variable', 'add a test for the empty case', 'this can be a list comprehension', 'log the error here', 'nit: spacing'])}"
            for _ in range(n)]


def migration(n):
    return [f"Step {i + 1}: {R.choice(['copy', 'backfill', 'index', 'verify', 'lock', 'unlock'])} table "
            f"{R.choice(SERVICES)}_{R.choice(['events', 'items', 'audit', 'archive'])} in batches of {num(100, 900)} rows."
            for i in range(n)]


#        title, filler, buried fact, question, what the answer must contain
FACTS = [
    ("last night's deploy log", ci_log,
     "2026-11-11 03:12:44 [gateway] api-gateway is listening on port 7421 (set in deploy/api.env)",
     "Which port does our API gateway listen on?", [r"7421"]),
    ("the notes from Monday's planning meeting", meeting,
     "- Agreed: no deploys on Thursdays after 4 pm, because the weekly billing run starts then.",
     "On which day, and from what time, are deploys not allowed?", [r"thursday", r"\b4\b|16:00|4\s*pm|four"]),
    ("the incident report for the refund failures", incident,
     "Root cause: the flaky test `test_refund_rounding` failed because refund_amount() used float rounding; "
     "the fix is to use Decimal.",
     "Which test was flaky in the refund incident, and what was the fix?", [r"test_refund_rounding", r"decimal"]),
    ("the current service config", config,
     "  checkout_v2_canary: true   # the new checkout, shown to 5% of users",
     "What is the name of the feature flag for the new checkout?", [r"checkout_v2_canary"]),
    ("the email thread with the platform team", emails,
     "Neha: For anything about the production database, page Meera - she is on call until the end of the month.",
     "Who should I page for a production database problem?", [r"meera"]),
    ("the draft API spec", api_spec,
     "POST /v1/uploads - uploads larger than 25 MB are rejected with HTTP 413.",
     "What is the maximum upload size in our API?", [r"\b25\s*MB|\b25\b"]),
    ("the code review comments on the payments PR", review,
     "- payments/client.py:88 every idempotency key must start with the prefix PAY-77 so the ledger can tell retries apart",
     "What prefix must payment idempotency keys start with?", [r"PAY-77"]),
    ("the migration plan for the orders tables", migration,
     "Note: the staging database is called orchid_stage; never point the migration at prod_main.",
     "What is the staging database called?", [r"orchid_stage"]),
]
TASK = "Reply with only the 3 most important points, one short line each."
LINES = 230                      # ~3,500 tokens per document


def documents():
    docs = []
    for title, filler, fact, _, _ in FACTS:
        lines = filler(LINES)
        lines.insert(int(len(lines) * 0.45), fact)
        docs.append(f"Here is {title}. {TASK}\n\n" + "\n".join(lines))
    return docs


def matches(text, patterns):
    return all(re.search(p, text or "", re.IGNORECASE) for p in patterns)


def correct(answer, patterns):
    """Strict: the fact is there and the answer does not start by saying it doesn't know."""
    refused = re.match(r"\W*i (don't|do not) know", answer or "", re.IGNORECASE)
    return matches(answer, patterns) and not refused


def claude_binary():
    found = os.environ.get("CLAUDE_BIN") or shutil.which("claude")
    if not found:
        candidates = sorted(glob.glob(os.path.expanduser(
            "~/.vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude")))
        found = candidates[-1] if candidates else ""
    if not found:
        raise SystemExit("No claude CLI found: set CLAUDE_BIN.")
    return found


def transcript_of(session_id):
    found = glob.glob(os.path.expanduser(f"~/.claude/projects/*/{session_id}.jsonl"))
    return found[0] if found else ""


def page_out_done(state_dir, session, transcript):
    """No page-out running and the whole conversation file saved."""
    lock = os.path.join(state_dir, f".{session}.lock")
    if os.path.exists(lock):
        with open(lock, "a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            fcntl.flock(handle, fcntl.LOCK_UN)
    try:
        with open(os.path.join(state_dir, f"{session}.json")) as handle:
            saved = json.load(handle)["transcripts"].get(os.path.realpath(transcript), 0)
    except (FileNotFoundError, ValueError, KeyError):
        return False
    return saved >= os.path.getsize(transcript)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--model", default="haiku")
    parser.add_argument("--port", type=int, default=8012)
    args = parser.parse_args()
    claude = claude_binary()
    tmp = tempfile.mkdtemp(prefix="amp-cc-eval-")
    project = os.path.join(tmp, "shop")
    os.makedirs(project)
    with open(os.path.join(project, "shop.py"), "w") as handle:
        handle.write('"""A tiny shop."""\n\n\ndef cart_total(items):\n    return sum(i["price"] * i["qty"] for i in items)\n')
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    url = f"http://localhost:{args.port}/mcp"
    signing = secrets.token_hex(32)
    amp_env = {"AMP_CC_MCP_URL": url, "AMP_CC_KEYS_FILE": os.path.join(tmp, "cc_keys"),
               "AMP_CC_STATE_DIR": os.path.join(tmp, "state"), "AMP_SIGNING_KEY": signing}
    server_env = {k: v for k, v in os.environ.items() if not k.startswith("AMP_")}
    server_env.update(AMP_CHROMA_PATH=os.path.join(tmp, "vault"), AMP_MCP_PORT=str(args.port),
                      AMP_AUDIT_LOG=os.path.join(tmp, "audit.log"), AMP_SIGNING_KEY=signing,
                      AMP_ADMIN_TOKEN=secrets.token_hex(16), AMP_SESSION_KEYS_FILE=os.path.join(tmp, "server_keys"))
    server = subprocess.Popen([PYTHON, "server/amp_memory_server.py"], cwd=REPO, env=server_env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    results = {"model": args.model, "folder": tmp, "facts": []}
    cost = 0.0

    def run(prompt, *extra, env=None):
        nonlocal cost
        clean = {"HOME": os.environ["HOME"], "PATH": "/usr/local/bin:/usr/bin:/bin", "USER": os.environ.get("USER", ""),
                 "LANG": os.environ.get("LANG", "C.UTF-8"), "TERM": "dumb", **amp_env, **(env or {})}
        done = subprocess.run([claude, "-p", prompt, "--model", args.model, "--output-format", "json", *extra],
                              cwd=project, env=clean, capture_output=True, text=True, timeout=900)
        try:
            out = json.loads(done.stdout)
        except ValueError:
            raise SystemExit(f"claude failed: {done.stdout[-500:]} {done.stderr[-500:]}")
        cost += float(out.get("total_cost_usd") or 0)
        return out

    try:
        for _ in range(60):
            try:
                import socket
                socket.create_connection(("localhost", args.port), timeout=1).close()
                break
            except OSError:
                time.sleep(1)
        from integrations.claude_code import install
        from integrations.claude_code import project as proj
        install.install(project, mcp_url=url)
        session = proj.session_id(project)

        # 1-2: build the session (plain Claude Code)
        plain = {"AMP_CC_DISABLE": "1"}
        started = time.time()
        sid = ""
        for n, doc in enumerate(documents(), 1):
            out = run(doc, *(["--resume", sid] if sid else []), env=plain)
            sid = out["session_id"]
            print(f"turn {n}: {out.get('result', '')[:90]!r}", flush=True)
        results["session_id"] = sid
        results["build_seconds"] = round(time.time() - started, 1)

        # 3: compact (plain Claude Code), then AMP pages the conversation out as its PreCompact hook does
        out = run("/compact", "--resume", sid, env=plain)
        sid = out["session_id"]
        path = transcript_of(sid)
        results["amp_text_in_session"] = "<amp_memory" in open(path).read()
        started = time.time()
        subprocess.run([PYTHON, "-m", "integrations.claude_code.store", "--project", project, "--transcript", path,
                        "--mode", "all"], cwd=REPO, env={**os.environ, **amp_env}, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        results["page_out_seconds"] = round(time.time() - started, 1)
        records = [json.loads(line) for line in open(path) if line.strip()]
        boundary = [r for r in records if r.get("type") == "system" and r.get("subtype") == "compact_boundary"]
        summary = next((r["message"]["content"] for r in reversed(records) if r.get("isCompactSummary")), "")
        if isinstance(summary, list):
            summary = " ".join(b.get("text", "") for b in summary if isinstance(b, dict))
        results["compaction"] = boundary[-1].get("compactMetadata", {}) if boundary else {}
        print(f"compacted: {results['compaction'].get('preTokens')} -> {results['compaction'].get('postTokens')} tokens",
              flush=True)

        if not page_out_done(amp_env["AMP_CC_STATE_DIR"], session, path):
            raise SystemExit("AMP page-out did not save the whole conversation")
        with open(os.path.join(amp_env["AMP_CC_STATE_DIR"], f"{session}.json")) as handle:
            results["amp_stored"] = json.load(handle)["stored"]
        print(f"AMP page-out took {results['page_out_seconds']}s: {results['amp_stored']}; "
              f"AMP text in the compacted session: {results['amp_text_in_session']}", flush=True)

        # 4: ask
        os.environ.update(amp_env)
        proj.setup_env()
        from integrations.claude_code.page_in import page_in
        for title, _, fact, question, patterns in FACTS:
            ask = (f"{question} Answer in one short sentence from what you know of this project. "
                   "If you don't know, say \"I don't know\".")
            paged, count, tokens = page_in(project, ask)
            with_amp = run(ask, "--resume", sid, "--fork-session", "--disallowedTools", *NO_TOOLS,
                           "--allowedTools", RECALL, env={"AMP_CC_READ_ONLY": "1"})
            fork = transcript_of(with_amp["session_id"])
            used_recall = bool(fork) and RECALL in open(fork).read()
            without = run(ask, "--resume", sid, "--fork-session", "--disallowedTools", *NO_TOOLS, RECALL,
                          env={"AMP_CC_DISABLE": "1"})
            row = {"document": title, "fact": fact, "question": question,
                   "in_claude_summary": matches(summary, patterns),
                   "in_amp_page_in": matches(paged, patterns), "page_in_notes": count, "page_in_tokens": tokens,
                   "answer_with_amp": with_amp.get("result", ""), "correct_with_amp": correct(with_amp.get("result"), patterns),
                   "mentions_with_amp": matches(with_amp.get("result"), patterns),
                   "recall_tool_used": used_recall,
                   "answer_without_amp": without.get("result", ""),
                   "correct_without_amp": correct(without.get("result"), patterns),
                   "mentions_without_amp": matches(without.get("result"), patterns)}
            results["facts"].append(row)
            print(f"{'OK ' if row['correct_with_amp'] else 'NO '} with AMP | {'OK ' if row['correct_without_amp'] else 'NO '}"
                  f" without | {question}", flush=True)
    finally:
        server.send_signal(signal.SIGINT)
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
    facts = results["facts"]
    results["score"] = {"with_amp": sum(f["correct_with_amp"] for f in facts),
                        "without_amp": sum(f["correct_without_amp"] for f in facts),
                        "mentions_with_amp": sum(f["mentions_with_amp"] for f in facts),
                        "mentions_without_amp": sum(f["mentions_without_amp"] for f in facts),
                        "in_claude_summary": sum(f["in_claude_summary"] for f in facts),
                        "in_amp_page_in": sum(f["in_amp_page_in"] for f in facts), "of": len(facts)}
    results["claude_cost_usd"] = round(cost, 4)
    with open(os.path.join(OUT, "compact_recall_results.json"), "w") as handle:
        json.dump(results, handle, indent=1)
    print(json.dumps(results["score"]), f"cost ${cost:.3f}")


if __name__ == "__main__":
    main()
