"""
integrations/claude_code/hooks.py - the hooks Claude Code runs
==============================================================
    python integrations/claude_code/hooks.py EVENT --project DIR     (hook input JSON on stdin)

    EVENT        Claude Code hook   what AMP does
    stop         Stop               start a page-out in the background (stop mode:
                                    only once ~3,000 tokens are new)
    precompact   PreCompact         start a page-out of everything new, before
                                    Claude Code squeezes the conversation
    sessionend   SessionEnd         start a page-out of everything new
    prompt       UserPromptSubmit   page in: add the memories matching the prompt
                                    to Claude's context
    start        SessionStart       after a compaction or a resume: page in for the
                                    last thing the user asked, and say AMP is on

The page-outs run as their own process (store.py), detached from Claude Code:
the hook returns at once, and the work finishes even if Claude Code exits.
Claude Code keeps the conversation file after a compaction, so nothing is
missed by not waiting.

A hook never blocks or breaks Claude Code: every error is logged to
storage/claude_code/amp_cc.log and the hook exits 0 with no output.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

if __package__ in (None, ""):              # run as a file: make the repository importable
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from integrations.claude_code import project as proj

PAGE_OUT_EVENTS = {"stop": "stop", "precompact": "all", "sessionend": "all"}
EVENTS = ("stop", "precompact", "sessionend", "prompt", "start")
RESUME_BUDGET = 700          # tokens of memories added after a compaction or a resume


def start_page_out(project: str, transcript: str, mode: str) -> None:
    """Run store.py in the background, in its own session so it outlives Claude Code."""
    folder = proj.state_dir()
    os.makedirs(folder, mode=0o700, exist_ok=True)
    with open(os.path.join(folder, "amp_cc.log"), "a") as log:
        subprocess.Popen([sys.executable, "-m", "integrations.claude_code.store", "--project", project,
                          "--transcript", transcript, "--mode", mode],
                         cwd=proj.REPO, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                         start_new_session=True, close_fds=True)


def _context(event_name: str, text: str, message: str = "") -> str:
    out = {"hookSpecificOutput": {"hookEventName": event_name, "additionalContext": text}}
    if message:
        out["systemMessage"] = message          # shown to the user
    return json.dumps(out)


def on_prompt(project: str, data: dict) -> str:
    from integrations.claude_code.page_in import page_in
    from integrations.claude_code.transcript import clean_prompt
    prompt = clean_prompt(str(data.get("prompt") or ""))
    if not prompt or prompt.startswith("/"):
        return ""
    text, count, tokens = page_in(project, prompt)
    if not text:
        return ""
    proj.log(f"page-in {project}: {count} notes, {tokens} tokens")
    return _context("UserPromptSubmit", text, f"AMP paged in {count} saved note{'s' * (count != 1)} ({tokens} tokens)")


def on_start(project: str, data: dict) -> str:
    from integrations.claude_code.page_in import page_in
    from integrations.claude_code.transcript import last_prompt
    state = proj.load_state(proj.session_id(project))
    saved = state["stored"].get("text", 0) + state["stored"].get("code", 0)
    if not saved:
        return ""
    note = (f"AMP long-term memory is on for this project: {saved} notes from earlier conversations are saved. "
            "Notes that match each message are added to it automatically, and the amp-memory recall tool "
            "searches them.")
    text = ""
    if data.get("source") in ("compact", "resume") and data.get("transcript_path"):
        query = last_prompt(str(data["transcript_path"]))
        if query:
            text, _, _ = page_in(project, query, budget=RESUME_BUDGET)
    return _context("SessionStart", note + ("\n\n" + text if text else ""))


def run(event: str, project: str, data: dict) -> str:
    """What the hook prints for one event ("" for nothing)."""
    if event in PAGE_OUT_EVENTS:
        transcript = str(data.get("transcript_path") or "")
        if transcript and os.path.exists(transcript) and not proj.read_only():
            start_page_out(project, transcript, PAGE_OUT_EVENTS[event])
        return ""
    proj.setup_env()
    if event == "prompt":
        return on_prompt(project, data)
    if event == "start":
        return on_start(project, data)
    return ""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="AMP hooks for Claude Code.")
    parser.add_argument("event", choices=EVENTS)
    parser.add_argument("--project", default="")
    args = parser.parse_args(argv)
    if proj.disabled():
        return 0
    try:
        data = json.load(sys.stdin)
    except ValueError:
        data = {}
    data = data if isinstance(data, dict) else {}
    project = args.project or os.environ.get("CLAUDE_PROJECT_DIR") or str(data.get("cwd") or os.getcwd())
    try:
        output = run(args.event, proj.project_dir(project), data)
    except Exception as err:                    # never break Claude Code
        proj.log(f"hook {args.event} failed: {type(err).__name__}: {err}")
        return 0
    if output:
        print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
