"""
integrations/claude_code/transcript.py - reading Claude Code's conversation file
================================================================================
Claude Code writes every conversation to a JSONL file (one JSON record per
line, only ever appended) and passes its path to each hook as
`transcript_path`. read_new(path, offset) reads what was added after `offset`
bytes and groups it into turns:

    Turn.prompt    what the user typed (system reminders, IDE context and slash
                   command echoes removed)
    Turn.replies   Claude's answers (thinking is not kept)
    Turn.actions   one line per tool call: "edited app/server.py", "ran: Run the tests"
    Turn.files     code files of the project that Claude wrote, edited or read in
                   full, with their content after the change: these become AST
                   code memories (files outside the project folder are left out)
    Turn.snippets  longer code blocks in Claude's answers (AST as well)

Tool output (terminal output, search results, web pages) is not kept: it is
large, mostly noise, and the files it came from are still on disk. Records of
subagents (isSidechain), injected context (isMeta) and Claude Code's own
compaction summaries (isCompactSummary) are skipped.

Only complete lines are read, so a line Claude Code is still writing is read
next time. The format is Claude Code's own and may change, so every field is
read defensively: an unknown record is skipped, never an error.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

MAX_ACTIONS = 40            # tool-call lines kept per turn
MAX_FILE_CHARS = 200_000    # bigger files are not stored as code memories
MIN_SNIPPET_LINES = 5       # shorter code blocks in answers stay in the text

_REMOVE = re.compile(
    r"<(system-reminder|ide_selection|ide_opened_file|ide_diagnostics|local-command-caveat|"
    r"local-command-stdout|local-command-stderr|command-name|command-message|command-args)\b[^>]*>"
    r".*?</\1>", re.DOTALL)
_PASTE_TAGS = re.compile(r"</?pasted_content\b[^>]*>")
_FENCE = re.compile(r"```[ \t]*([\w+#.-]*)[^\n]*\n(.*?)(?:```|\Z)", re.DOTALL)
_FENCE_LANGUAGES = {"py": "python", "python": "python", "python3": "python", "js": "javascript",
                    "javascript": "javascript", "jsx": "javascript", "ts": "typescript",
                    "typescript": "typescript", "tsx": "typescript", "java": "java", "c": "c",
                    "cpp": "cpp", "c++": "cpp", "cc": "cpp"}
_INTERRUPTED = "[Request interrupted by user"


def code_language(path: str) -> str:
    """The AST compactor's language for a file, or "" if it does not handle it."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".py":
        return "python"
    try:
        from compactor.ast_code.ts_parser import LANGUAGE_BY_EXT
    except Exception:                       # tree-sitter missing: Python files only
        return ""
    return LANGUAGE_BY_EXT.get(ext, "")


@dataclass
class Turn:
    prompt: str = ""
    replies: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    files: Dict[str, Tuple[str, str]] = field(default_factory=dict)    # path -> (language, content)
    snippets: List[Tuple[str, str]] = field(default_factory=list)      # (language, code)
    started: str = ""                       # timestamp of its first record
    finished: str = ""                      # timestamp of its last record
    end: int = 0                            # byte offset just after its last line

    def text(self) -> str:
        """The turn as one text for the RLM summarizer."""
        parts = []
        if self.prompt:
            parts.append(f"User: {self.prompt}")
        reply = "\n\n".join(r for r in self.replies if r.strip())
        if reply:
            parts.append(f"Assistant: {reply}")
        if self.actions:
            parts.append("Actions: " + "; ".join(self.actions))
        return "\n\n".join(parts)

    def empty(self) -> bool:
        return not (self.prompt or self.text() or self.files or self.snippets)


def clean_prompt(text: str) -> str:
    """What the user typed, without the context Claude Code adds around it."""
    text = _PASTE_TAGS.sub("", _REMOVE.sub("", text or ""))
    lines = [line for line in text.splitlines() if not line.strip().startswith(_INTERRUPTED)]
    return "\n".join(lines).strip()


def _relative(path: str, project: str) -> str:
    if project and path.startswith(project.rstrip(os.sep) + os.sep):
        return os.path.relpath(path, project)
    return path


def _action(name: str, args: Dict[str, Any], project: str) -> str:
    path = _relative(str(args.get("file_path") or args.get("notebook_path") or ""), project)
    if name == "Write":
        return f"wrote {path}"
    if name in ("Edit", "MultiEdit", "NotebookEdit"):
        return f"edited {path}"
    if name == "Read":
        return f"read {path}"
    if name == "Bash":
        return "ran: " + str(args.get("description") or args.get("command") or "")[:120]
    if name in ("Grep", "Glob"):
        return "searched the code for " + str(args.get("pattern") or "")[:80]
    if name == "WebSearch":
        return "searched the web for " + str(args.get("query") or "")[:80]
    if name == "WebFetch":
        return "read the web page " + str(args.get("url") or "")[:120]
    if name.startswith("mcp__amp-memory__") or name in ("TodoWrite", "ToolSearch"):
        return ""                            # AMP's own recall tool, bookkeeping
    return f"used {name}"


def _split_reply(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Longer code blocks in a supported language become snippets; the text keeps a marker."""
    snippets: List[Tuple[str, str]] = []

    def take(match: "re.Match[str]") -> str:
        language = _FENCE_LANGUAGES.get(match.group(1).strip().lower(), "")
        code = match.group(2)
        if language and code.count("\n") + 1 >= MIN_SNIPPET_LINES:
            snippets.append((language, code))
            return f"[code block: {language}]"
        return match.group(0)

    return _FENCE.sub(take, text), snippets


def _file_after(result: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """(path, full content) when a tool result holds a whole file: a file Claude
    wrote, a file Claude edited (its content after the edit), or a file read in full."""
    path = result.get("filePath")
    if isinstance(path, str) and isinstance(result.get("content"), str) and result.get("type") in ("create", "update"):
        return path, result["content"]
    if isinstance(path, str) and isinstance(result.get("originalFile"), str) \
            and isinstance(result.get("oldString"), str) and isinstance(result.get("newString"), str):
        old, new, original = result["oldString"], result["newString"], result["originalFile"]
        if old and old in original:
            return path, original.replace(old, new) if result.get("replaceAll") else original.replace(old, new, 1)
        return None
    info = result.get("file")
    if result.get("type") == "text" and isinstance(info, dict) and isinstance(info.get("content"), str):
        whole = info.get("startLine", 1) == 1 and info.get("numLines", 0) >= info.get("totalLines", 0)
        if whole and isinstance(info.get("filePath"), str):
            return info["filePath"], info["content"]
    return None


def _human_text(record: Dict[str, Any]) -> str:
    content = (record.get("message") or {}).get("content")
    if isinstance(content, str):
        return clean_prompt(content)
    if isinstance(content, list):
        texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        return clean_prompt("\n".join(texts))
    return ""


def _skipped(record: Dict[str, Any]) -> bool:
    return bool(record.get("isSidechain") or record.get("isMeta") or record.get("isCompactSummary")
                or record.get("isVisibleInTranscriptOnly"))


def read_new(path: str, offset: int = 0, project: str = "") -> Tuple[List[Turn], int]:
    """The turns written after byte `offset`, and the offset after the last
    complete line. A file smaller than `offset` was replaced: it is read again
    from the start."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return [], offset
    if offset > size:
        offset = 0
    turns: List[Turn] = []
    current: Optional[Turn] = None
    position = offset
    with open(path, "rb") as handle:
        handle.seek(offset)
        for raw in handle:
            if not raw.endswith(b"\n"):
                break                           # still being written
            position += len(raw)
            try:
                record = json.loads(raw)
            except ValueError:
                continue
            if not isinstance(record, dict) or _skipped(record) or record.get("type") not in ("user", "assistant"):
                if current is not None:
                    current.end = position
                continue
            stamp = str(record.get("timestamp") or "")
            if record["type"] == "user":
                prompt = _human_text(record)
                if prompt:                      # a new turn starts
                    if current is not None and not current.empty():
                        turns.append(current)
                    current = Turn(prompt=prompt, started=stamp)
            if current is None:
                current = Turn(started=stamp)   # the rest of a turn begun before `offset`
            current.finished = stamp or current.finished
            current.end = position
            content = (record.get("message") or {}).get("content")
            if record["type"] == "assistant" and isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text" and block.get("text", "").strip():
                        text, snippets = _split_reply(block["text"])
                        current.replies.append(text.strip())
                        current.snippets.extend(snippets)
                    elif block.get("type") == "tool_use":
                        line = _action(str(block.get("name", "")), block.get("input") or {}, project)
                        if line and len(current.actions) < MAX_ACTIONS and line not in current.actions[-1:]:
                            current.actions.append(line)
            result = record.get("toolUseResult")
            if record["type"] == "user" and isinstance(result, dict):
                found = _file_after(result)
                if found:
                    file_path, text = found
                    language = code_language(file_path)
                    path = _relative(file_path, project)
                    outside = bool(project) and os.path.isabs(path)      # a library, a temp file: not this project
                    if language and not outside and len(text) <= MAX_FILE_CHARS and text.strip():
                        current.files[path] = (language, text)
    if current is not None and not current.empty():
        turns.append(current)
    return turns, position


def last_prompt(path: str, max_bytes: int = 2_000_000) -> str:
    """The latest thing the user typed in a conversation file ("" if none)."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as handle:
            handle.seek(max(0, size - max_bytes))
            lines = handle.read().splitlines()
    except OSError:
        return ""
    for raw in reversed(lines):
        try:
            record = json.loads(raw)
        except ValueError:
            continue
        if isinstance(record, dict) and record.get("type") == "user" and not _skipped(record):
            prompt = _human_text(record)
            if prompt:
                return prompt
    return ""
