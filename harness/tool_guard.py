"""
harness/tool_guard.py — Layer 4 output monitoring (T4 — LangGraph Harness)

Sits between the LLM's reply and tool execution. The LLM asks for a tool with a
line like  TOOL: search_vault("query")  (see _build_secure_system_prompt). Before
anything runs, every requested call is checked here:

  1. Allowlist   Only search_vault and run_memory_code (read-only Python over
                 the memories, in the sandbox) may be requested. Mutating tools
                 (delete_memory, store_memory) and unknown tools are rejected.
  2. Origin      The LLM cannot say which memory prompted a call, so the call
                 text is compared with the memories in its context. A memory
                 that supplied the call's distinctive words is a suspected
                 source, and its execution_level (Layer 3) must be high enough
                 for the tool, otherwise the call is rejected.

This is what makes Layer 3's execution levels binding: without it they are only
text in the prompt. The module is pure Python (no LangGraph, no I/O) so it can be
unit-tested directly; graph.py does the wiring.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence

logger = logging.getLogger(__name__)

Decision = Literal["ALLOW", "REJECT"]

# Lowest to highest. A memory may influence a tool only if its level is at or
# above the level the tool requires.
EXECUTION_LEVELS = ("READ_ONLY", "DATA_ONLY", "SEARCH_ONLY")

# Tools the LLM may request, and the minimum execution_level of any memory that
# influenced the request.
TOOL_REQUIRED_LEVEL: Dict[str, str] = {"search_vault": "SEARCH_ONLY", "run_memory_code": "SEARCH_ONLY"}

CODE_TOOL = "run_memory_code"

# Never callable from model output, whatever the context says.
MUTATING_TOOLS = frozenset({"delete_memory", "store_memory"})


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolCall:
    """A tool request parsed from LLM output."""
    name: str
    raw_args: str = ""
    query: Optional[str] = None   # first argument without quotes, when it is a quoted string
    code: Optional[str] = None    # run_memory_code: the Python to run

    @property
    def text(self) -> str:
        """The text the call is judged on."""
        if self.code is not None:
            return self.code
        return self.query if self.query is not None else self.raw_args


@dataclass
class ValidationResult:
    decision: Decision
    tool: str
    reasons: List[str] = field(default_factory=list)            # why REJECT
    suspected_sources: List[str] = field(default_factory=list)  # memory ids


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# "TOOL: name(args)" on its own line. Parentheses are optional so that a bare
# "TOOL: delete_memory" is still seen (and then rejected).
_TOOL_LINE = re.compile(
    r"^[ \t]*TOOL:[ \t]*([A-Za-z_]\w*)[ \t]*(?:\((.*)\))?[ \t\r]*$", re.IGNORECASE | re.MULTILINE
)
_QUOTED_ARG = re.compile(r"""\s*(["'])(.*)\1\s*$""", re.DOTALL)
# run_memory_code takes its program as a fenced block right after the TOOL line:
#   TOOL: run_memory_code
#   ```python
#   print(len(memories))
#   ```
# (used with .match(text, pos), which anchors at pos; \A would mean the start of the whole text)
_CODE_BLOCK = re.compile(r"[ \t]*\r?\n\s*```[\w+-]*[^\n]*\n(.*?)(?:```|\Z)", re.DOTALL)


def parse_tool_calls(reply_text: str) -> List[ToolCall]:
    """Extract every TOOL: line from an LLM reply, in order."""
    calls: List[ToolCall] = []
    text = reply_text or ""
    for match in _TOOL_LINE.finditer(text):
        raw = (match.group(2) or "").strip()
        quoted = _QUOTED_ARG.match(raw)
        code = None
        if match.group(1).lower() == CODE_TOOL:
            block = _CODE_BLOCK.match(text, match.end())
            code = block.group(1) if block else (quoted.group(2) if quoted else raw)
        calls.append(ToolCall(
            name=match.group(1),
            raw_args=raw,
            query=quoted.group(2) if quoted else None,
            code=code,
        ))
    return calls


# ---------------------------------------------------------------------------
# Origin attribution
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"\w{4,}")
_STOPWORDS = frozenset({
    "that", "this", "with", "from", "have", "what", "about", "which", "would", "there",
    "their", "they", "been", "were", "will", "your", "into", "when", "where", "does",
    "show", "find", "tell", "please", "search", "memory", "memories", "vault", "previous",
    "earlier", "recall", "remember", "stored", "information", "details", "also", "than",
    "then", "them", "these", "those", "some", "more", "most", "other", "should", "could",
})


# Words any program over the memories uses: Python keywords and builtins and the
# memory record's field names. They are not evidence that a memory shaped the code.
_CODE_WORDS = frozenset({
    "print", "lambda", "return", "elif", "else", "while", "break", "continue", "pass", "import",
    "from", "true", "false", "none", "sorted", "reverse", "lower", "upper", "strip", "split",
    "join", "startswith", "endswith", "find", "replace", "count", "items", "keys", "values",
    "append", "extend", "list", "dict", "tuple", "range", "enumerate", "zip", "filter", "isinstance",
    "text", "time", "kind", "file", "decisions", "bugs", "constraints", "code", "latest", "result",
    "results", "item", "found", "matches", "match", "total", "counter", "collections", "datetime",
    "json", "math", "statistics", "itertools", "string", "fromisoformat", "group", "search",
})


def _content_tokens(text: str) -> set:
    return {t for t in _TOKEN.findall((text or "").lower()) if t not in _STOPWORDS}


def _level_rank(level: Any) -> int:
    """Unknown levels rank below READ_ONLY, so they can never pass a check."""
    return EXECUTION_LEVELS.index(level) if level in EXECUTION_LEVELS else -1


def _suspected_sources(call: ToolCall, context: Sequence[Dict[str, Any]], user_text: str) -> List[Dict[str, Any]]:
    """
    Memories in context that supplied the call's words.

    Words the user already said are not attributed to memory. For the remaining
    ("novel") words, a memory is a suspect if it contains at least two of them,
    or all of them. Conservative on purpose: a query built from words that only
    a stored memory contains is treated as influenced by that memory.
    """
    novel = _content_tokens(call.text) - _content_tokens(user_text)
    if call.name.lower() == CODE_TOOL:
        novel -= _CODE_WORDS
    if not novel:
        return []
    suspects = []
    for mem in context:
        shared = novel & _content_tokens(mem.get("text", ""))
        if shared and (len(shared) >= 2 or len(shared) == len(novel)):
            suspects.append(mem)
    return suspects


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_tool_call_origin(
    call: ToolCall,
    context: Sequence[Dict[str, Any]],
    user_text: str,
) -> ValidationResult:
    """
    Decide ALLOW or REJECT for one requested tool call.

    context: one dict per memory the LLM could see, as built by graph._context_entry:
             {"id", "level", "risk_score", "risk_level", "provenance", "verified", "text"}
    user_text: everything the user said (words from it are not attributed to memory).
    """
    reasons: List[str] = []
    sources = _suspected_sources(call, context, user_text)

    if call.name in MUTATING_TOOLS:
        reasons.append(f"mutating tool '{call.name}' can never be called from model output")
    elif call.name not in TOOL_REQUIRED_LEVEL:
        reasons.append(f"tool '{call.name}' is not on the allowlist")
    else:
        required = TOOL_REQUIRED_LEVEL[call.name]
        for mem in sources:
            if _level_rank(mem.get("level")) < _level_rank(required):
                reasons.append(
                    f"call appears derived from memory {mem.get('id')} "
                    f"(execution_level {mem.get('level')}); '{call.name}' needs {required}"
                )

    decision: Decision = "REJECT" if reasons else "ALLOW"
    if decision == "REJECT":
        logger.warning("[Layer4] REJECT tool=%s reasons=%s", call.name, reasons)
    return ValidationResult(
        decision=decision,
        tool=call.name,
        reasons=reasons,
        suspected_sources=[m.get("id") for m in sources],
    )


def validate_tool_calls(
    calls: Sequence[ToolCall],
    context: Sequence[Dict[str, Any]],
    user_text: str,
) -> List[ValidationResult]:
    """Validate every call from one reply."""
    return [validate_tool_call_origin(call, context, user_text) for call in calls]
