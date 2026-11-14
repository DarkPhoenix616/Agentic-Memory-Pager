"""
integrations/claude_code/store.py - page out: Claude Code conversation -> AMP vault
===================================================================================
    python -m integrations.claude_code.store --project DIR --transcript FILE [--mode all|stop]

Reads what is new in a conversation file (transcript.read_new) and stores it
in the project's AMP session, the same way AMP Chat pages out:

    text   consecutive turns are grouped up to TEXT_CHUNK_TOKENS (one RLM
           piece, so one sub-LLM call); a longer turn is split into parts.
           Each chunk -> RLM summarizer -> TextMemorySchema (summary,
           decisions, bugs, constraints). Exact names a summary tends to
           lose (`backticked` terms, files Claude changed) are added to the
           summary word for word.
    code   every code file Claude wrote, edited or read in full (its content
           after the change), and longer code blocks in its answers ->
           AST compactor -> CodeMemorySchema (imports, signatures,
           docstrings, call graph). A file whose content is unchanged since
           it was last stored is skipped.

Every memory has its secrets scrubbed first, is risk scored (Layer 2) and
signed (Layer 3) before the memory server stores it, encrypted with the
project's key. The memory server scores it again and checks it against older
memories for contradictions (fact poisoning).

Progress is saved after every chunk, so a stopped run continues where it
stopped and nothing is stored twice. Runs of one project never overlap.

    --mode all    store everything new (before a compaction, at the end of a session)
    --mode stop   after a reply: store only once at least STOP_MIN_TOKENS are new,
                  and skip if another run is busy (the next one catches up)
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

if __package__ in (None, ""):              # run as a file: make the repository importable
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from integrations.claude_code import project as proj
from integrations.claude_code.transcript import Turn, read_new

TEXT_CHUNK_TOKENS = 1800         # just under one RLM piece (2,000 tokens)
GROUP_TOKENS = 300               # turns shorter than this join the next one
STOP_MIN_TOKENS = int(os.environ.get("AMP_CC_STOP_MIN_TOKENS", "3000"))
_EXTENSIONS = {"python": "py", "javascript": "js", "typescript": "ts", "java": "java", "c": "c", "cpp": "cpp"}


def count_tokens(text: str) -> int:
    from compactor.ast_code import count_tokens as count
    return count(text)


@dataclass
class Chunk:
    text: str
    tokens: int
    end: int                                  # transcript offset once this chunk is stored
    started: str = ""
    finished: str = ""
    files: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    snippets: List[Tuple[str, str]] = field(default_factory=list)
    turns: List[str] = field(default_factory=list)      # fingerprints of the turns it completes


def turn_print(turn: Turn) -> str:
    """Fingerprint of a turn's text: the same turn in a resumed or forked
    conversation file (Claude Code copies the history) is not stored again."""
    return hashlib.sha256(turn.text().encode("utf-8")).hexdigest()[:16]


def _split(text: str, limit: int) -> List[str]:
    """Pieces of at most ~`limit` tokens, cut at line ends where possible."""
    pieces: List[str] = []
    current: List[str] = []
    used = 0
    for line in text.splitlines(keepends=True):
        tokens = count_tokens(line)
        if tokens > limit:                    # one huge line: cut it by characters
            step = max(200, limit * 3)
            parts = [line[i:i + step] for i in range(0, len(line), step)]
        else:
            parts = [line]
        for part in parts:
            size = count_tokens(part) if len(parts) > 1 else tokens
            if current and used + size > limit:
                pieces.append("".join(current))
                current, used = [], 0
            current.append(part)
            used += size
    if current:
        pieces.append("".join(current))
    return [p for p in pieces if p.strip()]


def plan_chunks(turns: List[Turn], start: int = 0, limit: int = TEXT_CHUNK_TOKENS,
                group_min: int = GROUP_TOKENS) -> List[Chunk]:
    """One chunk per turn, like AMP Chat's one chunk per exchange: a memory
    whose summary covers one topic is found by that topic. Short turns ("yes",
    "proceed") join the next turn until a chunk has `group_min` tokens; a turn
    longer than `limit` is split into parts. `start` is the transcript offset
    the turns were read from. A chunk's `end` only moves past a turn with that
    turn's last part, so a stopped run redoes the whole turn."""
    chunks: List[Chunk] = []
    group: List[Turn] = []
    group_tokens = 0
    previous_end = start

    def flush() -> None:
        nonlocal group, group_tokens
        if group:
            chunk = Chunk(text="\n\n---\n\n".join(t.text() for t in group if t.text()), tokens=group_tokens,
                          end=group[-1].end, started=group[0].started, finished=group[-1].finished)
            for turn in group:
                chunk.files.update(turn.files)
                chunk.snippets.extend(turn.snippets)
                chunk.turns.append(turn_print(turn))
            chunks.append(chunk)
        group, group_tokens = [], 0

    for turn in turns:
        text = turn.text()
        tokens = count_tokens(text) if text else 0
        if tokens > limit:
            flush()
            parts = _split(text, limit)
            for n, part in enumerate(parts, 1):
                last = n == len(parts)
                label = f"[Part {n} of {len(parts)} of one conversation turn]\n"
                chunk = Chunk(text=label + part, tokens=count_tokens(part), end=turn.end if last else previous_end,
                              started=turn.started, finished=turn.finished)
                if last:
                    chunk.files.update(turn.files)
                    chunk.snippets.extend(turn.snippets)
                    chunk.turns.append(turn_print(turn))
                chunks.append(chunk)
        else:
            if group and (group_tokens >= group_min or group_tokens + tokens > limit):
                flush()
            group.append(turn)
            group_tokens += tokens
        previous_end = turn.end
    flush()
    return chunks


# ---------------------------------------------------------------------------
# One memory (the compactors and the store; tests replace the first two)
# ---------------------------------------------------------------------------

_TICKED = re.compile(r"`([^`\n]{2,80})`")
_CHANGED = re.compile(r"\b(?:edited|wrote) ([^\s;]+)")
MAX_NAMES = 15


def exact_names(text: str) -> List[str]:
    """Names a summary tends to lose: `backticked` terms (branch names, functions,
    commands, settings) and the files Claude changed, in order, each once."""
    names: List[str] = []
    for name in [m.group(1).strip() for m in _TICKED.finditer(text)] + _CHANGED.findall(text):
        if name and name not in names and sum(len(n) + 4 for n in names) + len(name) <= 400:
            names.append(name)
    return names[:MAX_NAMES]


def _summarize(text: str, tokens: int, timeframe: str):
    from compactor.rlm_text import summarize_text
    return summarize_text(text, role="user", original_token_count=tokens, timeframe=timeframe or None)


def _compact_code(content: str, path: str, language: str, tokens: int):
    from compactor.ast_code import compact_code
    return compact_code(content, file_path=path, language=language, original_token_count=tokens)


def _seal_and_store(server, session: str, key: str, kind: str, payload, tokens: int):
    from harness.guardrails import UnifiedMemory, score_payload
    from harness.signing import sign_memory
    provenance = "codebase" if kind == "code" else "user_input"
    risk, _ = score_payload(payload, provenance=provenance)                     # Layer 2
    memory = UnifiedMemory(session_id=session, memory_type=kind, compaction_source="ast" if kind == "code" else "rlm",
                           code_payload=payload if kind == "code" else None,
                           text_payload=payload if kind == "text" else None,
                           raw_token_count=max(tokens, 1), risk_score=risk, provenance=provenance)
    memory.signature = sign_memory(memory)                                       # Layer 3: sign last
    return server.store_memory(memory, session_key=key)


def _store_chunk(server, session: str, key: str, chunk: Chunk, state: dict) -> Dict[str, int]:
    from harness.guardrails import _scrub_secrets
    done = {"text": 0, "code": 0, "raw_tokens": 0}
    if chunk.text.strip():
        timeframe = f"{chunk.started[:16].replace('T', ' ')} to {chunk.finished[:16].replace('T', ' ')} UTC" \
            if chunk.started else ""
        text = _scrub_secrets(chunk.text)
        schema = _summarize(text, chunk.tokens, timeframe)
        names = exact_names(text)
        if names:                              # kept word for word next to the summary
            schema.factual_summary = (schema.factual_summary.rstrip()
                                      + "\nNames: " + ", ".join(f"`{n}`" for n in names))[:8000]
        _seal_and_store(server, session, key, "text", schema, chunk.tokens)
        done["text"] += 1
        done["raw_tokens"] += chunk.tokens
    code = [(path, language, content) for path, (language, content) in chunk.files.items()]
    code += [(f"claude-reply/{hashlib.sha256(snippet.encode()).hexdigest()[:10]}.{_EXTENSIONS.get(language, 'txt')}",
              language, snippet) for language, snippet in chunk.snippets]
    for path, language, content in code:
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if state["files"].get(path) == digest:
            continue                           # stored before, unchanged
        tokens = count_tokens(content)
        schema = _compact_code(_scrub_secrets(content), path, language, tokens)
        state["files"][path] = digest
        if not schema.symbols:
            continue                           # nothing to find it by (an empty or docstring-only file)
        _seal_and_store(server, session, key, "code", schema, tokens)
        done["code"] += 1
        done["raw_tokens"] += tokens
    return done


def page_out(project: str, transcript: str, mode: str = "all") -> Dict[str, int]:
    """Store what is new in `transcript` for the project. Returns what was done:
    text/code memories stored, raw tokens covered, and `waiting` (stop mode:
    new tokens below the threshold, nothing stored yet) or `busy` (another run
    holds the project)."""
    folder = proj.project_dir(project)
    session = proj.session_id(folder)
    transcript = os.path.realpath(transcript)
    result = {"text": 0, "code": 0, "raw_tokens": 0}
    with proj.project_lock(session, wait=mode != "stop") as locked:
        if not locked:
            result["busy"] = 1
            return result
        state = proj.load_state(session)
        offset = int(state["transcripts"].get(transcript, 0))
        if os.path.exists(transcript) and offset > os.path.getsize(transcript):
            offset = 0                                  # the file was replaced
        turns, end = read_new(transcript, offset, folder)
        seen = set(state.setdefault("turn_prints", []))
        chunks = plan_chunks([t for t in turns if turn_print(t) not in seen], offset)
        new_tokens = sum(c.tokens for c in chunks) + sum(count_tokens(content) for c in chunks
                                                         for _, content in c.files.values())
        if mode == "stop" and new_tokens < STOP_MIN_TOKENS:
            result["waiting"] = new_tokens
            return result
        if chunks:
            server = proj.memory_server()
            key = proj.session_key(session)
            for chunk in chunks:
                done = _store_chunk(server, session, key, chunk, state)
                for name, value in done.items():
                    result[name] += value
                    if name in ("text", "code"):
                        state["stored"][name] = state["stored"].get(name, 0) + value
                state["stored"]["raw_tokens"] = state["stored"].get("raw_tokens", 0) + done["raw_tokens"]
                state["turn_prints"].extend(chunk.turns)
                state["transcripts"][transcript] = chunk.end
                proj.save_state(session, state)        # progress after every chunk
        state["transcripts"][transcript] = end
        state["project"] = folder
        state["last_saved"] = time.strftime("%Y-%m-%d %H:%M:%S")
        proj.save_state(session, state)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Store the new part of a Claude Code conversation in AMP.")
    parser.add_argument("--project", required=True)
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--mode", choices=("all", "stop"), default="all")
    args = parser.parse_args(argv)
    if proj.disabled():
        return 0
    started = time.time()
    try:
        proj.setup_env()
        result = page_out(args.project, args.transcript, args.mode)
    except Exception as err:                    # logged; the next run retries from the saved progress
        proj.log(f"page-out ({args.mode}) failed for {args.project}: {type(err).__name__}: {err}")
        return 1
    if result.get("text") or result.get("code"):
        proj.log(f"page-out ({args.mode}) {args.project}: {result['text']} text + {result['code']} code memories "
                 f"from {result['raw_tokens']} tokens in {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
