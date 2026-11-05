"""
compactor/ast_code/code_parser.py — AST Code Compactor  [Teammate 2]

Turns a chunk of Python source code into a CodeMemorySchema: the structural
skeleton of the file (imports, classes, function and method signatures,
docstrings, module-level variables) without the function bodies.

Why
    When the context window fills up, the harness pages the oldest code chunk
    out to long-term memory. Storing the skeleton instead of the raw text keeps
    what an agent needs to recall a file's API (names, signatures, types,
    docstrings, where things are) at a fraction of the tokens.

Public API
    compact_code(source, file_path, language="python", original_token_count=None,
                 docstrings="full") -> CodeMemorySchema
    render_skeleton(schema) -> str     plain-text view of a stored skeleton
    skeleton_token_count(schema) -> int
    count_tokens(text) -> int          same tokenizer as harness/graph.py

Robustness
    Context chunks are often fragments, not whole files. Parsing is tried in
    this order and the first strategy that works is used:
      1. the source as given
      2. the source dedented          (e.g. a method copied out of a class)
      3. trailing lines removed       (e.g. a chunk cut off mid-function);
                                      the removed lines are then line-scanned
      4. a line-based scan for def / class / import lines
    compact_code() never raises on bad input. The summary says which strategy
    was used, and every value is kept inside the limits of CodeMemorySchema.

Security
    Signatures, docstrings and summaries pass through the shared secret
    scrubber of harness/guardrails.py. Values of module-level variables are
    kept only when they are short literals, and never for names that look like
    credentials (API_KEY, PASSWORD, TOKEN, ...).
"""
from __future__ import annotations

import ast
import copy
import re
import textwrap
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional, Tuple

from harness.guardrails import CodeMemorySchema, CodeSymbol, _scrub_secrets

# ---------------------------------------------------------------------------
# Limits — mirror the Field constraints of CodeSymbol / CodeMemorySchema in
# harness/guardrails.py, so the compactor never builds a payload they reject.
# ---------------------------------------------------------------------------
MAX_SYMBOLS = 500
MAX_IMPORTS = 200
MAX_DEPENDENCIES = 100
MAX_NAME = 128
MAX_SIGNATURE = 512
MAX_DOCSTRING = 4000
MAX_IMPORT_LEN = 256
MAX_DEPENDENCY_LEN = 128
MAX_SUMMARY = 2000
MAX_PATH = 260
MAX_LANGUAGE = 32
MAX_TOKENS = 2_000_000

MAX_VALUE_PREVIEW = 80   # longest literal kept for a module-level variable
MAX_TRIM_LINES = 200     # strategy 3 stops after removing this many lines

_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_.]*$")
_LANGUAGE_RE = re.compile(r"^[a-z0-9_-]+$")
# Words that mark a variable as a credential (compared as whole words, so
# MAX_TOKENS or SORT_KEY are not hidden, but ACCESS_TOKEN or apiKey are).
_SECRET_WORDS = {"secret", "token", "password", "passwd", "pass", "pwd",
                 "credential", "credentials", "apikey"}
_KEY_PREFIXES = {"api", "private", "secret", "access", "signing", "encryption", "auth"}
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

# Line-scan patterns (strategy 4, and languages the AST parser does not handle)
_DEF_LINE_RE = re.compile(r"^(?P<indent>\s*)(?:async\s+)?def\s+(?P<name>[A-Za-z_]\w*)\s*\(")
_JS_FUNC_LINE_RE = re.compile(r"^(?P<indent>\s*)(?:export\s+)?(?:async\s+)?function\s+(?P<name>[A-Za-z_$][\w$]*)\s*\(")
_CLASS_LINE_RE = re.compile(r"^(?P<indent>\s*)(?:export\s+)?class\s+(?P<name>[A-Za-z_]\w*)")
_IMPORT_LINE_RE = re.compile(r"^\s*(?:import\s+\S|from\s+\S+\s+import\s+\S)")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _encoder():
    """cl100k_base encoder, or None if tiktoken is missing or cannot load it
    (it downloads its vocabulary file on first use)."""
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str) -> int:
    """Token count with the same tokenizer as harness/graph.py (cl100k_base).
    Falls back to ~4 characters per token if the tokenizer is unavailable."""
    if not text:
        return 0
    encoder = _encoder()
    if encoder is None:
        return max(1, len(text) // 4)
    # disallowed_special=(): code may contain marker strings like '<|endoftext|>',
    # which tiktoken otherwise rejects with an error.
    return len(encoder.encode(text, disallowed_special=()))


def _clip(text: str, limit: int) -> str:
    """Cut text to at most `limit` characters, marking the cut with '...'."""
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _safe_relative_path(path: Optional[str]) -> str:
    """Turn any path into the relative form CodeMemorySchema accepts:
    no drive letter, no leading slash, no '..' components."""
    p = (path or "").replace("\\", "/").strip()
    if len(p) >= 2 and p[1] == ":":
        p = p[2:]
    parts = [part for part in p.split("/") if part not in ("", ".", "..")]
    safe = "/".join(parts)
    if len(safe) > MAX_PATH:
        safe = safe[-MAX_PATH:].lstrip("/")
    return safe or "unknown/file.py"


def _normalise_language(language: Optional[str], file_path: str) -> str:
    lang = (language or "").strip().lower()
    if lang in ("py", "python3", "python2") or (not lang and file_path.endswith(".py")):
        lang = "python"
    if not lang or len(lang) > MAX_LANGUAGE or not _LANGUAGE_RE.match(lang):
        lang = "unknown"
    return lang


DOCSTRING_MODES = ("full", "first_paragraph")


def _looks_like_credential(name: str) -> bool:
    words = [w.lower() for w in re.split(r"[^A-Za-z0-9]+", _CAMEL_BOUNDARY_RE.sub("_", name)) if w]
    if any(word in _SECRET_WORDS for word in words):
        return True
    return any(
        word == "key" and (len(words) == 1 or (i > 0 and words[i - 1] in _KEY_PREFIXES))
        for i, word in enumerate(words)
    )


def _clean_docstring(doc: Optional[str], mode: str = "full") -> Optional[str]:
    """Scrub and bound a docstring. mode='first_paragraph' keeps only the text
    up to the first blank line (much smaller for heavily documented files)."""
    if not doc:
        return None
    doc = doc.strip()
    if mode == "first_paragraph":
        doc = re.split(r"\n\s*\n", doc, maxsplit=1)[0]
    return _clip(_scrub_secrets(doc), MAX_DOCSTRING) or None


# ---------------------------------------------------------------------------
# Parsing (strategies 1-3)
# ---------------------------------------------------------------------------

def _try_parse(text: str) -> Optional[ast.Module]:
    try:
        return ast.parse(text)
    except (SyntaxError, ValueError):   # ValueError: source contains null bytes
        return None


def _close_open_block(text: str) -> str:
    """If text ends with a block header ('def f(x):'), give it a body so the
    header itself can still be parsed and kept."""
    lines = text.rstrip().splitlines()
    if not lines or not lines[-1].rstrip().endswith(":"):
        return text
    indent = len(lines[-1]) - len(lines[-1].lstrip())
    return text.rstrip() + "\n" + " " * (indent + 4) + "..."


@dataclass
class _Parsed:
    tree: Optional[ast.Module]  # None if every AST strategy failed
    text: str                   # the text that was parsed (line numbers unchanged)
    strategy: str
    tail: str = ""              # trailing lines removed by strategy 3
    tail_first_line: int = 0    # line number of the first tail line


def _parse(source: str) -> _Parsed:
    tree = _try_parse(source)
    if tree is not None:
        return _Parsed(tree, source, "full")

    text = textwrap.dedent(source)
    if text != source:
        tree = _try_parse(text)
        if tree is not None:
            return _Parsed(tree, text, "dedented")

    lines = text.splitlines()
    for cut in range(1, min(len(lines), MAX_TRIM_LINES) + 1):
        kept = lines[:-cut]
        candidate = "\n".join(kept)
        if not candidate.strip():
            break
        for attempt in (candidate, _close_open_block(candidate)):
            tree = _try_parse(attempt)
            if tree is not None:
                return _Parsed(tree, attempt, f"trimmed {cut} trailing line(s)",
                               tail="\n".join(lines[-cut:]), tail_first_line=len(kept) + 1)

    return _Parsed(None, text, "line scan")


def _open_classes_at(tree: ast.Module, line: int) -> List[Tuple[int, str]]:
    """Classes still open at `line` (outermost first), as (indent, name).
    Used to put methods found in a trimmed tail under the right class."""
    open_classes: List[Tuple[int, str]] = []
    body = tree.body
    while True:
        enclosing = [n for n in body if isinstance(n, ast.ClassDef)
                     and (n.end_lineno or n.lineno) >= line]
        if not enclosing:
            return open_classes
        node = enclosing[-1]
        open_classes.append((node.col_offset, node.name))
        body = node.body


# ---------------------------------------------------------------------------
# Extraction from the AST
# ---------------------------------------------------------------------------

@dataclass
class _Extracted:
    imports: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    symbols: List[CodeSymbol] = field(default_factory=list)
    skipped: int = 0            # symbols dropped because their name is not ASCII-safe
    module_doc: Optional[str] = None


def _header(node: ast.AST) -> str:
    """'def f(x: int = 1) -> str:' or 'class A(B):' — the first line of the
    node, without its body. Decorators are added on separate lines."""
    stub = copy.copy(node)
    stub.body = [ast.Expr(ast.Constant(Ellipsis))]
    stub.decorator_list = []
    header = ast.unparse(stub).splitlines()[0]
    decorators = ["@" + ast.unparse(d) for d in getattr(node, "decorator_list", [])]
    return "\n".join(decorators + [header])


def _variable_signature(name: str, node: ast.AST) -> str:
    annotation = ""
    if isinstance(node, ast.AnnAssign):
        annotation = ": " + ast.unparse(node.annotation)
    value = getattr(node, "value", None)
    if value is None:
        return f"{name}{annotation}"
    if _looks_like_credential(name):
        return f"{name}{annotation} = ..."          # never store credential values
    shown = ast.unparse(value)
    simple = isinstance(value, (ast.Constant, ast.Name, ast.Attribute)) or (
        isinstance(value, (ast.Tuple, ast.List, ast.Set, ast.Dict))
        and all(isinstance(n, (ast.Constant, ast.Name)) for n in ast.walk(value)
                if isinstance(n, ast.expr) and n is not value)
    )
    if not simple or len(shown) > MAX_VALUE_PREVIEW:
        shown = "..."
    return f"{name}{annotation} = {shown}"


class _Visitor:
    """Walks module-level and class-level statements. Function bodies are not
    entered: their contents are implementation detail, not structure."""

    def __init__(self, text: str, docstrings: str = "full"):
        self.text = text
        self.docstrings = docstrings
        self.out = _Extracted()

    def add_symbol(self, name: str, kind: str, signature: str,
                   docstring: Optional[str], line: int) -> None:
        if len(name) > MAX_NAME or not _IDENTIFIER_RE.match(name):
            self.out.skipped += 1
            return
        signature = _clip(_scrub_secrets(signature), MAX_SIGNATURE)
        if len(signature) < 3:
            signature = (signature + " ...")[:MAX_SIGNATURE]
        self.out.symbols.append(CodeSymbol(
            name=name, type=kind, signature=signature,
            docstring=_clean_docstring(docstring, self.docstrings), start_line=max(1, line),
        ))

    def add_import(self, node: ast.AST) -> None:
        segment = ast.get_source_segment(self.text, node) or ast.unparse(node)
        self.out.imports.append(_clip(" ".join(segment.split()), MAX_IMPORT_LEN))
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        else:
            modules = ["." * node.level + (node.module or "")]
        for module in modules:
            module = _clip(module, MAX_DEPENDENCY_LEN)
            if module and module not in self.out.dependencies:
                self.out.dependencies.append(module)

    def visit_block(self, statements: List[ast.stmt], prefix: str = "", in_class: bool = False) -> None:
        for node in statements:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self.add_import(node)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.add_symbol(prefix + node.name, "method" if in_class else "function",
                                _header(node), ast.get_docstring(node), node.lineno)
            elif isinstance(node, ast.ClassDef):
                self.add_symbol(prefix + node.name, "class", _header(node),
                                ast.get_docstring(node), node.lineno)
                self.visit_block(node.body, prefix=f"{prefix}{node.name}.", in_class=True)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)) and not in_class:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        self.add_symbol(prefix + target.id, "variable",
                                        _variable_signature(target.id, node), None, node.lineno)
            elif isinstance(node, ast.If):
                # e.g. 'if TYPE_CHECKING:' imports or platform-specific definitions
                self.visit_block(node.body, prefix, in_class)
                self.visit_block(node.orelse, prefix, in_class)
            elif isinstance(node, (ast.Try, getattr(ast, "TryStar", ast.Try))):
                # e.g. 'try: import ujson as json / except ImportError: import json'
                self.visit_block(node.body, prefix, in_class)
                for handler in node.handlers:
                    self.visit_block(handler.body, prefix, in_class)
                self.visit_block(node.orelse, prefix, in_class)
                self.visit_block(node.finalbody, prefix, in_class)


# ---------------------------------------------------------------------------
# Extraction by line scan (strategy 4)
# ---------------------------------------------------------------------------

def _extract_from_lines(text: str, first_line: int = 1,
                        open_classes: Optional[List[Tuple[int, str]]] = None,
                        visitor: Optional[_Visitor] = None) -> _Extracted:
    visitor = visitor or _Visitor(text)
    out = visitor.out
    classes: List[Tuple[int, str]] = list(open_classes or [])   # (indent, name) of enclosing classes
    for number, line in enumerate(text.splitlines(), start=first_line):
        stripped = line.strip()
        if not stripped:
            continue
        if _IMPORT_LINE_RE.match(line):
            out.imports.append(_clip(stripped, MAX_IMPORT_LEN))
            module = stripped.split()[1].rstrip(",;")
            module = _clip(module, MAX_DEPENDENCY_LEN)
            if module and module not in out.dependencies:
                out.dependencies.append(module)
            continue
        match = _CLASS_LINE_RE.match(line) or _DEF_LINE_RE.match(line) or _JS_FUNC_LINE_RE.match(line)
        if not match:
            continue
        indent = len(match.group("indent"))
        while classes and classes[-1][0] >= indent:
            classes.pop()
        prefix = "".join(name + "." for _, name in classes)
        signature = stripped.rstrip("{").rstrip()
        if match.re is _CLASS_LINE_RE:
            visitor.add_symbol(prefix + match.group("name"), "class", signature, None, number)
            classes.append((indent, match.group("name")))
        else:
            kind = "method" if classes else "function"
            visitor.add_symbol(prefix + match.group("name"), kind, signature, None, number)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _summary(file_path: str, language: str, out: _Extracted, strategy: str, total_symbols: int) -> str:
    counts = {kind: 0 for kind in ("class", "function", "method", "variable")}
    for symbol in out.symbols:
        counts[symbol.type] += 1
    parts = []
    if out.module_doc:
        first_line = out.module_doc.strip().splitlines()[0]
        parts.append(_clip(_scrub_secrets(first_line), 300))
    parts.append(
        f"{language} file {file_path}: {counts['class']} classes, {counts['function']} functions, "
        f"{counts['method']} methods, {counts['variable']} variables, {len(out.imports)} imports."
    )
    if strategy != "full":
        parts.append(f"Partial parse ({strategy}).")
    if total_symbols > len(out.symbols):
        parts.append(f"Only the first {len(out.symbols)} of {total_symbols} symbols are kept.")
    if out.skipped:
        parts.append(f"{out.skipped} symbol(s) with non-ASCII names were skipped.")
    return _clip(" ".join(parts), MAX_SUMMARY)


def compact_code(
    source: str,
    file_path: str = "unknown/file.py",
    language: str = "python",
    original_token_count: Optional[int] = None,
    docstrings: str = "full",
) -> CodeMemorySchema:
    """Compact a source chunk into its structural skeleton.

    Args:
        source: the raw code chunk (a whole file or a fragment).
        file_path: path of the file the chunk comes from; made relative and safe.
        language: 'python' gets the full AST parser; other languages get the line scan.
        original_token_count: token count of `source`, if the caller already has it.
        docstrings: 'full' keeps docstrings verbatim (the CodeSymbol contract);
            'first_paragraph' keeps only their first paragraph for more compression.

    Returns:
        A CodeMemorySchema that always passes the schema's validation.
    """
    if docstrings not in DOCSTRING_MODES:
        raise ValueError(f"docstrings must be one of {DOCSTRING_MODES}, not {docstrings!r}")
    source = source or ""
    safe_path = _safe_relative_path(file_path)
    lang = _normalise_language(language, safe_path)
    if original_token_count is None:
        original_token_count = count_tokens(source)
    original_token_count = min(max(0, int(original_token_count)), MAX_TOKENS)

    if lang == "python":
        parsed = _parse(source)
    else:
        parsed = _Parsed(None, source, "line scan, no AST parser for this language")
    strategy = parsed.strategy

    if parsed.tree is None:
        out = _extract_from_lines(parsed.text)
        if strategy == "line scan":
            strategy = "line scan, the code could not be parsed"
    else:
        visitor = _Visitor(parsed.text, docstrings)
        visitor.out.module_doc = ast.get_docstring(parsed.tree)
        visitor.visit_block(parsed.tree.body)
        if parsed.tail:
            # Strategy 3 dropped the last lines: still pick up the definitions in them.
            _extract_from_lines(parsed.tail, first_line=parsed.tail_first_line,
                                open_classes=_open_classes_at(parsed.tree, parsed.tail_first_line - 1),
                                visitor=visitor)
            strategy += " + line scan of the rest"
        out = visitor.out

    out.symbols.sort(key=lambda s: s.start_line)
    total_symbols = len(out.symbols)
    out.symbols = out.symbols[:MAX_SYMBOLS]
    out.imports = out.imports[:MAX_IMPORTS]
    out.dependencies = out.dependencies[:MAX_DEPENDENCIES]

    return CodeMemorySchema(
        file_path=safe_path,
        language=lang,
        imports=out.imports,
        symbols=out.symbols,
        dependencies=out.dependencies,
        summary=_summary(safe_path, lang, out, strategy, total_symbols),
        original_token_count=original_token_count,
    )


def render_skeleton(schema: CodeMemorySchema) -> str:
    """Plain-text view of a compacted file: what the agent reads when the
    memory is paged back in. Methods are indented under their class."""
    lines = [f"# {schema.file_path} ({schema.language})"]
    if schema.summary:
        lines.append(f"# {schema.summary}")
    lines.extend(schema.imports)
    for symbol in schema.symbols:
        depth = symbol.name.count(".")
        pad = "    " * depth
        signature_lines = symbol.signature.splitlines()
        lines.extend(pad + s for s in signature_lines)
        if symbol.docstring:
            doc = textwrap.indent(f'"""{symbol.docstring}"""', pad + "    ")
            lines.append(doc)
        if symbol.type in ("function", "method"):
            lines.append(pad + "    ...")
    return "\n".join(lines)


def skeleton_token_count(schema: CodeMemorySchema) -> int:
    """Tokens of the rendered skeleton — compare with schema.original_token_count."""
    return count_tokens(render_skeleton(schema))
