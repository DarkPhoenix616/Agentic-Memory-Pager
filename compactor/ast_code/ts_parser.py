"""
compactor/ast_code/ts_parser.py: Tree-sitter extraction for JavaScript, TypeScript, Java, C and C++
===================================================================================================
Python files go through Python's own `ast` (code_parser.py). Other languages
used to get only a line scan; with Tree-sitter (`tree-sitter` +
`tree-sitter-language-pack`) they get a real syntax tree:

    imports        import / import ... from / #include, with their modules as dependencies
    symbols        classes, interfaces, structs, enums, functions, methods and
                   arrow functions assigned to a const, with their signature
                   (the header up to the body), the comment just above them as
                   the docstring, and their line number
    calls          the code graph, as for Python: which function or method
                   calls which (this.f() and an implicit call to a method of
                   the same class are written as Class.f)

Function bodies are not stored; they are only read for calls. Everything goes
through code_parser's _Visitor.add_symbol, so the same name checks, secret
scrubbing and size limits apply. If Tree-sitter is not installed, or it fails
on a file, code_parser falls back to the line scan.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

LANGUAGE_BY_EXT = {".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
                   ".ts": "typescript", ".tsx": "typescript", ".java": "java", ".c": "c", ".h": "c",
                   ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp"}
ALIASES = {"js": "javascript", "ts": "typescript", "c++": "cpp", "cxx": "cpp"}

_FUNCTIONS = {"function_declaration", "generator_function_declaration", "method_definition",
              "method_declaration", "constructor_declaration", "function_definition"}
_CLASSES = {"class_declaration", "abstract_class_declaration", "interface_declaration", "enum_declaration",
            "class_specifier", "struct_specifier"}
_CALLS = {"call_expression", "method_invocation"}
_IMPORTS = {"import_statement", "import_declaration", "preproc_include"}
_NAME_NODES = {"identifier", "field_identifier", "qualified_identifier", "type_identifier", "property_identifier"}
_DOTTED = re.compile(r"[A-Za-z_][\w.]*")


def grammar_for(language: str, file_path: str) -> Optional[str]:
    """The Tree-sitter grammar to use, or None for a language it does not cover."""
    if language == "typescript" and file_path.endswith(".tsx"):
        return "tsx"
    return language if language in {"javascript", "typescript", "java", "c", "cpp"} else None


def available(grammar: str) -> bool:
    try:
        from tree_sitter_language_pack import get_parser
        get_parser(grammar)
        return True
    except Exception:
        return False


def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _name(node, src: bytes) -> Optional[str]:
    """A definition's name; for C/C++ the identifier inside its declarator ("Cart::total" -> "Cart.total")."""
    found = node.child_by_field_name("name")
    if found is not None:
        return _text(found, src)
    decl = node.child_by_field_name("declarator")
    while decl is not None:
        if decl.type in _NAME_NODES:
            return _text(decl, src).replace("::", ".")
        decl = decl.child_by_field_name("declarator") or decl.child_by_field_name("name")
    return None


def _header(node, body, src: bytes) -> str:
    """The definition's text up to its body, on one line."""
    end = body.start_byte if body is not None else node.end_byte
    return " ".join(src[node.start_byte:end].decode("utf-8", "replace").split())


def _comment_above(node, src: bytes) -> Optional[str]:
    """The comment that ends on the line just above the definition, without its markers."""
    target = node.parent if node.parent is not None and node.parent.type == "export_statement" else node
    prev = target.prev_named_sibling
    if prev is None or prev.type not in ("comment", "block_comment", "line_comment") \
            or target.start_point[0] - prev.end_point[0] > 1:
        return None
    lines = [re.sub(r"^\s*(/\*\*?|\*/|\*|//+)\s?", "", line) for line in _text(prev, src).splitlines()]
    text = "\n".join(line.replace("*/", "").rstrip() for line in lines).strip()
    return text or None


def _callee(call, src: bytes, cls: Optional[str], methods: Dict[str, set]) -> Optional[str]:
    """The name a call refers to: f, obj.f, Class.f for this.f() / an implicit method call."""
    if call.type == "method_invocation":                      # Java: obj.name(...) or name(...)
        name, obj = call.child_by_field_name("name"), call.child_by_field_name("object")
        if name is None:
            return None
        name = _text(name, src)
        if obj is None or _text(obj, src) == "this":
            return f"{cls}.{name}" if cls and name in methods.get(cls, ()) else name
        text = f"{_text(obj, src)}.{name}"
    else:
        function = call.child_by_field_name("function")
        if function is None:
            return None
        text = _text(function, src).replace("::", ".").replace("->", ".")
        if text.startswith("this."):
            text = f"{cls}.{text[5:]}" if cls else text[5:]
        elif cls and text in methods.get(cls, ()):
            text = f"{cls}.{text}"
    return text if _DOTTED.fullmatch(text) else None


def _descendants(node):
    stack = list(reversed(node.named_children))
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.named_children))


def _add_import(node, src: bytes, out) -> None:
    out.imports.append(" ".join(_text(node, src).split())[:256])
    if node.type == "preproc_include":                        # #include <stdio.h> / "cart.h"
        path = node.child_by_field_name("path")
        module = _text(path, src).strip('<>"') if path is not None else ""
    elif node.type == "import_declaration":                   # Java: import java.util.List;
        module = re.sub(r"^import\s+(static\s+)?|;$", "", _text(node, src)).strip()
        out.imported.add(module.rsplit(".", 1)[-1])
    else:                                                     # JS/TS: import { a, b as c } from "x";
        source = node.child_by_field_name("source")
        module = _text(source, src).strip("'\"`") if source is not None else ""
        for child in _descendants(node):
            if child.type == "identifier":
                out.imported.add(_text(child, src))
    module = module[:128]
    if module and module not in out.dependencies:
        out.dependencies.append(module)


def extract(text: str, grammar: str, visitor) -> None:
    """Fill code_parser's _Visitor with the imports, symbols and calls of `text`."""
    from tree_sitter_language_pack import get_parser
    src = text.encode("utf-8")
    root = get_parser(grammar).parse(src).root_node
    found: List[tuple] = []                                   # (node, header node, body, name, kind, class)
    methods: Dict[str, set] = {}

    def walk(node, cls: Optional[str]) -> None:
        for child in node.named_children:
            kind = child.type
            if kind in _IMPORTS:
                _add_import(child, src, visitor.out)
            elif kind in _CLASSES and child.child_by_field_name("body") is not None:
                name = _name(child, src)
                if name:
                    found.append((child, child, child.child_by_field_name("body"), name, "class", None))
                    walk(child.child_by_field_name("body"), name)
            elif kind in _FUNCTIONS:
                name = _name(child, src)
                if not name:
                    continue
                full = name if "." in name else (f"{cls}.{name}" if cls else name)
                owner = full.rsplit(".", 1)[0] if "." in full else None
                if owner:
                    methods.setdefault(owner, set()).add(full.rsplit(".", 1)[1])
                found.append((child, child, child.child_by_field_name("body"), full,
                               "method" if owner else "function", owner))
            elif kind in ("lexical_declaration", "variable_declaration") and cls is None:
                for decl in child.named_children:                 # const total = (cart) => ...
                    value = decl.child_by_field_name("value") if decl.type == "variable_declarator" else None
                    if value is not None and value.type in ("arrow_function", "function_expression", "function"):
                        name = decl.child_by_field_name("name")
                        if name is not None:
                            found.append((value, child, value.child_by_field_name("body"),
                                          _text(name, src), "function", None))
            else:
                walk(child, cls)                              # export, namespace, class bodies, ...

    walk(root, None)
    for node, head, body, name, kind, cls in found:
        visitor.add_symbol(name.replace("::", "."), kind, _header(head, body, src),
                           _comment_above(head, src), head.start_point[0] + 1)
        if kind != "class" and body is not None:
            calls = [c for c in (_callee(n, src, cls, methods) for n in _descendants(body) if n.type in _CALLS) if c]
            visitor.out.raw_calls[name.replace("::", ".")] = calls
