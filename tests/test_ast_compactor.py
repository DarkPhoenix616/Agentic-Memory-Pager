"""
tests/test_ast_compactor.py — Unit tests for compactor/ast_code/code_parser.py

Run with:
    pytest tests/test_ast_compactor.py -v
    pytest tests/test_ast_compactor.py -v -s     # also prints the compression table
"""
from pathlib import Path

import pytest

from compactor.ast_code import compact_code, count_tokens, render_skeleton, skeleton_token_count
from harness.guardrails import CodeMemorySchema

ROOT = Path(__file__).resolve().parent.parent

SAMPLE = '''"""Token budget helpers for the agent harness.

Longer description that only matters when reading the whole module.
"""
import os
from typing import List, Optional
from .storage import vault as memory_vault

MAX_TOKENS: int = 128_000
DEFAULT_MODEL = "llama3:8b"
API_KEY = "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789"

try:
    import ujson as json
except ImportError:
    import json


def count(text: str, *, model: Optional[str] = None) -> int:
    """Count tokens in text."""
    secret_body_marker = "THIS_IS_IN_THE_BODY"
    return len(text.split())


@staticmethod
async def fetch(url: str, timeout: float = 5.0) -> bytes:
    return b""


class Pager(object):
    """Pages memories in and out.

    Second paragraph with details.
    """

    LIMIT = 10

    def page_in(self, query: str, budget: int = 500) -> List[str]:
        """Fetch memories for a query."""
        return []

    class Config:
        strict = True

        def validate(self) -> bool:
            return True
'''


@pytest.fixture
def sample():
    return compact_code(SAMPLE, "harness/token_budget.py")


def symbol(schema, name):
    return next(s for s in schema.symbols if s.name == name)


# ---------------------------------------------------------------------------
# Structure extraction
# ---------------------------------------------------------------------------

class TestExtraction:

    def test_returns_valid_schema(self, sample):
        assert isinstance(sample, CodeMemorySchema)
        CodeMemorySchema.model_validate(sample.model_dump())   # round trip passes validation

    def test_imports_verbatim_including_try_block(self, sample):
        assert sample.imports == [
            "import os",
            "from typing import List, Optional",
            "from .storage import vault as memory_vault",
            "import ujson as json",
            "import json",
        ]

    def test_dependencies(self, sample):
        assert sample.dependencies == ["os", "typing", ".storage", "ujson", "json"]

    def test_symbol_names_and_types(self, sample):
        found = [(s.name, s.type) for s in sample.symbols]
        assert found == [
            ("MAX_TOKENS", "variable"),
            ("DEFAULT_MODEL", "variable"),
            ("API_KEY", "variable"),
            ("count", "function"),
            ("fetch", "function"),
            ("Pager", "class"),
            ("Pager.page_in", "method"),
            ("Pager.Config", "class"),
            ("Pager.Config.validate", "method"),
        ]

    def test_class_attributes_are_not_symbols(self, sample):
        names = [s.name for s in sample.symbols]
        assert "LIMIT" not in names and "Pager.LIMIT" not in names

    def test_signatures_keep_type_hints_and_defaults(self, sample):
        assert symbol(sample, "count").signature == "def count(text: str, *, model: Optional[str]=None) -> int:"
        assert symbol(sample, "Pager.page_in").signature == (
            "def page_in(self, query: str, budget: int=500) -> List[str]:"
        )
        assert symbol(sample, "Pager").signature == "class Pager(object):"

    def test_decorators_and_async_are_kept(self, sample):
        assert symbol(sample, "fetch").signature == (
            "@staticmethod\nasync def fetch(url: str, timeout: float=5.0) -> bytes:"
        )

    def test_variable_values(self, sample):
        assert symbol(sample, "MAX_TOKENS").signature == "MAX_TOKENS: int = 128000"
        assert symbol(sample, "DEFAULT_MODEL").signature == "DEFAULT_MODEL = 'llama3:8b'"

    def test_docstrings_and_line_numbers(self, sample):
        assert symbol(sample, "count").docstring == "Count tokens in text."
        assert symbol(sample, "fetch").docstring is None
        assert symbol(sample, "count").start_line == SAMPLE.splitlines().index(
            "def count(text: str, *, model: Optional[str] = None) -> int:") + 1

    def test_function_bodies_are_dropped(self, sample):
        assert "THIS_IS_IN_THE_BODY" not in render_skeleton(sample)
        assert "THIS_IS_IN_THE_BODY" not in sample.model_dump_json()

    def test_summary_uses_module_docstring_and_counts(self, sample):
        assert sample.summary.startswith("Token budget helpers for the agent harness.")
        assert "2 classes, 2 functions, 2 methods, 3 variables, 5 imports" in sample.summary

    def test_token_count_is_computed_or_taken_from_caller(self, sample):
        assert sample.original_token_count == count_tokens(SAMPLE)
        assert compact_code(SAMPLE, "a.py", original_token_count=1234).original_token_count == 1234

    def test_first_paragraph_docstring_mode(self):
        full = compact_code(SAMPLE, "a.py")
        short = compact_code(SAMPLE, "a.py", docstrings="first_paragraph")
        assert "Second paragraph" in symbol(full, "Pager").docstring
        assert symbol(short, "Pager").docstring == "Pages memories in and out."

    def test_unknown_docstring_mode_is_rejected(self):
        with pytest.raises(ValueError):
            compact_code(SAMPLE, "a.py", docstrings="everything")


# ---------------------------------------------------------------------------
# Fragments and bad input (context chunks are rarely whole files)
# ---------------------------------------------------------------------------

class TestFragments:

    def test_truncated_chunk_keeps_earlier_symbols(self):
        cut = SAMPLE[: SAMPLE.index("return []") + len("return [")]   # unclosed '[' in Pager.page_in
        schema = compact_code(cut, "a.py")
        names = [s.name for s in schema.symbols]
        assert names[:7] == ["MAX_TOKENS", "DEFAULT_MODEL", "API_KEY", "count", "fetch",
                             "Pager", "Pager.page_in"]
        assert "Partial parse (trimmed" in schema.summary

    def test_open_block_at_the_end_keeps_its_header(self):
        schema = compact_code("def first():\n    pass\n\ndef second(x):\n    y = (1,\n", "a.py")
        assert [s.name for s in schema.symbols] == ["first", "second"]

    def test_indented_method_is_dedented(self):
        method = "    def page_in(self, query: str) -> list:\n        return []\n"
        schema = compact_code(method, "a.py")
        assert [s.name for s in schema.symbols] == ["page_in"]
        assert "dedented" in schema.summary

    def test_unparseable_python_falls_back_to_line_scan(self):
        broken = "import os\nclass A:\n    def m(self) ->:\n        )))\ndef f(:\n"
        schema = compact_code(broken, "a.py")
        assert [(s.name, s.type) for s in schema.symbols] == [
            ("A", "class"), ("A.m", "method"), ("f", "function")]
        assert schema.imports == ["import os"]
        assert "line scan" in schema.summary

    @pytest.mark.parametrize("text", ["", "   \n\n", "not ) code ((( at all\n:::", "\x00\x01 binary"])
    def test_garbage_never_raises(self, text):
        schema = compact_code(text, "a.py")
        assert schema.symbols == []

    def test_other_languages_use_line_scan(self):
        js = "import React from 'react'\nexport class App {}\nexport async function load(url) {\n}\n"
        schema = compact_code(js, "src/app.js", language="javascript")
        assert schema.language == "javascript"
        assert [(s.name, s.type) for s in schema.symbols] == [("App", "class"), ("load", "function")]


# ---------------------------------------------------------------------------
# Security and schema limits
# ---------------------------------------------------------------------------

class TestSafety:

    def test_credential_values_are_not_stored(self, sample):
        assert symbol(sample, "API_KEY").signature == "API_KEY = ..."
        assert "sk-proj-" not in sample.model_dump_json()

    @pytest.mark.parametrize("name", ["ACCESS_TOKEN", "apiKey", "SECRET_KEY", "DB_PASSWORD", "token"])
    def test_credential_like_names_hide_their_value(self, name):
        assert compact_code(f'{name} = "abc123"', "a.py").symbols[0].signature == f"{name} = ..."

    @pytest.mark.parametrize("name", ["MAX_TOKENS", "SORT_KEY", "AUTHOR", "keyboard_layout"])
    def test_ordinary_names_keep_their_value(self, name):
        assert compact_code(f'{name} = "abc123"', "a.py").symbols[0].signature == f"{name} = 'abc123'"

    def test_secrets_in_signatures_are_scrubbed(self):
        source = 'def connect(token="ghp_' + "a" * 40 + '"):\n    pass\n'
        schema = compact_code(source, "a.py")
        assert "ghp_" not in schema.symbols[0].signature
        assert "[REDACTED_SECRET]" in schema.symbols[0].signature

    @pytest.mark.parametrize("raw, expected", [
        ("/home/user/project/main.py", "home/user/project/main.py"),
        ("../../etc/passwd", "etc/passwd"),
        ("C:\\Users\\me\\app.py", "Users/me/app.py"),
        ("./harness/graph.py", "harness/graph.py"),
        ("", "unknown/file.py"),
        (None, "unknown/file.py"),
    ])
    def test_file_paths_are_made_safe(self, raw, expected):
        assert compact_code("x = 1", raw).file_path == expected

    @pytest.mark.parametrize("raw, expected", [
        ("Python", "python"), ("py", "python"), ("", "unknown"), ("C++", "cpp"), ("rust", "rust")])
    def test_language_is_normalised(self, raw, expected):
        assert compact_code("x = 1", "a.txt", language=raw).language == expected

    def test_symbol_count_is_capped(self):
        many = "\n".join(f"def f{i}(): pass" for i in range(600))
        schema = compact_code(many, "a.py")
        assert len(schema.symbols) == 500
        assert "Only the first 500 of 600 symbols are kept." in schema.summary

    def test_long_signature_is_clipped(self):
        args = ", ".join(f"argument_{i}: int = {i}" for i in range(60))
        schema = compact_code(f"def f({args}):\n    pass\n", "a.py")
        assert len(schema.symbols[0].signature) == 512
        assert schema.symbols[0].signature.endswith("...")

    def test_non_ascii_names_are_skipped_not_fatal(self):
        schema = compact_code("def café():\n    pass\n\ndef ok():\n    pass\n", "a.py")
        assert [s.name for s in schema.symbols] == ["ok"]
        assert "1 symbol(s) with non-ASCII names were skipped." in schema.summary

    def test_special_token_text_does_not_crash_token_count(self):
        assert compact_code('MARKER = "<|endoftext|>"\n', "a.py").original_token_count > 0


# ---------------------------------------------------------------------------
# Compression on the project's own source files
# ---------------------------------------------------------------------------

PROJECT_FILES = [
    "harness/guardrails.py",
    "harness/graph.py",
    "server/amp_memory_server.py",
    "server/sandbox.py",
]


@pytest.mark.parametrize("relative_path", PROJECT_FILES)
def test_project_files_compact_and_validate(relative_path):
    path = ROOT / relative_path
    if not path.exists():
        pytest.skip(f"{relative_path} not present on this branch")
    schema = compact_code(path.read_text(encoding="utf-8"), relative_path)
    CodeMemorySchema.model_validate(schema.model_dump())
    assert schema.symbols, "a real module should yield symbols"
    assert "Partial parse" not in schema.summary
    assert skeleton_token_count(schema) < schema.original_token_count


def test_compression_report():
    """Prints the compression table (run with -s). Asserts the overall ratio
    stays well below the raw size in both docstring modes."""
    totals = {"orig": 0, "full": 0, "first_paragraph": 0}
    rows = []
    for relative_path in PROJECT_FILES:
        path = ROOT / relative_path
        if not path.exists():
            continue
        source = path.read_text(encoding="utf-8")
        full = compact_code(source, relative_path)
        short = compact_code(source, relative_path, docstrings="first_paragraph")
        orig, a, b = full.original_token_count, skeleton_token_count(full), skeleton_token_count(short)
        totals["orig"] += orig
        totals["full"] += a
        totals["first_paragraph"] += b
        rows.append(f"{relative_path:30} {orig:7} {a / orig:9.1%} {b / orig:9.1%}")

    print(f"\n{'file':30} {'tokens':>7} {'full':>9} {'1st para':>9}")
    print("\n".join(rows))
    full_ratio = totals["full"] / totals["orig"]
    short_ratio = totals["first_paragraph"] / totals["orig"]
    print(f"{'TOTAL':30} {totals['orig']:7} {full_ratio:9.1%} {short_ratio:9.1%}")
    assert full_ratio < 0.5
    assert short_ratio < 0.3


# ---------------------------------------------------------------------------
# The code graph: which function calls which
# ---------------------------------------------------------------------------

SHOP = '''from payments import pay
import json

class Cart:
    def total(self):
        return sum(self.prices())

    def prices(self):
        return [1, 2]

def load_cart(user):
    return Cart()

def checkout(user):
    cart = load_cart(user)
    print(json.dumps({}))
    return pay(cart.total())
'''


def test_the_code_graph_links_calls_to_what_the_file_defines_or_imports():
    calls = {s.name: s.calls for s in compact_code(SHOP, file_path="shop.py").symbols}
    assert calls["checkout"] == ["load_cart", "pay"]       # pay is imported by name; print and json.dumps are not
    assert calls["Cart.total"] == ["Cart.prices"]          # self.prices() is written as Cart.prices
    assert calls["load_cart"] == ["Cart"] and calls["Cart"] == [] and calls["Cart.prices"] == []


def test_calls_must_be_dotted_names():
    from harness.guardrails import CodeSymbol
    import pydantic
    CodeSymbol(name="f", type="function", signature="def f():", start_line=1, calls=["Cart.total", "load"])
    with pytest.raises(pydantic.ValidationError):
        CodeSymbol(name="f", type="function", signature="def f():", start_line=1, calls=["rm -rf /"])


def test_old_seals_still_verify_and_new_calls_are_sealed(monkeypatch):
    from harness import signing
    from harness.guardrails import UnifiedMemory
    monkeypatch.setenv(signing.SIGNING_KEY_ENV, "graph-test-key")
    schema = compact_code(SHOP, file_path="shop.py")
    memory = UnifiedMemory(session_id="g", memory_type="code", compaction_source="ast", raw_token_count=50,
                           code_payload=schema)
    without = memory.model_copy(deep=True)
    for symbol in without.code_payload.symbols:
        symbol.calls = []
    assert b'"calls"' not in signing._canonical_bytes(without)    # same bytes as before the code graph existed
    memory.signature = signing.sign_memory(memory)
    assert signing.verify_memory(memory)
    memory.code_payload.symbols[-1].calls = ["load_cart", "delete_everything"]
    assert not signing.verify_memory(memory)                      # the graph is covered by the seal


def test_the_code_graph_is_kept_for_run_memory_code_but_not_in_the_search_text():
    import server.amp_memory_server as server
    from harness import graph
    from harness.guardrails import UnifiedMemory
    memory = UnifiedMemory(session_id="g", memory_type="code", compaction_source="ast", raw_token_count=50,
                           code_payload=compact_code(SHOP, file_path="shop.py"))
    assert " calls " not in server._embedding_text(memory)          # lighter: no call lines (report 5p)
    assert memory.code_payload.symbols[-1].calls == ["load_cart", "pay"]           # still in the memory
    record = server._sandbox_record(memory)
    assert record["calls"]["checkout"] == ["load_cart", "pay"] and "checkout calls load_cart, pay" in record["text"]
    assert "calls: which function calls which" in graph._build_secure_system_prompt("ctx", "s1")
    _, body, plain = graph._memory_view(memory)                       # what the model sees when it is paged in
    assert "\nCalls: Cart.total -> Cart.prices; load_cart -> Cart; checkout -> load_cart, pay" in body
    assert "checkout -> load_cart, pay" in plain                       # Layer 4 sees the same text


# ---------------------------------------------------------------------------
# Tree-sitter: JavaScript, TypeScript, Java, C and C++
# ---------------------------------------------------------------------------

JS = ('import { pay } from "./payments";\n/** Adds the prices. */\nfunction total(cart) { return cart.items.length; }\n'
      'const checkout = (user) => { const c = loadCart(user); return pay(total(c)); };\n'
      'function loadCart(user) { return { user }; }\nclass Cart {\n  size() { return this.count(); }\n  count() { return 1; }\n}\n')
JAVA = ('import java.util.List;\n/** A shopping cart. */\npublic class Cart {\n'
        '  public int total() { return sum(); }\n  private int sum() { return 2; }\n}\n')
CPP = '#include "pay.h"\nclass Cart {\n public:\n  int total() { return sum(); }\n  int sum() { return 2; }\n};\n'
C = '#include <stdio.h>\nstatic int add(int a, int b) { return a + b; }\nint twice(int x) { return add(x, x); }\n'
TS = 'import { Item } from "./item";\nexport function first(items: Item[]): Item { return items[0]; }\n'

needs_tree_sitter = pytest.mark.skipif(not __import__("compactor.ast_code.ts_parser", fromlist=["x"]).available("javascript"),
                                       reason="tree-sitter-language-pack is not installed")


def graph(code, path):
    schema = compact_code(code, file_path=path)
    return schema, {s.name: (s.type, s.calls) for s in schema.symbols}


@needs_tree_sitter
def test_javascript_gets_symbols_imports_docs_and_calls():
    schema, symbols = graph(JS, "cart.js")
    assert schema.language == "javascript" and schema.dependencies == ["./payments"]
    assert symbols["checkout"] == ("function", ["loadCart", "pay", "total"])      # an arrow function, pay is imported
    assert symbols["Cart.size"] == ("method", ["Cart.count"])                     # this.count() -> Cart.count
    assert next(s for s in schema.symbols if s.name == "total").docstring == "Adds the prices."
    assert next(s for s in schema.symbols if s.name == "total").signature == "function total(cart)"


@needs_tree_sitter
def test_java_c_cpp_and_typescript():
    schema, symbols = graph(JAVA, "Cart.java")
    assert symbols == {"Cart": ("class", []), "Cart.total": ("method", ["Cart.sum"]), "Cart.sum": ("method", [])}
    assert schema.symbols[0].docstring == "A shopping cart." and schema.dependencies == ["java.util.List"]
    assert graph(CPP, "cart.cpp")[1]["Cart.total"] == ("method", ["Cart.sum"])
    schema, symbols = graph(C, "util.c")
    assert symbols["twice"] == ("function", ["add"]) and schema.dependencies == ["stdio.h"]
    schema, symbols = graph(TS, "list.ts")
    assert schema.language == "typescript" and "first" in symbols and schema.dependencies == ["./item"]


def test_without_tree_sitter_other_languages_fall_back_to_the_line_scan(monkeypatch):
    from compactor.ast_code import ts_parser
    monkeypatch.setattr(ts_parser, "available", lambda grammar: False)
    schema = compact_code(JAVA, file_path="Cart.java")
    assert schema.language == "java" and "line scan" in schema.summary
