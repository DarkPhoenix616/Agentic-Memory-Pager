"""
harness/guardrails.py — AMP Shared Pydantic Data Contracts
===========================================================
This is the single source of truth for all data schemas used across the
Agentic Memory Pager (AMP) system. Every module — the FastMCP Server (T1),
the AST Compactor (T2), the RLM Summarizer (T3), and the LangGraph Harness (T4)
— must import and use these models to exchange data. Never pass raw dicts between
modules; always use these validated Pydantic models.

Schema hierarchy:
    CodeSymbol          -- A single extracted code symbol (function / class / method)
    CodeMemorySchema    -- Full compacted output for one source file (T2 output)
    TextMemorySchema    -- Compacted conversational history chunk (T3 output)
    UnifiedMemory       -- Universal envelope stored to ChromaDB + SQLite (T1 input)
    TokenState          -- Live LangGraph state object (T4 internal state machine)

Security guardrails implemented (Phase 1):
    - ConfigDict(str_strip_whitespace=True, extra="forbid") on all models.
    - Identifier validation on CodeSymbol.name (safe Python identifier regex).
    - Path traversal prevention on CodeMemorySchema.file_path.
    - Language format validation on CodeMemorySchema.language.
    - Secret / credential scrubbing on TextMemorySchema text fields.
    - Null-byte and control-character sanitization.
    - Session ID slug enforcement on UnifiedMemory and TokenState.
    - DoS-protection via max_length / max_items bounds on every field.

Usage example:
    from harness.guardrails import UnifiedMemory, TokenState
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Security helpers — module-level compiled regexes for performance
# ---------------------------------------------------------------------------

# Safe Python/JS identifier: starts with letter or underscore, then letters,
# digits, underscores, or dots (supports dotted names like "pkg.module").
_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_.]*$")

# Session IDs must be URL-safe slugs: letters, digits, hyphens, underscores.
# Length 1–64 is enforced by the regex itself.
_SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

# Language codes: lowercase letters, digits, hyphens, underscores only.
_LANGUAGE_RE = re.compile(r"^[a-z0-9_-]+$")

# Secret / credential patterns — ordered most-specific to least-specific.
_SECRET_PATTERNS: list[tuple[re.Pattern, str]] = [
    # PEM private key blocks
    (re.compile(
        r"-----BEGIN [A-Z ]+ PRIVATE KEY-----.*?-----END [A-Z ]+ PRIVATE KEY-----",
        re.DOTALL,
    ), "[REDACTED_SECRET]"),
    # OpenAI API keys  sk-proj-... or sk-...
    (re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9]{20,}\b"), "[REDACTED_SECRET]"),
    # GitHub tokens  ghp_, gho_, ghs_, ghr_, ghu_  (36+ alphanumeric chars)
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"), "[REDACTED_SECRET]"),
    # AWS access key IDs
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED_SECRET]"),
    # Bearer tokens
    (re.compile(r"\bBearer\s+[A-Za-z0-9\-._~+/]{20,}\b", re.IGNORECASE), "[REDACTED_SECRET]"),
    # Generic key/secret/password assignments with high-entropy values
    (re.compile(
        r"(?i)\b(?:secret|token|key|password|passwd|pwd)\s*[=:]\s*['\"]?[A-Za-z0-9+/]{32,}['\"]?"
    ), "[REDACTED_SECRET]"),
]

# Control characters to strip (null byte + ASCII C0 control chars except TAB/LF/CR)
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _scrub_secrets(text: str) -> str:
    """Strip control characters and replace known secret patterns with [REDACTED_SECRET]."""
    text = _CONTROL_CHAR_RE.sub("", text)
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _scrub_list(items: List[str]) -> List[str]:
    """Apply _scrub_secrets to every string in a list."""
    return [_scrub_secrets(item) for item in items]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CompactionRoute(str, Enum):
    """
    Represents which compaction engine the DynamicRouter should invoke.

    This enum is returned by the 80% trigger logic and consumed by
    LangGraph nodes to determine the next step in the agent loop.
    """

    AST = "ast"   # Route to Teammate 2's AST Code Compactor
    RLM = "rlm"   # Route to Teammate 3's RLM Text Compactor


# ---------------------------------------------------------------------------
# Model 1: CodeSymbol
# ---------------------------------------------------------------------------

class CodeSymbol(BaseModel):
    """
    Represents a single structural unit extracted from a source file by the
    AST Compactor (Teammate 2).

    A 'symbol' is any named, top-level declaration in a source file — a
    function, a class, a method inside a class, or a module-level variable.
    Storing symbols individually allows the search_vault() MCP tool to
    retrieve a *specific function* rather than the entire file.

    Security guardrails:
        - name is validated against a safe identifier regex to block XSS,
          script injection, or markdown injection attacks.
        - All string fields have max_length bounds to prevent DoS via oversized
          payloads reaching the ChromaDB embedding service.
        - extra='forbid' ensures unknown fields from LLM output are rejected.

    Attributes:
        name       : The symbol's identifier (e.g., "calculate_tokens").
        type       : The kind of construct this symbol represents.
        signature  : The full signature string with type hints preserved.
        docstring  : The symbol's existing docstring, if one was present.
        start_line : The line number in the original source file where this
                     symbol was defined — enables traceability back to source.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    name: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="The identifier name of the symbol (e.g., 'calculate_tokens').",
    )
    type: Literal["class", "function", "method", "variable"] = Field(
        ...,
        description="The kind of code construct this symbol represents.",
    )
    signature: str = Field(
        ...,
        min_length=3,
        max_length=512,
        description=(
            "Full signature string with type hints, e.g., "
            "'def calculate_tokens(text: str) -> int:'."
        ),
    )
    docstring: Optional[str] = Field(
        default=None,
        max_length=4000,
        description="The existing docstring from the source code, verbatim, if present.",
    )
    start_line: int = Field(
        ...,
        ge=1,          # Line numbers are 1-indexed
        le=1_000_000,  # Guard against absurdly large line numbers
        description="Line number in the original source file where this symbol begins.",
    )

    @field_validator("name")
    @classmethod
    def name_must_be_valid_identifier(cls, v: str) -> str:
        """
        Ensure the symbol name is a valid Python/JS identifier.

        Rejects: script tags, markdown, spaces, operators, path characters,
        SQL injection strings, and any other non-identifier input.
        """
        if not _IDENTIFIER_RE.match(v):
            raise ValueError(
                f"Symbol name '{v}' is not a valid identifier. "
                "Must match ^[a-zA-Z_][a-zA-Z0-9_.]*$"
            )
        return v


# ---------------------------------------------------------------------------
# Model 2: CodeMemorySchema
# ---------------------------------------------------------------------------

class CodeMemorySchema(BaseModel):
    """
    The complete compacted output for a single source file, produced by the
    AST Compactor pipeline (Teammate 2).

    A raw source file may contain thousands of tokens. This schema captures
    only the structural skeleton — imports, signatures, class hierarchy, and
    docstrings — reducing a 5,000-token file to roughly 500 tokens while
    preserving full recoverability of the file's architecture.

    This schema is the 'code_payload' field of a UnifiedMemory envelope when
    memory_type == 'code'.

    Security guardrails:
        - file_path is validated to prevent path traversal attacks (no '..', no
          leading slashes, no absolute drive letters like 'C:').
        - language is validated to allow only safe slug characters.
        - All list fields have max_length to prevent DoS via massive payloads.
        - Per-item max_length enforced on imports (256) and dependencies (128).
        - extra='forbid' rejects unknown/injected fields from LLM outputs.

    Attributes:
        file_path           : Relative path to the parsed source file.
        language            : Programming language — needed so tree-sitter loads
                              the correct grammar parser.
        imports             : All import/require statements, verbatim.
        symbols             : List of all extracted CodeSymbol objects.
        dependencies        : Cross-file or external module dependencies inferred
                              from the imports and symbol usages.
        summary             : Optional AI-generated one-paragraph description of
                              what the file does at a high level.
        original_token_count: Token count of the raw file before compaction.
                              Used to compute and report the compression ratio.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    file_path: str = Field(
        ...,
        min_length=1,
        max_length=260,  # Windows MAX_PATH equivalent
        description="Relative path to the parsed source file, e.g., 'harness/token_monitor.py'.",
    )
    language: str = Field(
        ...,
        min_length=1,
        max_length=32,
        description="Programming language of the file (e.g., 'python', 'javascript').",
    )
    imports: List[str] = Field(
        default_factory=list,
        max_length=200,  # DoS guard: no file should have >200 top-level imports
        description="All import/require statements found in the file, preserved verbatim.",
    )
    symbols: List[CodeSymbol] = Field(
        default_factory=list,
        max_length=500,  # DoS guard: a single file should not have >500 symbols
        description="All structural symbols (functions, classes, methods) extracted by the AST parser.",
    )
    dependencies: List[str] = Field(
        default_factory=list,
        max_length=100,
        description="Cross-file or external package dependencies inferred from imports and call sites.",
    )
    summary: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="AI-generated high-level description of the file's purpose and responsibilities.",
    )
    original_token_count: int = Field(
        ...,
        ge=0,
        le=2_000_000,  # Guard: no single file should produce >2M tokens
        description="Number of tokens in the raw source file before compaction.",
    )

    @field_validator("file_path")
    @classmethod
    def file_path_must_be_safe_relative(cls, v: str) -> str:
        """
        Prevent path traversal attacks on the stored file_path field.

        Rules enforced:
          - No '..' components (directory traversal).
          - No leading '/' or '\\' (absolute Unix/Windows paths).
          - No Windows drive letters like 'C:' or 'c:'.
        """
        # Reject directory traversal
        if ".." in v.replace("\\", "/").split("/"):
            raise ValueError(
                f"file_path '{v}' contains path traversal ('..') which is not allowed."
            )
        # Reject absolute Unix paths
        if v.startswith("/") or v.startswith("\\"):
            raise ValueError(
                f"file_path '{v}' must be a relative path (no leading slash)."
            )
        # Reject Windows absolute paths (e.g. C:\, D:/)
        if len(v) >= 2 and v[1] == ":":
            raise ValueError(
                f"file_path '{v}' appears to be an absolute Windows path, which is not allowed."
            )
        return v

    @field_validator("language")
    @classmethod
    def language_must_be_slug(cls, v: str) -> str:
        """Ensure language is a safe slug (e.g. 'python', 'javascript', 'c-sharp')."""
        if not _LANGUAGE_RE.match(v):
            raise ValueError(
                f"language '{v}' must contain only lowercase letters, digits, hyphens, "
                "and underscores (e.g., 'python', 'javascript', 'c-sharp')."
            )
        return v

    @field_validator("imports")
    @classmethod
    def imports_item_max_length(cls, v: List[str]) -> List[str]:
        """Each individual import statement must not exceed 256 characters."""
        for item in v:
            if len(item) > 256:
                raise ValueError(
                    f"An import statement exceeds the 256-character limit: '{item[:60]}...'"
                )
        return v

    @field_validator("dependencies")
    @classmethod
    def dependencies_item_max_length(cls, v: List[str]) -> List[str]:
        """Each individual dependency must not exceed 128 characters."""
        for item in v:
            if len(item) > 128:
                raise ValueError(
                    f"A dependency string exceeds the 128-character limit: '{item[:60]}...'"
                )
        return v


# ---------------------------------------------------------------------------
# Model 3: TextMemorySchema
# ---------------------------------------------------------------------------

class TextMemorySchema(BaseModel):
    """
    The compacted output for a single chunk of conversation history, produced
    by the RLM Summarizer pipeline (Teammate 3).

    Instead of keeping 50,000+ tokens of raw dialogue in the active context
    window, the RLM engine splits it into chunks and distills each chunk into
    a structured 'fact map' containing only the semantically important
    information: architectural decisions, resolved bugs, and user constraints.

    Multi-pass compaction: if the conversation is very long, the RLM engine
    may run multiple passes. chunk_index and total_chunks track position within
    a single pass, ensuring no chunk's output is lost.

    This schema is the 'text_payload' field of a UnifiedMemory envelope when
    memory_type == 'text'.

    Security guardrails:
        - factual_summary, architectural_decisions, bugs_resolved, and
          user_constraints are scrubbed for secrets (API keys, tokens, private
          keys) before the data can be stored in ChromaDB. Detected secrets are
          replaced with '[REDACTED_SECRET]'.
        - Null bytes and dangerous control characters are stripped from all text
          fields to prevent encoding attacks.
        - All list and string fields have max_length bounds to prevent DoS.
        - extra='forbid' prevents unknown field injection.

    Attributes:
        chunk_index          : 0-indexed position of this chunk in the current pass.
        total_chunks         : Total number of chunks in this compaction pass.
        timeframe            : Human-readable description of the time window covered.
        architectural_decisions: Key design choices explicitly made in this window.
        bugs_resolved        : Issues identified and fixed within this context chunk.
        user_constraints     : Explicit rules or preferences the user stated.
        factual_summary      : A dense prose paragraph summarizing everything in
                               this chunk — the primary content used for retrieval.
        original_token_count : Token count of the raw dialogue before compaction.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    chunk_index: int = Field(
        ...,
        ge=0,
        description="0-indexed position of this chunk within the current compaction pass.",
    )
    total_chunks: int = Field(
        ...,
        ge=1,
        le=1000,  # DoS guard: no reasonable compaction run needs >1000 chunks
        description="Total number of chunks produced in this compaction pass.",
    )
    timeframe: str = Field(
        ...,
        min_length=3,
        max_length=128,
        description=(
            "Human-readable window description, e.g., "
            "'Session 0 to 80% token limit, messages 1-247'."
        ),
    )
    architectural_decisions: List[str] = Field(
        default_factory=list,
        max_length=50,  # DoS guard
        description="Key design choices made in this context window.",
    )
    bugs_resolved: List[str] = Field(
        default_factory=list,
        max_length=50,  # DoS guard
        description="Bugs or errors that were identified and fixed within this chunk.",
    )
    user_constraints: List[str] = Field(
        default_factory=list,
        max_length=50,  # DoS guard
        description="Explicit rules, preferences, or constraints the user stated in this window.",
    )
    factual_summary: str = Field(
        ...,
        min_length=10,
        max_length=8000,
        description=(
            "A highly dense narrative paragraph summarising everything that happened "
            "in this chunk. This is the primary field used for semantic search retrieval."
        ),
    )
    original_token_count: int = Field(
        ...,
        ge=0,
        le=2_000_000,
        description="Number of tokens in the raw conversation chunk before summarization.",
    )

    # ------------------------------------------------------------------
    # Secret scrubbing validators — run before the cross-field model_validator
    # so secrets are redacted before being stored in ChromaDB.
    # ------------------------------------------------------------------

    @field_validator("factual_summary")
    @classmethod
    def scrub_factual_summary(cls, v: str) -> str:
        """Scrub secrets and control characters from the primary retrieval field."""
        return _scrub_secrets(v)

    @field_validator("architectural_decisions", "bugs_resolved", "user_constraints")
    @classmethod
    def scrub_and_bound_text_lists(cls, v: List[str]) -> List[str]:
        """
        Scrub secrets from every item in the list and enforce per-item max_length.
        Items exceeding 500 characters are rejected (not silently truncated).
        """
        scrubbed = _scrub_list(v)
        for item in scrubbed:
            if len(item) > 500:
                raise ValueError(
                    f"A list item exceeds the 500-character limit: '{item[:60]}...'"
                )
        return scrubbed

    @field_validator("timeframe")
    @classmethod
    def sanitize_timeframe(cls, v: str) -> str:
        """Strip control characters from the timeframe label."""
        return _CONTROL_CHAR_RE.sub("", v)

    # ------------------------------------------------------------------
    # Cross-field validator (existing guardrail preserved)
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def chunk_index_within_bounds(self) -> "TextMemorySchema":
        """Ensure chunk_index is a valid position within total_chunks."""
        if self.chunk_index >= self.total_chunks:
            raise ValueError(
                f"chunk_index ({self.chunk_index}) must be less than "
                f"total_chunks ({self.total_chunks})."
            )
        return self


# ---------------------------------------------------------------------------
# Model 4: UnifiedMemory
# ---------------------------------------------------------------------------

class UnifiedMemory(BaseModel):
    """
    The universal memory envelope that is persisted to ChromaDB (for vector
    search) and SQLite (for relational metadata queries) by the FastMCP Server
    (Teammate 1).

    This model is the single type that Teammate 1's store_memory() and
    search_vault() MCP tools accept and return. By wrapping both CodeMemorySchema
    and TextMemorySchema in one envelope with consistent metadata, the storage
    layer remains completely agnostic to the type of memory being stored.

    The 'vector_embedding_id' field is intentionally left None at creation time.
    Teammate 1's store_memory() tool sets it after writing to ChromaDB, then
    returns the updated UnifiedMemory to the caller.

    Security guardrails:
        - session_id is validated as a safe alphanumeric slug
          (^[a-zA-Z0-9_-]{1,64}$) to prevent SQL injection and path injection
          in session-based metadata queries.
        - vector_embedding_id max_length=64 guards against oversized ChromaDB IDs.
        - extra='forbid' prevents injection of unexpected fields from LLM output.
        - Cross-field model_validator ensures memory_type, compaction_source,
          and payload fields are consistent (existing guardrail preserved).

    Attributes:
        id                 : UUID4 string — unique identifier for this memory object.
        timestamp          : UTC datetime of when this memory was created.
        session_id         : Identifies which agent session this memory belongs to.
        memory_type        : Whether this contains code or text payload.
        compaction_source  : Which compaction engine produced the payload.
        code_payload       : Populated when memory_type == 'code'. Otherwise None.
        text_payload       : Populated when memory_type == 'text'. Otherwise None.
        raw_token_count    : Tokens consumed by the original content before compaction.
        vector_embedding_id: ChromaDB document ID — set by T1 after storage. None until stored.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="UUID4 string — unique identifier for this memory object.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of when this memory envelope was created.",
    )
    session_id: str = Field(
        ...,
        description=(
            "Unique identifier for the agent session this memory belongs to. "
            "Must be a safe alphanumeric slug (letters, digits, hyphens, underscores, 1-64 chars)."
        ),
    )
    memory_type: Literal["code", "text"] = Field(
        ...,
        description="Indicates which payload field is populated — 'code' or 'text'.",
    )
    compaction_source: Literal["ast", "rlm"] = Field(
        ...,
        description="The compaction engine that produced the payload ('ast' for T2, 'rlm' for T3).",
    )
    code_payload: Optional[CodeMemorySchema] = Field(
        default=None,
        description="Populated when memory_type == 'code'. Contains the AST-compacted file skeleton.",
    )
    text_payload: Optional[TextMemorySchema] = Field(
        default=None,
        description="Populated when memory_type == 'text'. Contains the RLM-summarized fact map.",
    )
    raw_token_count: int = Field(
        ...,
        ge=0,
        le=2_000_000,
        description="Number of tokens the original (pre-compaction) content consumed.",
    )
    vector_embedding_id: Optional[str] = Field(
        default=None,
        max_length=64,
        description=(
            "ChromaDB document ID assigned after this memory is written to the vector store. "
            "Remains None until store_memory() is called by Teammate 1."
        ),
    )

    @field_validator("session_id")
    @classmethod
    def session_id_must_be_safe_slug(cls, v: str) -> str:
        """
        Enforce a safe alphanumeric slug format for session_id.

        Rejects: empty strings, spaces, SQL injection characters (' ; --),
        path characters (/ \\ . ..), angle brackets, quotes, and any other
        non-slug characters. Prevents session IDs being used as injection vectors
        in ChromaDB 'where' filters or SQLite metadata queries.
        """
        if not _SESSION_ID_RE.match(v):
            raise ValueError(
                f"session_id '{v}' is invalid. Must match ^[a-zA-Z0-9_-]{{1,64}}$ "
                "(letters, digits, hyphens, underscores; 1-64 characters)."
            )
        return v

    @model_validator(mode="after")
    def validate_payload_consistency(self) -> "UnifiedMemory":
        """
        Enforce that memory_type, compaction_source, and payload fields are consistent.

        Rules:
          - memory_type == 'code'  requires compaction_source == 'ast'  and code_payload set.
          - memory_type == 'text'  requires compaction_source == 'rlm'  and text_payload set.
        """
        if self.memory_type == "code":
            # Code memory must come from the AST engine and carry a code payload
            if self.compaction_source != "ast":
                raise ValueError("memory_type='code' requires compaction_source='ast'.")
            if self.code_payload is None:
                raise ValueError("memory_type='code' requires code_payload to be set.")
        else:
            # Text memory must come from the RLM engine and carry a text payload
            if self.compaction_source != "rlm":
                raise ValueError("memory_type='text' requires compaction_source='rlm'.")
            if self.text_payload is None:
                raise ValueError("memory_type='text' requires text_payload to be set.")
        return self

# ---------------------------------------------------------------------------
# Model 4b: RetrievedMemory and PageContextResult
# ---------------------------------------------------------------------------

class RetrievedMemory(BaseModel):
    """
    A UnifiedMemory returned by a search, together with the details that only
    make sense for one specific query.

    UnifiedMemory stays the pure stored data produced by the compactors.
    Distance and token cost describe a single search, so they are kept out of
    UnifiedMemory and carried here instead.

    Attributes:
        memory     : The stored UnifiedMemory.
        distance   : Vector distance between the query and this memory.
                     Smaller means a closer semantic match.
        token_cost : Tokens this memory costs in context (token count of its
                     searchable text), used for token-budget packing.
    """

    model_config = ConfigDict(extra="forbid")

    memory: UnifiedMemory = Field(
        ...,
        description="The stored memory envelope.",
    )
    distance: float = Field(
        ...,
        description="Vector distance to the query. Smaller means closer.",
    )
    token_cost: int = Field(
        ...,
        ge=0,
        description="Tokens this memory costs in context.",
    )


class PageContextResult(BaseModel):
    """
    Result of the page_context tool: the memories that fit within a token
    budget, plus how the budget was used.

    Attributes:
        memories     : Selected memories, best match first.
        tokens_used  : Sum of token_cost over the selected memories.
        total_budget : The token budget that was requested.
        skipped      : How many matches were left out because they did not fit.
    """

    model_config = ConfigDict(extra="forbid")

    memories: List[RetrievedMemory] = Field(
        default_factory=list,
        description="Selected memories, best match first.",
    )
    tokens_used: int = Field(
        default=0,
        ge=0,
        description="Total token_cost of the selected memories.",
    )
    total_budget: int = Field(
        ...,
        description="The token budget that was requested.",
    )
    skipped: int = Field(
        default=0,
        ge=0,
        description="Number of matches left out because they did not fit the budget.",
    )


# ---------------------------------------------------------------------------
# Model 5: TokenState
# ---------------------------------------------------------------------------

class TokenState(BaseModel):
    """
    The live state object that flows through every node of the LangGraph
    agent state machine (Teammate 4).

    TokenState is NOT persisted to the database — it lives in memory for the
    duration of a single agent session. It carries all the information needed
    to decide: should we compact now? What should we compact? How many times
    have we already compacted?

    This is T4's central nervous system. Every routing decision, every
    compaction trigger, and every telemetry metric passes through this object.

    Security guardrails:
        - session_id validated as a safe alphanumeric slug (same rule as
          UnifiedMemory) to ensure consistency when cross-referencing stored
          memory records.
        - context_chunks max_length=1000 to prevent runaway memory accumulation
          in the live state machine.
        - token_limit bounded to [1, 10_000_000] to prevent division-by-zero
          and guard against absurd model sizes being injected.
        - extra='forbid' prevents unexpected state fields from LangGraph nodes.

    Attributes:
        session_id             : Session ID, links to UnifiedMemory objects in storage.
        current_token_count    : Live token count in the active context window.
        token_limit            : The model's maximum context window size (e.g., 128000).
        compaction_threshold   : Fraction of token_limit at which compaction fires (default 0.80).
        context_chunks         : The raw message dicts currently loaded in the context window.
        oldest_chunk_type      : Content type of the oldest chunk — drives routing to T2 or T3.
        compaction_in_progress : Guard flag — prevents re-entrant compaction triggers.
        total_compactions      : Telemetry counter — how many times compaction has fired this session.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    session_id: str = Field(
        ...,
        description=(
            "Current session identifier. Matches the session_id in UnifiedMemory records. "
            "Must be a safe alphanumeric slug (letters, digits, hyphens, underscores, 1-64 chars)."
        ),
    )
    current_token_count: int = Field(
        default=0,
        ge=0,
        description="Live number of tokens currently occupying the active context window.",
    )
    token_limit: int = Field(
        ...,
        ge=1,
        le=10_000_000,  # Guard: no model today exceeds 10M tokens; prevents division near zero
        description="Maximum context window size for the active model (e.g., 128000 for Llama-3).",
    )
    compaction_threshold: float = Field(
        default=0.80,
        gt=0.0,
        le=1.0,
        description=(
            "Fraction of token_limit at which the 80% compaction trigger fires. "
            "Default is 0.80 (80%). Configurable per-session."
        ),
    )
    context_chunks: List[dict] = Field(
        default_factory=list,
        max_length=1000,  # DoS guard: prevent runaway accumulation in live state
        description=(
            "The raw message/chunk dicts currently loaded in the context window. "
            "Each dict has at minimum: {'role': str, 'content': str, 'type': 'code'|'text'}."
        ),
    )
    oldest_chunk_type: Literal["code", "text"] = Field(
        default="text",
        description=(
            "Content type of the oldest (first) chunk in context_chunks. "
            "Determines whether the DynamicRouter calls the AST or RLM compactor."
        ),
    )
    compaction_in_progress: bool = Field(
        default=False,
        description=(
            "Guard flag that prevents re-entrant compaction. Set to True when "
            "compaction begins and back to False when it completes."
        ),
    )
    total_compactions: int = Field(
        default=0,
        ge=0,
        description="Telemetry counter — how many compaction cycles have run in this session.",
    )

    @field_validator("session_id")
    @classmethod
    def session_id_must_be_safe_slug(cls, v: str) -> str:
        """
        Enforce the same safe alphanumeric slug format as UnifiedMemory.session_id.
        Ensures session IDs are consistent and injection-safe across both models.
        """
        if not _SESSION_ID_RE.match(v):
            raise ValueError(
                f"session_id '{v}' is invalid. Must match ^[a-zA-Z0-9_-]{{1,64}}$ "
                "(letters, digits, hyphens, underscores; 1-64 characters)."
            )
        return v

    @property
    def utilization(self) -> float:
        """
        Returns the current context window utilization as a fraction (0.0 to 1.0).

        Example: If current_token_count=102400 and token_limit=128000,
        utilization = 0.80, meaning we are exactly at the compaction threshold.
        """
        return self.current_token_count / self.token_limit

    @property
    def should_compact(self) -> bool:
        """
        Returns True if the context window utilization has reached or exceeded
        the compaction threshold AND no compaction is currently in progress.

        This is the single decision point for the LangGraph conditional edge.
        """
        return self.utilization >= self.compaction_threshold and not self.compaction_in_progress



