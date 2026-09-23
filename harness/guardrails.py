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

Usage example:
    from harness.guardrails import UnifiedMemory, TokenState
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


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

    Attributes:
        name       : The symbol's identifier (e.g., "calculate_tokens").
        type       : The kind of construct this symbol represents.
        signature  : The full signature string with type hints preserved.
        docstring  : The symbol's existing docstring, if one was present.
        start_line : The line number in the original source file where this
                     symbol was defined — enables traceability back to source.
    """

    name: str = Field(
        ...,
        description="The identifier name of the symbol (e.g., 'calculate_tokens').",
    )
    type: Literal["class", "function", "method", "variable"] = Field(
        ...,
        description="The kind of code construct this symbol represents.",
    )
    signature: str = Field(
        ...,
        description=(
            "Full signature string with type hints, e.g., "
            "'def calculate_tokens(text: str) -> int:'."
        ),
    )
    docstring: Optional[str] = Field(
        default=None,
        description="The existing docstring from the source code, verbatim, if present.",
    )
    start_line: int = Field(
        ...,
        ge=1,  # Line numbers are 1-indexed
        description="Line number in the original source file where this symbol begins.",
    )


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

    file_path: str = Field(
        ...,
        description="Relative path to the parsed source file, e.g., 'harness/token_monitor.py'.",
    )
    language: str = Field(
        ...,
        description="Programming language of the file (e.g., 'python', 'javascript').",
    )
    imports: List[str] = Field(
        default_factory=list,
        description="All import/require statements found in the file, preserved verbatim.",
    )
    symbols: List[CodeSymbol] = Field(
        default_factory=list,
        description="All structural symbols (functions, classes, methods) extracted by the AST parser.",
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="Cross-file or external package dependencies inferred from imports and call sites.",
    )
    summary: Optional[str] = Field(
        default=None,
        description="AI-generated high-level description of the file's purpose and responsibilities.",
    )
    original_token_count: int = Field(
        ...,
        ge=0,
        description="Number of tokens in the raw source file before compaction.",
    )


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

    chunk_index: int = Field(
        ...,
        ge=0,
        description="0-indexed position of this chunk within the current compaction pass.",
    )
    total_chunks: int = Field(
        ...,
        ge=1,
        description="Total number of chunks produced in this compaction pass.",
    )
    timeframe: str = Field(
        ...,
        description=(
            "Human-readable window description, e.g., "
            "'Session 0 to 80% token limit, messages 1-247'."
        ),
    )
    architectural_decisions: List[str] = Field(
        default_factory=list,
        description="Key design choices made in this context window.",
    )
    bugs_resolved: List[str] = Field(
        default_factory=list,
        description="Bugs or errors that were identified and fixed within this chunk.",
    )
    user_constraints: List[str] = Field(
        default_factory=list,
        description="Explicit rules, preferences, or constraints the user stated in this window.",
    )
    factual_summary: str = Field(
        ...,
        description=(
            "A highly dense narrative paragraph summarising everything that happened "
            "in this chunk. This is the primary field used for semantic search retrieval."
        ),
    )
    original_token_count: int = Field(
        ...,
        ge=0,
        description="Number of tokens in the raw conversation chunk before summarization.",
    )

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
        description="Unique identifier for the agent session this memory belongs to.",
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
        description="Number of tokens the original (pre-compaction) content consumed.",
    )
    vector_embedding_id: Optional[str] = Field(
        default=None,
        description=(
            "ChromaDB document ID assigned after this memory is written to the vector store. "
            "Remains None until store_memory() is called by Teammate 1."
        ),
    )

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

    session_id: str = Field(
        ...,
        description="Current session identifier. Matches the session_id in UnifiedMemory records.",
    )
    current_token_count: int = Field(
        default=0,
        ge=0,
        description="Live number of tokens currently occupying the active context window.",
    )
    token_limit: int = Field(
        ...,
        ge=1,
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
