"""
tests/test_schemas.py — Unit Tests for harness/guardrails.py
=============================================================
Validates that all five Pydantic schemas in the AMP data contracts work
correctly. Tests are organized into three categories:

  1. POSITIVE TESTS  — valid data deserializes without errors.
  2. NEGATIVE TESTS  — missing/invalid fields raise ValidationError.
  3. PROPERTY TESTS  — computed properties (utilization, should_compact) return
                       correct values.

Run:
    pytest tests/test_schemas.py -v

Requirements:
    pip install pydantic pytest
"""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

# Import all schemas from the shared data contract module
from harness.guardrails import (
    CodeMemorySchema,
    CodeSymbol,
    CompactionRoute,
    TextMemorySchema,
    TokenState,
    UnifiedMemory,
)


# ===========================================================================
# Fixtures — reusable valid sample objects
# ===========================================================================

@pytest.fixture
def valid_code_symbol() -> dict:
    """Returns a minimal valid CodeSymbol payload as a dict."""
    return {
        "name": "calculate_tokens",
        "type": "function",
        "signature": "def calculate_tokens(text: str) -> int:",
        "docstring": "Count the number of tokens in a text string.",
        "start_line": 42,
    }


@pytest.fixture
def valid_code_memory(valid_code_symbol) -> dict:
    """Returns a minimal valid CodeMemorySchema payload as a dict."""
    return {
        "file_path": "harness/token_monitor.py",
        "language": "python",
        "imports": ["import re", "from typing import List"],
        "symbols": [valid_code_symbol],
        "dependencies": ["harness.guardrails"],
        "summary": "Monitors token usage and triggers compaction at the 80% threshold.",
        "original_token_count": 4800,
    }


@pytest.fixture
def valid_text_memory() -> dict:
    """Returns a minimal valid TextMemorySchema payload as a dict."""
    return {
        "chunk_index": 0,
        "total_chunks": 3,
        "timeframe": "Session 0 to 80% token limit, messages 1-247",
        "architectural_decisions": [
            "Use ChromaDB not Pinecone",
            "80% compaction threshold, not 90%",
        ],
        "bugs_resolved": ["Fixed NoneType in token_monitor.py line 42"],
        "user_constraints": ["No paid APIs", "Use Llama-3 not GPT-4"],
        "factual_summary": (
            "The team decided to use ChromaDB for vector storage and SQLite for metadata. "
            "Token monitoring fires at 80% capacity. A NoneType bug in token_monitor was resolved."
        ),
        "original_token_count": 48000,
    }


@pytest.fixture
def valid_unified_code_memory(valid_code_memory) -> dict:
    """Returns a valid UnifiedMemory envelope wrapping a CodeMemorySchema."""
    return {
        "session_id": "session-abc-123",
        "memory_type": "code",
        "compaction_source": "ast",
        "code_payload": valid_code_memory,
        "raw_token_count": 4800,
    }


@pytest.fixture
def valid_unified_text_memory(valid_text_memory) -> dict:
    """Returns a valid UnifiedMemory envelope wrapping a TextMemorySchema."""
    return {
        "session_id": "session-abc-123",
        "memory_type": "text",
        "compaction_source": "rlm",
        "text_payload": valid_text_memory,
        "raw_token_count": 48000,
    }


@pytest.fixture
def valid_token_state() -> dict:
    """Returns a valid TokenState payload as a dict."""
    return {
        "session_id": "session-abc-123",
        "current_token_count": 0,
        "token_limit": 128000,
    }


# ===========================================================================
# 1. POSITIVE TESTS — valid data should parse without errors
# ===========================================================================

class TestCodeSymbolPositive:
    """Positive tests for the CodeSymbol model."""

    def test_full_valid_symbol(self, valid_code_symbol):
        """A fully populated CodeSymbol should parse without errors."""
        symbol = CodeSymbol(**valid_code_symbol)
        assert symbol.name == "calculate_tokens"
        assert symbol.type == "function"
        assert symbol.start_line == 42

    def test_symbol_without_docstring(self, valid_code_symbol):
        """docstring is optional — omitting it should be valid."""
        valid_code_symbol.pop("docstring")
        symbol = CodeSymbol(**valid_code_symbol)
        assert symbol.docstring is None  # Should default to None

    def test_class_type_symbol(self):
        """A symbol of type 'class' should be accepted."""
        symbol = CodeSymbol(
            name="TokenMonitor",
            type="class",
            signature="class TokenMonitor:",
            start_line=10,
        )
        assert symbol.type == "class"


class TestCodeMemorySchemaPositive:
    """Positive tests for the CodeMemorySchema model."""

    def test_full_valid_code_memory(self, valid_code_memory):
        """A fully populated CodeMemorySchema should parse without errors."""
        mem = CodeMemorySchema(**valid_code_memory)
        assert mem.file_path == "harness/token_monitor.py"
        assert mem.language == "python"
        assert len(mem.symbols) == 1
        assert mem.original_token_count == 4800

    def test_empty_imports_and_symbols(self):
        """imports and symbols have defaults — omitting both should be valid."""
        mem = CodeMemorySchema(
            file_path="server/mcp_server.py",
            language="python",
            original_token_count=200,
        )
        assert mem.imports == []
        assert mem.symbols == []


class TestTextMemorySchemaPositive:
    """Positive tests for the TextMemorySchema model."""

    def test_full_valid_text_memory(self, valid_text_memory):
        """A fully populated TextMemorySchema should parse without errors."""
        mem = TextMemorySchema(**valid_text_memory)
        assert mem.chunk_index == 0
        assert mem.total_chunks == 3
        assert len(mem.architectural_decisions) == 2

    def test_single_chunk_pass(self):
        """chunk_index=0, total_chunks=1 (single-chunk pass) should be valid."""
        mem = TextMemorySchema(
            chunk_index=0,
            total_chunks=1,
            timeframe="Full session",
            factual_summary="A brief session where the project was scaffolded.",
            original_token_count=5000,
        )
        assert mem.chunk_index == 0
        assert mem.total_chunks == 1


class TestUnifiedMemoryPositive:
    """Positive tests for the UnifiedMemory model."""

    def test_code_memory_envelope(self, valid_unified_code_memory):
        """A code-type UnifiedMemory with an ast source should parse correctly."""
        mem = UnifiedMemory(**valid_unified_code_memory)
        assert mem.memory_type == "code"
        assert mem.compaction_source == "ast"
        assert mem.code_payload is not None
        assert mem.text_payload is None  # Should not be set for code type

    def test_text_memory_envelope(self, valid_unified_text_memory):
        """A text-type UnifiedMemory with an rlm source should parse correctly."""
        mem = UnifiedMemory(**valid_unified_text_memory)
        assert mem.memory_type == "text"
        assert mem.compaction_source == "rlm"
        assert mem.text_payload is not None
        assert mem.code_payload is None  # Should not be set for text type

    def test_id_auto_generated(self, valid_unified_code_memory):
        """If id is not provided, a UUID should be auto-generated."""
        mem = UnifiedMemory(**valid_unified_code_memory)
        assert mem.id is not None
        assert len(mem.id) == 36  # Standard UUID4 format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

    def test_timestamp_auto_generated(self, valid_unified_code_memory):
        """If timestamp is not provided, it should default to current UTC time."""
        mem = UnifiedMemory(**valid_unified_code_memory)
        assert mem.timestamp is not None
        assert mem.timestamp.tzinfo == timezone.utc

    def test_vector_embedding_id_defaults_none(self, valid_unified_code_memory):
        """vector_embedding_id should be None before T1 sets it after storage."""
        mem = UnifiedMemory(**valid_unified_code_memory)
        assert mem.vector_embedding_id is None


class TestTokenStatePositive:
    """Positive tests for the TokenState model."""

    def test_minimal_valid_token_state(self, valid_token_state):
        """A TokenState with just session_id and token_limit should parse."""
        state = TokenState(**valid_token_state)
        assert state.session_id == "session-abc-123"
        assert state.token_limit == 128000
        assert state.current_token_count == 0

    def test_defaults(self, valid_token_state):
        """Verify all default values are set correctly."""
        state = TokenState(**valid_token_state)
        assert state.compaction_threshold == 0.80
        assert state.context_chunks == []
        assert state.oldest_chunk_type == "text"
        assert state.compaction_in_progress is False
        assert state.total_compactions == 0


# ===========================================================================
# 2. NEGATIVE TESTS — invalid data must raise ValidationError
# ===========================================================================

class TestCodeSymbolNegative:
    """Negative tests for CodeSymbol."""

    def test_missing_name_raises(self):
        """Omitting 'name' (required) must raise ValidationError."""
        with pytest.raises(ValidationError):
            CodeSymbol(type="function", signature="def foo():", start_line=1)

    def test_invalid_type_raises(self):
        """An unrecognised symbol type must raise ValidationError."""
        with pytest.raises(ValidationError):
            CodeSymbol(name="foo", type="decorator", signature="@foo", start_line=1)

    def test_start_line_zero_raises(self):
        """start_line must be >= 1 (1-indexed). Zero should raise ValidationError."""
        with pytest.raises(ValidationError):
            CodeSymbol(name="foo", type="function", signature="def foo():", start_line=0)


class TestCodeMemoryNegative:
    """Negative tests for CodeMemorySchema."""

    def test_missing_file_path_raises(self):
        """Omitting 'file_path' (required) must raise ValidationError."""
        with pytest.raises(ValidationError):
            CodeMemorySchema(language="python", original_token_count=100)

    def test_missing_language_raises(self):
        """Omitting 'language' (required) must raise ValidationError."""
        with pytest.raises(ValidationError):
            CodeMemorySchema(file_path="foo.py", original_token_count=100)

    def test_negative_token_count_raises(self):
        """original_token_count must be >= 0. Negative values should raise."""
        with pytest.raises(ValidationError):
            CodeMemorySchema(
                file_path="foo.py",
                language="python",
                original_token_count=-1,
            )


class TestTextMemoryNegative:
    """Negative tests for TextMemorySchema."""

    def test_missing_factual_summary_raises(self):
        """Omitting 'factual_summary' (required) must raise ValidationError."""
        with pytest.raises(ValidationError):
            TextMemorySchema(
                chunk_index=0,
                total_chunks=1,
                timeframe="Full session",
                original_token_count=1000,
            )

    def test_chunk_index_out_of_bounds_raises(self):
        """chunk_index >= total_chunks must raise ValidationError (model_validator)."""
        with pytest.raises(ValidationError):
            TextMemorySchema(
                chunk_index=3,   # Invalid: index 3 is out of bounds for 3 total chunks
                total_chunks=3,
                timeframe="Session",
                factual_summary="Summary.",
                original_token_count=1000,
            )

    def test_total_chunks_zero_raises(self):
        """total_chunks must be >= 1. Zero should raise ValidationError."""
        with pytest.raises(ValidationError):
            TextMemorySchema(
                chunk_index=0,
                total_chunks=0,  # Invalid: must be at least 1
                timeframe="Session",
                factual_summary="Summary.",
                original_token_count=1000,
            )


class TestUnifiedMemoryNegative:
    """Negative tests for UnifiedMemory cross-field validation."""

    def test_code_type_with_rlm_source_raises(self, valid_code_memory):
        """memory_type='code' + compaction_source='rlm' must raise ValidationError."""
        with pytest.raises(ValidationError):
            UnifiedMemory(
                session_id="s1",
                memory_type="code",
                compaction_source="rlm",  # Inconsistent with memory_type='code'
                code_payload=valid_code_memory,
                raw_token_count=100,
            )

    def test_code_type_without_code_payload_raises(self):
        """memory_type='code' without a code_payload must raise ValidationError."""
        with pytest.raises(ValidationError):
            UnifiedMemory(
                session_id="s1",
                memory_type="code",
                compaction_source="ast",
                code_payload=None,  # Missing required payload for code type
                raw_token_count=100,
            )

    def test_text_type_with_ast_source_raises(self, valid_text_memory):
        """memory_type='text' + compaction_source='ast' must raise ValidationError."""
        with pytest.raises(ValidationError):
            UnifiedMemory(
                session_id="s1",
                memory_type="text",
                compaction_source="ast",  # Inconsistent with memory_type='text'
                text_payload=valid_text_memory,
                raw_token_count=100,
            )

    def test_missing_session_id_raises(self, valid_unified_code_memory):
        """Omitting 'session_id' (required) must raise ValidationError."""
        valid_unified_code_memory.pop("session_id")
        with pytest.raises(ValidationError):
            UnifiedMemory(**valid_unified_code_memory)


# ===========================================================================
# 3. PROPERTY TESTS — computed properties on TokenState
# ===========================================================================

class TestTokenStateProperties:
    """Tests for the computed properties on the TokenState model."""

    def test_utilization_at_zero(self, valid_token_state):
        """At 0 tokens used, utilization should be 0.0."""
        state = TokenState(**valid_token_state)
        assert state.utilization == 0.0

    def test_utilization_at_80_percent(self, valid_token_state):
        """At exactly 80% of token_limit, utilization should equal threshold."""
        state = TokenState(**valid_token_state)
        state.current_token_count = int(128000 * 0.80)  # = 102400
        assert abs(state.utilization - 0.80) < 1e-9  # Float equality with tolerance

    def test_utilization_at_full(self, valid_token_state):
        """At full token_limit, utilization should be 1.0."""
        state = TokenState(**valid_token_state)
        state.current_token_count = 128000
        assert state.utilization == 1.0

    def test_should_compact_false_below_threshold(self, valid_token_state):
        """Below the 80% threshold, should_compact must be False."""
        state = TokenState(**valid_token_state)
        state.current_token_count = 50000  # ~39%, well below threshold
        assert state.should_compact is False

    def test_should_compact_true_at_threshold(self, valid_token_state):
        """At exactly the threshold, should_compact must be True."""
        state = TokenState(**valid_token_state)
        state.current_token_count = int(128000 * 0.80)  # Exactly 80%
        assert state.should_compact is True

    def test_should_compact_false_when_in_progress(self, valid_token_state):
        """Even above threshold, should_compact must be False if a compaction is running."""
        state = TokenState(**valid_token_state)
        state.current_token_count = 120000  # Above 80%
        state.compaction_in_progress = True  # Guard flag is set
        assert state.should_compact is False  # Must not trigger again


# ===========================================================================
# 4. SERIALIZATION TESTS — ensure models round-trip to/from JSON cleanly
# ===========================================================================

class TestSerialization:
    """Tests that models serialize to JSON and deserialize back correctly."""

    def test_code_memory_json_roundtrip(self, valid_code_memory):
        """CodeMemorySchema should survive a JSON serialize/deserialize round-trip."""
        original = CodeMemorySchema(**valid_code_memory)
        json_str = original.model_dump_json()
        restored = CodeMemorySchema.model_validate_json(json_str)
        assert restored.file_path == original.file_path
        assert len(restored.symbols) == len(original.symbols)

    def test_unified_memory_json_roundtrip(self, valid_unified_text_memory):
        """UnifiedMemory should survive a JSON serialize/deserialize round-trip."""
        original = UnifiedMemory(**valid_unified_text_memory)
        json_str = original.model_dump_json()
        restored = UnifiedMemory.model_validate_json(json_str)
        assert restored.id == original.id
        assert restored.session_id == original.session_id
        assert restored.text_payload.factual_summary == original.text_payload.factual_summary

    def test_compaction_route_enum_values(self):
        """CompactionRoute enum should have the expected string values."""
        assert CompactionRoute.AST == "ast"
        assert CompactionRoute.RLM == "rlm"
