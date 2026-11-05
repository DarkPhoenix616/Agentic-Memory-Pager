"""
compactor/rlm_text/ - RLM Text Compactor
========================================
    summarize_text(text, role, original_token_count) -> TextMemorySchema
    distill(text, ...) -> RLMResult (memory plus run statistics)
    split_into_pieces(text, piece_tokens) -> List[Piece]

Command line:
    python -m compactor.rlm_text transcript.txt
"""
from compactor.rlm_text.recursive_summarizer import (
    FactMap,
    Piece,
    RLMConfig,
    RLMResult,
    Turn,
    distill,
    key_terms,
    merge_items,
    ollama_llm,
    rule_based_facts,
    split_into_pieces,
    split_turns,
    summarize_text,
)

__all__ = [
    "FactMap", "Piece", "RLMConfig", "RLMResult", "Turn",
    "distill", "key_terms", "merge_items", "ollama_llm", "rule_based_facts",
    "split_into_pieces", "split_turns", "summarize_text",
]
