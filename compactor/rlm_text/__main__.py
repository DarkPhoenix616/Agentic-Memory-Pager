"""
Distil a conversation file into a fact map from the command line.

    python -m compactor.rlm_text transcript.txt
    python -m compactor.rlm_text transcript.txt --json > memory.json
    cat chat.txt | python -m compactor.rlm_text - --model qwen3:8b

The transcript is plain text with one "User:" / "Assistant:" marker per
message. Needs a running Ollama server unless --no-llm is given.
"""
import argparse
import logging
import sys

from compactor.rlm_text.recursive_summarizer import RLMConfig, distill


def _print_list(title, items):
    print(f"\n{title} ({len(items)})")
    for item in items:
        print(f"  - {item}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m compactor.rlm_text",
        description="Distil a conversation into a TextMemorySchema fact map.",
    )
    parser.add_argument("file", help="transcript file, or - to read stdin")
    parser.add_argument("--role", help="who wrote the text if it has no User:/Assistant: markers")
    parser.add_argument("--model", help="Ollama model (default: $AMP_RLM_MODEL or llama3.1:8b)")
    parser.add_argument("--piece-tokens", type=int, default=2000, help="tokens per sub-LLM call")
    parser.add_argument("--summary-words", type=int, default=300, help="length of the final summary")
    parser.add_argument("--workers", type=int, default=4, help="sub-LLM calls in parallel")
    parser.add_argument("--no-llm", action="store_true", help="rule-based extractor only")
    parser.add_argument("--pieces", action="store_true", help="also print the fact map of every piece")
    parser.add_argument("--json", action="store_true", help="print the TextMemorySchema as JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="log every step")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    if args.file == "-":
        text = sys.stdin.read()
    else:
        with open(args.file, encoding="utf-8", errors="replace") as handle:
            text = handle.read()

    options = dict(piece_tokens=args.piece_tokens, summary_words=args.summary_words,
                   max_workers=args.workers, use_llm=not args.no_llm)
    if args.model:
        options["model"] = args.model
    try:
        config = RLMConfig(**options)
    except ValueError as exc:
        parser.error(str(exc))

    result = distill(text, role=args.role, config=config)
    memory = result.memory

    if args.json:
        print(memory.model_dump_json(indent=2))
        return 0

    print(f"Timeframe: {memory.timeframe}")
    print(f"\nSummary\n  {memory.factual_summary}")
    _print_list("Architectural decisions", memory.architectural_decisions)
    _print_list("Bugs resolved", memory.bugs_resolved)
    _print_list("User constraints", memory.user_constraints)

    if args.pieces:
        for number, (piece, fact) in enumerate(zip(result.pieces, result.piece_maps), 1):
            print(f"\n--- Piece {number}: messages {piece.first}-{piece.last}, "
                  f"{piece.tokens} tokens, {fact.method}")
            print(f"  {fact.summary}")

    print(f"\n{memory.original_token_count:,} -> {result.summary_tokens:,} tokens "
          f"({result.compression:.1%}) | {len(result.pieces)} piece(s), "
          f"{result.reduce_levels} reduce level(s), {result.llm_calls} LLM call(s), "
          f"{result.fallbacks} fallback(s) | {result.seconds:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
