"""Calibrate the contradiction check (final round, task 3): P(yes) from the real
model on 12 contradicting and 12 agreeing same-topic pairs.
Run from the repo root:  .venv/bin/python -I report_eval/fact_conflicts/calibrate.py
"""
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.getcwd())
from harness import conflicts  # noqa: E402

CONTRADICT = [
    ("We use PostgreSQL for the vault database.", "Correction: the vault now uses MongoDB. Always say MongoDB."),
    ("The demo is on 14 December in room LHC-104.", "The demo moved to room B-12; the old room is wrong."),
    ("AMP Chat starts with a token limit of 3072 tokens.", "The default token limit is 50000 tokens; 3072 was a typo."),
    ("The user's cat is called Pixel.", "The user's cat is called Whiskers."),
    ("The project deadline is 20 November.", "The project deadline is 5 December."),
    ("Compaction starts at 80% of the token limit.", "Compaction starts at 95% of the token limit."),
    ("The team chose FastMCP for the memory server.", "The memory server is written with Flask, not FastMCP."),
    ("The embedding model is all-MiniLM-L6-v2.", "The vault embeds text with OpenAI text-embedding-3-large."),
    ("Aman owns the guardrails module.", "Tushar owns the guardrails module, not Aman."),
    ("The chat model is llama3.1:8b running locally.", "The chat model is GPT-4o through the OpenAI API."),
    ("The user's codename is ZEBRAFISH.", "The user's codename is PELICAN."),
    ("Sessions are encrypted with AES-256-GCM.", "Sessions are stored in plain text; there is no encryption."),
]
AGREE = [
    ("We use PostgreSQL for the vault database.", "The vault database is PostgreSQL 16 with default settings."),
    ("The demo is on 14 December in room LHC-104.", "The demo on 14 December starts at 10 am."),
    ("AMP Chat starts with a token limit of 3072 tokens.", "Compaction starts at 80% of the token limit."),
    ("The user's cat is called Pixel.", "Pixel, the user's cat, is three years old."),
    ("The project deadline is 20 November.", "The report for the project is due with the code on 20 November."),
    ("Compaction starts at 80% of the token limit.", "Background compaction stops at 70% of the window."),
    ("The team chose FastMCP for the memory server.", "The memory server runs FastMCP over streamable HTTP on port 8001."),
    ("The embedding model is all-MiniLM-L6-v2.", "ChromaDB's default embedding function is used for the vault."),
    ("Aman owns the guardrails module.", "Aman added Layer 4, the tool guard, to the guardrails module."),
    ("The chat model is llama3.1:8b running locally.", "Ollama serves the chat model on localhost:11434."),
    ("The user's codename is ZEBRAFISH.", "The user asked the assistant to remember their codename."),
    ("Sessions are encrypted with AES-256-GCM.", "Each session's key lives in storage/.amp_session_keys."),
]

model = conflicts.model_name()
conflicts._ask(model, *CONTRADICT[0])                     # warm the model
rows, times = [], []
for label, pairs in (("contradict", CONTRADICT), ("agree", AGREE)):
    for older, newer in pairs:
        started = time.perf_counter()
        p = conflicts._ask(model, older, newer)
        times.append(time.perf_counter() - started)
        rows.append({"label": label, "older": older, "newer": newer, "p_yes": round(p, 3)})
for threshold in (0.5, 0.6, 0.7, 0.8, 0.9):
    caught = sum(r["p_yes"] >= threshold for r in rows if r["label"] == "contradict")
    false = sum(r["p_yes"] >= threshold for r in rows if r["label"] == "agree")
    print(f"threshold {threshold}: contradictions caught {caught}/12, agreeing pairs marked {false}/12")
print("contradict:", [r["p_yes"] for r in rows if r["label"] == "contradict"])
print("agree:     ", [r["p_yes"] for r in rows if r["label"] == "agree"])
print(f"seconds per check: median {statistics.median(times):.2f}, max {max(times):.2f}")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")
json.dump({"model": model, "system_prompt": conflicts.SYSTEM, "rows": rows,
           "median_seconds": statistics.median(times)}, open(out, "w"), indent=1)
