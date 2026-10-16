"""
harness/ - LangGraph Agent Harness Package
==========================================
This package contains the core orchestration layer for the Agentic Memory Pager (AMP).

Modules:
    guardrails  : Pydantic data schemas shared across all AMP modules (data contracts).
    token_monitor: Token counting logic and the 80% compaction trigger.
    agent_loop  : LangGraph graph definition — the main agent state machine.
"""
