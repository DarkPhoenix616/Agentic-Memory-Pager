"""
integrations/claude_code/ - AMP memory for Claude Code
======================================================
Claude Code keeps one context window per conversation. When it fills up,
Claude Code compacts it: the whole conversation becomes one summary and the
details are gone from the window. This package gives Claude Code AMP's
long-term memory, without changing Claude Code itself:

    page out   hooks save the conversation into the AMP vault before details
               are lost: after replies (once enough is new), just before a
               compaction, and when the session ends. Text goes through the
               RLM summarizer, code through the AST compactor, both are risk
               scored and signed, and the memory server stores them encrypted
               with the project's key.
    page in    before every prompt, a hook asks the memory server for the
               memories that match the prompt (page_context: 1,000-token
               budget, relevance cut-off) and adds them to Claude's context.
    recall     an MCP tool, so Claude can also search the vault itself.

Memory is per project folder: every Claude Code session in the same folder
shares one AMP session, so what was decided yesterday is there today.

    project.py        settings, the project's AMP session and key, saved progress
    transcript.py     reads Claude Code's conversation file (JSONL)
    store.py          page out: turns -> RLM / AST memories -> vault
    page_in.py        page in: prompt -> matching memories -> context text
    hooks.py          the hook entry point Claude Code runs
    recall_server.py  the "recall" MCP tool (stdio)
    install.py        adds / removes the hooks and the tool in a project
"""
