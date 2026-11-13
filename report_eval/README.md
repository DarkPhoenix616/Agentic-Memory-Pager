# report_eval: evaluation scripts and their saved results

Every number in `docs/AMP_Work_Report.md` comes from a script in this folder, and its raw output is saved next to it. Run the scripts from the **repo root**:
- **Live scripts** need Ollama with `llama3.1:8b`.
- **Folders:** each script uses its own temporary vault, keyring and chats folder, never `storage/`.

| Path | What it measures | Report section |
|---|---|---|
| `chat_recall_eval.py`, `chat_recall_eval_2048.*` | AMP vs a plain chatbot: 16 facts pushed out of a 2,048-token window | 5.3b |
| `make_chat.py`, `facts.json`, `recall.py`, `recall_text.py`, `run3/`, `rlm_run*` | RLM summarizer on a ~58k-token chat: compression and planted facts kept | 4 |
| `guard_eval/` | Risk model choice: word rules vs llama3.1:8b vs Llama Guard 3 on 15 attacks and 111 normal memories | 5b |
| `user_facts_eval/` | Facts the user states, kept in summaries (17/25 → 25/25) | 5d |
| `live_chat_history.py` | Leave a chat, reopen it, recall (real model) | 5e |
| `red_team.py`, `red_team_results*.json`, `red_team_run_*.log`, `red_team_v1_2026-11-10.py` | 13 attacks planted in the vault, against the real model | 5f, 5k |
| `auto_save/` | Ctrl+C and kill -9, then reopen and recall | 5j |
| `fact_conflicts/` | Threshold for the contradiction check (24 note pairs) | 5k |
| `layer4_live/` | The tool guard against planted tool calls, live (6 scenarios × 3 runs) | 5l |
| `final_check/` | Fresh clone and fresh venv, then a full chat in-process and over MCP | 5m |
| `chat_transcript/` | Reopened chats show their messages (with a screenshot) | 5n |
| `relevance_cutoff/` | The 16-fact chat without and with the relevance cut-off; vague questions | 5o |
| `code_graph/` | Cost and search value of call lines; call questions with the real model | 5p |
| `ui_check/` | The chat page in headless Chrome: greeting, New chat, reopening | 5q–5t |

Examples:

```bash
.venv/bin/python -I report_eval/red_team.py
.venv/bin/python -I report_eval/relevance_cutoff/recall_cutoff_eval.py B_cutoff_1p4
python3 report_eval/relevance_cutoff/compare.py report_eval/relevance_cutoff/{A_no_cutoff,B_cutoff_1p4}.json
```

Section 8 of the report lists the command for every result.
