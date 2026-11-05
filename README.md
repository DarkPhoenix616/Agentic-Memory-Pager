```markdown
# Agentic Memory Pager (AMP) 🧠⚡
> **A Virtual Memory Paging & Long-Term Context Server for AI Agents**

---

## 📌 Executive Summary

Modern AI coding agents (Claude Code, Cursor, Aider) suffer from **Context Rot** and **State Amnesia**. When long sessions exceed the model's token limit, current systems perform **lossy compaction**—summarizing the chat and permanently discarding raw logs, exact variable names, and early architectural decisions.

**Agentic Memory Pager (AMP)** solves this by introducing **Virtual Memory Paging** for LLMs (inspired by OS paging):
- **Short-Term Context (RAM):** The model's active working context window.
- **Long-Term Storage (Disk):** An external Vector Database indexed via the **Model Context Protocol (MCP)**.
- **Dual Compaction Engine:** Uses **AST-aware parsing** for code and **Recursive Language Models (RLMs)** for dialogue to compress data without losing granular details.

---

## 🏗️ System Architecture

```text
                       ┌────────────────────────┐
                       │  User / Agent Runtime  │
                       └───────────┬────────────┘
                                   │ (Token Count Monitored)
                                   ▼
                   ┌────────────────────────────────┐
                   │   LangGraph Agent Harness      │
                   │   - Token Tracker (80% Limit)  │
                   │   - Dynamic Router             │
                   └───────┬────────────────┬───────┘
                           │                │
            [If Code]      │                │  [If Text/Dialogue]
                           ▼                ▼
             ┌──────────────────┐     ┌──────────────────┐
             │ AST Code Parser  │     │   RLM Engine     │
             │  (Tree-sitter)   │     │ (Sub-LLM Batches)│
             └─────────┬────────┘     └─────────┬────────┘
                       │                        │
                       └───────────┬────────────┘
                                   │ Pydantic / Instructor Validation
                                   ▼
             ┌───────────────────────────────────────────┐
             │       FastMCP Memory Server (Local)       │
             │   Tools: store_memory(), search_vault()   │
             └─────────────────────┬─────────────────────┘
                                   ▼
             ┌───────────────────────────────────────────┐
             │   ChromaDB (Vectors) + SQLite (Metadata)  │
             └───────────────────────────────────────────┘
```

---

## 👥 Module Breakdown & Team Responsibilities

### 🔹 Teammate 1: Protocol & Sandbox Engineer
* **Core Focus:** MCP Server architecture and secure execution environment for recursive queries.
* **Key Tasks:**
    1. Build the standalone **FastMCP Server** exposing standard tools (`store_memory`, `search_vault`, `page_context`).
    2. Implement an isolated **Python REPL Sandbox** where sub-agents can programmatically inspect stored context objects.
    3. Manage workspace-level session keys and encrypted local storage.
* **Deliverable:** A runnable MCP server exposing validated JSON-RPC endpoints to any MCP client.

---

### 🔹 Teammate 2: Code Compactor (AST Engineer)
* **Core Focus:** Syntax-aware code compression and structural mapping.
* **Key Tasks:**
    1. Implement **Tree-sitter** (or Python's native `ast`) to parse source code files.
    2. Build a pipeline that strips out function bodies/implementation details while preserving imports, class structures, type signatures, and docstrings (Tier-2 Structural Skeleton).
    3. Generate compact symbol dependency graphs for indexing into the Vector DB.
* **Deliverable:** A pipeline converting a 5,000-token source file into a lossless, 500-token structural index.

---

### 🔹 Teammate 3: Text Compactor (RLM Engineer)
* **Core Focus:** Conversational history partitioning and Recursive Language Model execution.
* **Key Tasks:**
    1. Implement **Recursive Delegation Logic** that splits massive conversation histories into contextual chunks.
    2. Set up parallel **Sub-LLM batching** using local models (e.g., `Phi-3-Mini` / `Llama-3-8B` via Ollama) to extract architectural decisions, bugs resolved, and user constraints.
    3. Prevent loss of semantic meaning during multi-pass compaction.
* **Deliverable:** A script that takes 50k tokens of conversational text and recursively distills it into structured fact maps.

---

### 🔹 Teammate 4: Harness Orchestrator & Guardrails
* **Core Focus:** State management, token tracking, validation guardrails, and telemetry.
* **Key Tasks:**
    1. Build the **LangGraph State Machine** managing the primary agent loop and monitoring context usage.
    2. Implement the **80% Compaction Trigger** that routes data to either the AST Compactor (Teammate 2) or RLM Compactor (Teammate 3).
    3. Integrate **Instructor + Pydantic** schemas to validate that all compacted outputs conform to strict JSON structures before DB ingestion.
    4. Build a lightweight **web UI** (`app/`) / CLI displaying live token counts and memory page-ins.
* **Deliverable:** The core execution harness that integrates all three sub-systems.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Protocol** | `FastMCP` (Python) | Model Context Protocol server implementation |
| **Orchestration** | `LangGraph` + `LiteLLM` | Stateful agent execution and multi-model routing |
| **Code Parsing** | `Tree-sitter` / `ast` | Abstract Syntax Tree extraction for code compaction |
| **Inference (Local)** | `Ollama` (`Llama-3`, `Phi-3`) | Free local execution for summarization & reasoning |
| **Storage** | `ChromaDB` + `SQLite` | Vector embeddings and structured relational metadata |
| **Guardrails** | `Instructor` + `Pydantic` | Deterministic schema validation and output bounding |
| **Observability** | `Langfuse` / web UI (`app/`) | Trace visualization and real-time memory monitoring |

---

## 📁 Repository Structure

```text
agentic-memory-pager/
├── docs/                   # Architecture diagrams and specifications
├── server/                 # [Teammate 1] FastMCP Server & REPL Sandbox
│   ├── mcp_server.py
│   └── sandbox.py
├── compactor/
│   ├── ast_code/           # [Teammate 2] Tree-sitter / AST Code Extraction
│   │   └── code_parser.py
│   └── rlm_text/           # [Teammate 3] Recursive Language Model Pipeline
│       └── recursive_summarizer.py
├── harness/                # [Teammate 4] LangGraph State Machine & Guardrails
│   ├── agent_loop.py
│   ├── guardrails.py
│   └── token_monitor.py
├── storage/                # ChromaDB & SQLite database handlers
│   └── memory_vault.py
├── app/                    # AMP Chat: web UI (static/) + server.py
│   └── app.py
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart (Week 1 Setup)

1. **Clone & Virtual Environment:**
   ```bash
   git clone https://github.com/<your-org>/agentic-memory-pager.git
   cd agentic-memory-pager
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Pull Local Models via Ollama:**
   ```bash
   ollama pull llama3:8b
   ollama pull phi3:mini
   ollama pull nomic-embed-text
   ```

3. **Run the MCP Server:**
   ```bash
   python server/mcp_server.py
   ```
```
---

## 💬 AMP Chat: run the whole system as a local chatbot

AMP Chat is a web app on top of the LangGraph harness. You chat with a local model. Behind the scenes, a side panel shows the memory pager at work:

- **Token meter:** how full the context window is, with a line at the 80% trigger.
- **Compactions:** each one is shown as it happens. Code blocks go to the **AST compactor**, and question-and-answer exchanges go to the **RLM summarizer**. The result is stored in the ChromaDB vault.
- **Paged-in memories:** which memories were paged back in to answer each question.

Old messages in the chat are marked *📦 paged out*. Ask about them later and the bot still answers, because the memory is paged back in.

Paging works like an operating system reclaiming memory:
- **Background reclaim:** after each answer, if the window passed 80%, the oldest chunks are compacted in a background thread, down to 70%, while you read.
- **Direct reclaim:** if a new message still pushes the window past 80%, the graph compacts inside the turn.
- **Working set:** the latest exchange is never paged out in the background.

```bash
ollama pull llama3.1:8b              # the chat model and the summarizer share it
pip install -r requirements.txt
python -m app.server                 # then open http://127.0.0.1:8000
```

It is plain HTML, CSS and JavaScript (`app/static/`), with no build step and no external libraries. The browser talks to a small Starlette server (`app/server.py`):

| Endpoint | Purpose |
|---|---|
| `POST /api/chat` | Send a message. The reply is a stream of the behind-the-scenes steps as they happen, then the answer. |
| `GET /api/state` | The chat, context meter, activity and vault. |
| `POST /api/settings` | Change the context window, the trigger and background compaction. |
| `POST /api/new` | Start a new chat. |

The server only listens on `127.0.0.1` and only accepts same-origin JSON, so other websites cannot use it.

Everything runs locally:
- **Chat and summaries:** through Ollama.
- **Search:** ChromaDB's built-in embedding model.
- **Vault:** stored in `storage/chroma_db/`.

The context window is small by default (3,072 tokens), so you can see paging after a few messages. You can change it in the sidebar.

| Variable | Default | Purpose |
|---|---|---|
| `AMP_CHAT_MODEL` | `llama3.1:8b` | chat model |
| `AMP_RLM_MODEL` | `llama3.1:8b` | RLM summarizer model |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server |

The core is `harness/chat_session.py` (`ChatSession.ask()`). It has no UI code, so it can also be used from scripts and tests.
