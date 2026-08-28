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
    4. Build a lightweight **Streamlit UI** / CLI displaying live token counts and memory page-ins.
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
| **Observability** | `Langfuse` / `Streamlit` | Trace visualization and real-time memory monitoring |

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
├── ui/                     # Streamlit observability dashboard
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