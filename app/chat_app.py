"""
app/chat_app.py - AMP Chat: a local chatbot with paged memory
=============================================================
Run from the project root:

    streamlit run app/chat_app.py

Left: the chat. Right: what the memory pager does behind the scenes, turn by
turn: how full the context window is, which chunks were compacted (code by the
AST compactor, dialogue by the RLM summarizer) and stored in the ChromaDB
vault, and which memories were paged back in for each answer.

Everything runs locally: the chat model and the summarizer through Ollama,
the vault in storage/chroma_db.
"""
import html
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st  # noqa: E402

from harness.chat_model import ChatModelError, model_status  # noqa: E402
from harness.chat_session import (  # noqa: E402
    DEFAULT_TOKEN_LIMIT, MAX_TOKEN_LIMIT, ChatSession, MessageTooLong,
)

ICONS = {"monitor": "📊", "dispatch": "🔀", "compaction": "🗜️", "search": "🔎",
         "page_in": "📥", "reply": "💬", "error": "⚠️"}

st.set_page_config(page_title="AMP Chat", page_icon="🧠", layout="wide")


@st.cache_resource
def get_graph():
    from harness.graph import create_amp_graph
    return create_amp_graph()


def new_session(limit: int, threshold: float) -> None:
    st.session_state.session = ChatSession(token_limit=limit, threshold=threshold, graph=get_graph())
    st.session_state.error = None


def context_bar(used: int, limit: int, threshold: float) -> str:
    """HTML bar: context fill, with a marker where compaction starts."""
    fill = min(100.0, used / limit * 100)
    colour = "#d9534f" if used >= limit * threshold else "#4a7bd0"
    return f"""
<div style="position:relative;height:18px;background:#e9ecef;border-radius:9px;overflow:hidden;margin:4px 0 2px 0">
  <div style="width:{fill:.1f}%;height:100%;background:{colour}"></div>
  <div style="position:absolute;top:0;left:{threshold * 100:.0f}%;width:2px;height:100%;background:#333"></div>
</div>
<div style="font-size:0.85em;color:#555">{used:,} / {limit:,} tokens ({used / limit:.0%}) · compaction starts at
{int(limit * threshold):,} (the black line)</div>"""


def draw_memory(memory: dict, *, distance: float | None = None) -> None:
    label = f"{memory['engine']} · {memory['label']} · {memory['tokens']} tokens"
    if distance is not None:
        label += f" · distance {distance:.2f}"
    with st.expander(label):
        st.markdown(html.escape(memory["text"]) or "_empty_")
        for name in ("decisions", "bugs", "constraints"):
            if memory.get(name):
                st.markdown(f"**{name.capitalize()}**")
                for item in memory[name]:
                    st.markdown(f"- {html.escape(item)}")


def draw_panel(session: ChatSession) -> None:
    st.subheader("Behind the scenes")
    st.markdown(context_bar(session.context_tokens, session.token_limit, session.threshold),
                unsafe_allow_html=True)
    if session.reclaiming:
        st.info("🗜️ Compacting old messages in the background…")
    code = sum(1 for c in session.chunks if c.get("type") == "code")
    st.caption(f"{len(session.chunks)} chunks in the context window "
               f"({len(session.chunks) - code} text, {code} code)")

    left, middle, right = st.columns(3)
    left.metric("Compactions", session.compactions)
    middle.metric("Tokens paged out", f"{session.tokens_evicted:,}")
    right.metric("Stored as", f"{session.tokens_stored:,}",
                 help="Tokens of the compact memories that replaced the paged-out chunks")

    st.markdown("**Paged in for the last answer**")
    if session.last_paged:
        for memory in session.last_paged:
            draw_memory(memory, distance=memory.get("distance"))
    else:
        st.caption("Nothing yet: the vault fills up once the context window passes the trigger.")

    st.markdown("**Activity**")
    if not session.events:
        st.caption("Each step of a turn shows up here.")
    for turn in sorted({e.turn for e in session.events}, reverse=True)[:8]:
        st.caption(f"Turn {turn}")
        for event in (e for e in session.events if e.turn == turn):
            took = f" · {event.seconds:.1f}s" if event.seconds >= 0.05 else ""
            line = f"{ICONS.get(event.kind, '•')} {event.title}{took}"
            if event.kind == "compaction" and event.data.get("memory"):
                with st.expander(line):
                    st.markdown(html.escape(event.detail) or "_empty_")
            else:
                st.markdown(line + (f"  \n<span style='color:#666;font-size:0.85em'>{html.escape(event.detail[:300])}</span>"
                                    if event.detail else ""), unsafe_allow_html=True)

    vault = session.vault()
    with st.expander(f"Vault: {len(vault)} memories in this chat"):
        for memory in vault:
            draw_memory(memory)


# ---------------------------------------------------------------------------
# Sidebar: model status and settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🧠 Agentic Memory Pager")
    st.caption("A local chatbot whose context window works like RAM. When it fills up, the oldest "
               "messages are compressed (code by AST, dialogue by an RLM summarizer), stored in a "
               "ChromaDB vault, and paged back in when a question needs them.")
    ready, status = model_status()
    (st.success if ready else st.error)(status)
    limit = st.slider("Context window (tokens)", 1024, MAX_TOKEN_LIMIT, DEFAULT_TOKEN_LIMIT, step=256,
                      help="Kept small so paging is easy to see. The model itself runs with 8,192 tokens.")
    threshold = st.slider("Compact when the window is this full", 50, 95, 80, step=5, format="%d%%") / 100
    background = st.toggle("Compact in the background", value=True,
                           help="On: old messages are compressed right after each answer, while you read. "
                                "Off: they are compressed before the next answer, inside the turn.")

    if "session" not in st.session_state:
        new_session(limit, threshold)
    session: ChatSession = st.session_state.session
    session.token_limit, session.threshold, session.background = limit, threshold, background

    if st.button("New chat", use_container_width=True):
        new_session(limit, threshold)
        st.rerun()
    st.caption(f"Chat id `{session.session_id}`")


# ---------------------------------------------------------------------------
# Main area: chat on the left, the pager on the right
# ---------------------------------------------------------------------------

chat_col, panel_col = st.columns([3, 2], gap="large")

with chat_col:
    st.title("AMP Chat")
    for entry in session.transcript:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            where = session.message_status(entry["id"])
            if where != "in context":
                st.caption(f"📦 {where}: compressed into the vault")
    if st.session_state.get("error"):
        st.error(st.session_state.error)
    prompt = st.chat_input("Message the local model…", disabled=not ready)


@st.fragment(run_every=1.5)
def live_panel() -> None:
    """The panel, refreshed while background compaction runs; one full rerun when it ends
    so the chat's "paged out" labels update too."""
    draw_panel(session)
    if not session.reclaiming:
        st.session_state.watching = False
        st.rerun(scope="app")


if prompt:
    st.session_state.error = None
    with chat_col:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.status("Working behind the scenes…", expanded=True) as box:
                def show(event):
                    box.write(f"{ICONS.get(event.kind, '•')} {event.title}")
                try:
                    session.ask(prompt, on_event=show)
                    box.update(label="Done", state="complete", expanded=False)
                except (MessageTooLong, ChatModelError) as exc:
                    box.update(label="Could not answer", state="error")
                    st.session_state.error = str(exc)
    st.rerun()

# The panel is drawn last: a pending message is handled first, so the panel's
# refresh (when background compaction ends) can never swallow it.
with panel_col:
    if session.reclaiming or st.session_state.get("watching"):
        st.session_state.watching = True
        live_panel()
    else:
        draw_panel(session)
