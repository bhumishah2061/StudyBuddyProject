"""
capstone_streamlit.py – Study Buddy Physics Streamlit UI.

Features
--------
- @st.cache_resource for LLM, embedder, ChromaDB collection, and compiled graph.
- st.session_state for messages and thread_id.
- Sidebar: domain description, topics covered, New Conversation button.
- Persistent MemorySaver conversation across reruns within a session.
- Source attribution for retrieved answers.
- encoding='utf-8' enforced for any file operations.

Run: streamlit run capstone_streamlit.py
"""

from __future__ import annotations

import os
import uuid
import sys

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Study Buddy Physics",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CACHED RESOURCE LOADERS
# (Each runs exactly once per Streamlit session; heavy objects are reused.)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading language model…")
def load_llm():
    """Cache the ChatAnthropic LLM singleton."""
    from nodes import LLM
    return LLM


@st.cache_resource(show_spinner="Loading sentence embedder…")
def load_embedder():
    """Cache the SentenceTransformer embedder."""
    from nodes import EMBEDDER
    return EMBEDDER


@st.cache_resource(show_spinner="Loading physics knowledge base…")
def load_collection():
    """Cache the ChromaDB collection."""
    from nodes import collection
    return collection


@st.cache_resource(show_spinner="Compiling reasoning graph…")
def load_graph():
    """
    Import and cache the compiled LangGraph application.
    This triggers module-level initialisation in nodes.py and graph.py
    (LLM, embedder, ChromaDB, smoke-tests, compilation).
    """
    from graph import app
    return app


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — send_question()
# ─────────────────────────────────────────────────────────────────────────────

def send_question(question: str, thread_id: str) -> dict:
    """
    Invoke the Study Buddy Physics agent and return the structured response.
    Preserves conversation memory within the session via MemorySaver checkpoints.
    """
    compiled_app = load_graph()
    config = {"configurable": {"thread_id": thread_id}}

    # Restore persisted state from checkpoint
    current_messages: list = []
    current_user_name: str = ""
    try:
        snapshot = compiled_app.get_state(config)
        if snapshot and snapshot.values:
            current_messages = list(snapshot.values.get("messages") or [])
            current_user_name = str(snapshot.values.get("user_name") or "")
    except Exception:
        pass

    input_state = {
        "question": question,
        "messages": current_messages,
        "route": "",
        "retrieved": "",
        "sources": [],
        "tool_result": "",
        "answer": "",
        "faithfulness": 0.0,
        "eval_retries": 0,
        "user_name": current_user_name,
    }

    result = compiled_app.invoke(input_state, config=config)

    return {
        "answer": result.get("answer", ""),
        "route": result.get("route", ""),
        "sources": result.get("sources", []),
        "faithfulness": result.get("faithfulness", 0.0),
        "user_name": result.get("user_name", ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"session_{uuid.uuid4().hex[:12]}"

if "messages" not in st.session_state:
    # Each entry: {"role": "user"|"assistant", "content": str, "meta": dict|None}
    st.session_state.messages = []

if "user_name" not in st.session_state:
    st.session_state.user_name = ""

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Rutherford_atom.svg/240px-Rutherford_atom.svg.png",
        width=120,
        caption="",
    )
    st.title("⚛️ Study Buddy Physics")

    st.markdown("---")
    st.subheader("📚 About")
    st.markdown(
        """
**Study Buddy Physics** is an AI-powered tutoring assistant for B.Tech students.

It provides grounded, faithful explanations from a curated physics knowledge base,
remembers your name and conversation context, and never fabricates formulas.
        """
    )

    st.markdown("---")
    st.subheader("🗂️ Topics Covered")
    topics = [
        "🔵 Newton's Laws",
        "⚡ Work, Energy & Power",
        "🌍 Gravitation",
        "🔄 Oscillations (SHM)",
        "〰️ Waves",
        "🌡️ Thermodynamics",
        "⚡ Electrostatics",
        "🧲 Magnetism",
        "🔭 Optics",
        "⚛️ Quantum Basics",
    ]
    for t in topics:
        st.markdown(f"- {t}")

    st.markdown("---")
    st.subheader("🛠️ Tools Available")
    st.markdown("- 🧮 **Calculator** — numerical physics problems\n- 🕐 **Datetime** — current date/time")

    st.markdown("---")
    if st.button("🔄 New Conversation", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.session_state.thread_id = f"session_{uuid.uuid4().hex[:12]}"
        st.session_state.user_name = ""
        st.rerun()

    st.markdown("---")
    st.caption(
        f"Session ID: `{st.session_state.thread_id[:16]}…`\n\n"
        "Built with LangGraph · ChromaDB · SentenceTransformers"
    )

# ─────────────────────────────────────────────────────────────────────────────
# MAIN CHAT AREA
# ─────────────────────────────────────────────────────────────────────────────

# Header
col_title, col_status = st.columns([3, 1])
with col_title:
    greeting = (
        f"Hello, {st.session_state.user_name}! " if st.session_state.user_name else ""
    )
    st.title(f"⚛️ {greeting}Study Buddy Physics")
    st.caption(
        "Ask me anything about Mechanics, Thermodynamics, Electromagnetism, Optics, "
        "or Quantum Basics. I'll only answer from my physics knowledge base — never guessing."
    )

with col_status:
    st.markdown("<br>", unsafe_allow_html=True)
    st.success("🟢 Agent Ready")

st.divider()

# ── Display existing conversation ─────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        st.info(
            "👋 Welcome! You can ask me physics questions like:\n"
            "- *Explain Newton's second law*\n"
            "- *What is Snell's law?*\n"
            "- *Calculate KE for mass 2 kg at 5 m/s*\n"
            "- *My name is Priya* (I'll remember it!)"
        )
    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

                # Show metadata for assistant messages
                if msg["role"] == "assistant" and msg.get("meta"):
                    meta = msg["meta"]
                    cols = st.columns([2, 2, 2])
                    with cols[0]:
                        route_emoji = {"retrieve": "📚", "tool": "🛠️", "skip": "💬"}.get(
                            meta.get("route", ""), "❓"
                        )
                        st.caption(f"{route_emoji} Route: **{meta.get('route', 'N/A')}**")
                    with cols[1]:
                        faith = meta.get("faithfulness", 0.0)
                        faith_color = "🟢" if faith >= 0.7 else "🟡"
                        st.caption(f"{faith_color} Faithfulness: **{faith:.2f}**")
                    with cols[2]:
                        sources = meta.get("sources", [])
                        if sources:
                            st.caption(f"📖 Sources: {', '.join(sources[:2])}")

# ── Chat input ────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
user_input = st.chat_input(
    placeholder="Ask a physics question… (e.g. 'What is Planck's constant?')"
)

if user_input and user_input.strip():
    question = user_input.strip()

    # Display user message immediately
    st.session_state.messages.append({"role": "user", "content": question, "meta": None})

    with st.chat_message("user"):
        st.markdown(question)

    # ── Get agent response ────────────────────────────────────────────────────
    with st.chat_message("assistant"):
        with st.spinner("🔍 Thinking…"):
            try:
                # Pre-load resources (cached after first call)
                load_llm()
                load_embedder()
                load_collection()

                result = send_question(question, st.session_state.thread_id)

                answer = result["answer"]
                route = result["route"]
                sources = result["sources"]
                faith = result["faithfulness"]
                user_name = result.get("user_name", "")

                # Update user_name in session state if extracted
                if user_name and user_name != st.session_state.user_name:
                    st.session_state.user_name = user_name

                # Display answer
                st.markdown(answer)

                # Display metadata ribbon
                meta_cols = st.columns([2, 2, 2])
                with meta_cols[0]:
                    route_emoji = {"retrieve": "📚", "tool": "🛠️", "skip": "💬"}.get(route, "❓")
                    st.caption(f"{route_emoji} Route: **{route}**")
                with meta_cols[1]:
                    faith_color = "🟢" if faith >= 0.7 else "🟡"
                    st.caption(f"{faith_color} Faithfulness: **{faith:.2f}**")
                with meta_cols[2]:
                    if sources:
                        st.caption(f"📖 Sources: {', '.join(sources[:2])}")

                # Save to session state
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "meta": {
                            "route": route,
                            "faithfulness": faith,
                            "sources": sources,
                        },
                    }
                )

            except Exception as exc:
                error_msg = (
                    f"⚠️ An error occurred while generating the response:\n\n"
                    f"```\n{exc}\n```\n\n"
                    "Please check your ANTHROPIC_API_KEY and try again."
                )
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg, "meta": None}
                )

    # Rerun to refresh the chat display
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "⚠️ Study Buddy Physics answers only from its curated knowledge base. "
    "For exam-critical details, always verify with your professor or textbook. "
    "| Built with ❤️ using LangGraph · ChromaDB · Anthropic Claude · Streamlit"
)
