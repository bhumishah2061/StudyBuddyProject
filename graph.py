"""
graph.py – Assembles and compiles the Study Buddy Physics LangGraph pipeline.

Architecture
------------
User Question
  → memory_node
  → router_node
  → [conditional] retrieval_node | tool_node | skip_retrieval_node
  → answer_node
  → eval_node
  → [conditional] answer_node (retry if faithfulness < 0.7 and retries < MAX)
                | save_node
  → END
"""

from __future__ import annotations

# ── LangGraph imports ─────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END

try:
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:
    try:
        from langgraph.checkpoint import MemorySaver  # type: ignore[no-redef]
    except ImportError:
        raise ImportError(
            "MemorySaver not found. Please install langgraph>=0.2.0: "
            "pip install 'langgraph>=0.2.0'"
        )

# ── Project imports ───────────────────────────────────────────────────────────
from state import CapstoneState
from nodes import (
    MAX_EVAL_RETRIES,
    answer_node,
    eval_node,
    memory_node,
    retrieval_node,
    router_node,
    save_node,
    skip_retrieval_node,
    tool_node,
)


# =============================================================================
# CONDITIONAL EDGE FUNCTIONS
# =============================================================================

def route_decision(state: CapstoneState) -> str:
    """
    After router_node: returns 'retrieve', 'tool', or 'skip'.
    Defaults to 'retrieve' for any unrecognised value.
    """
    route = state.get("route", "retrieve")
    if route not in ("retrieve", "tool", "skip"):
        return "retrieve"
    return route


def eval_decision(state: CapstoneState) -> str:
    """
    After eval_node: returns 'answer' (retry) or 'save' (proceed to end).

    Decision logic:
      - faithfulness >= 0.7              → save  (PASS)
      - eval_retries >= MAX_EVAL_RETRIES → save  (give up after max retries)
      - otherwise                        → answer (retry with escalated prompt)
    """
    faithfulness: float = state.get("faithfulness") or 1.0
    eval_retries: int = state.get("eval_retries") or 0

    if faithfulness >= 0.7 or eval_retries >= MAX_EVAL_RETRIES:
        return "save"
    return "answer"


# =============================================================================
# GRAPH ASSEMBLY
# =============================================================================

def build_graph() -> StateGraph:
    """
    Construct the LangGraph StateGraph with all 8 nodes and edges.
    Returns the uncompiled graph (useful for testing/visualisation).
    """
    workflow = StateGraph(CapstoneState)

    # ── Add all 8 nodes ───────────────────────────────────────────────────────
    workflow.add_node("memory", memory_node)
    workflow.add_node("router", router_node)
    workflow.add_node("retrieve", retrieval_node)
    workflow.add_node("skip", skip_retrieval_node)
    workflow.add_node("tool", tool_node)
    workflow.add_node("answer", answer_node)
    workflow.add_node("eval", eval_node)
    workflow.add_node("save", save_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    workflow.set_entry_point("memory")

    # ── Static edges ──────────────────────────────────────────────────────────
    workflow.add_edge("memory", "router")
    workflow.add_edge("retrieve", "answer")
    workflow.add_edge("skip", "answer")
    workflow.add_edge("tool", "answer")
    workflow.add_edge("answer", "eval")
    workflow.add_edge("save", END)

    # ── Conditional edge: router → {retrieve | tool | skip} ───────────────────
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "retrieve": "retrieve",
            "tool": "tool",
            "skip": "skip",
        },
    )

    # ── Conditional edge: eval → {answer (retry) | save} ─────────────────────
    workflow.add_conditional_edges(
        "eval",
        eval_decision,
        {
            "answer": "answer",
            "save": "save",
        },
    )

    return workflow


# =============================================================================
# COMPILED APPLICATION (singleton used across the project)
# =============================================================================

_workflow = build_graph()
app = _workflow.compile(checkpointer=MemorySaver())

print("Graph compiled successfully.")


# =============================================================================
# QUICK SANITY CHECK (when run directly)
# =============================================================================

if __name__ == "__main__":
    import pprint
    print("\nGraph node list:", list(_workflow.nodes.keys()))
    print("\nRunning a quick test question…")
    result = app.invoke(
        {
            "question": "What is Newton's second law?",
            "messages": [],
            "route": "",
            "retrieved": "",
            "sources": [],
            "tool_result": "",
            "answer": "",
            "faithfulness": 0.0,
            "eval_retries": 0,
            "user_name": "",
        },
        config={"configurable": {"thread_id": "sanity_check"}},
    )
    print(f"\nRoute : {result.get('route')}")
    print(f"Sources: {result.get('sources')}")
    print(f"Faithfulness: {result.get('faithfulness'):.2f}")
    print(f"\nAnswer:\n{result.get('answer')}")
