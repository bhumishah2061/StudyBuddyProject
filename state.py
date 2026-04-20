"""
state.py – Defines the CapstoneState TypedDict for Study Buddy Physics.

All fields written by any node in the LangGraph pipeline must be declared here.
"""

from typing import TypedDict, List


class CapstoneState(TypedDict):
    """
    Shared state passed through every node in the LangGraph pipeline.

    Fields
    ------
    question     : The current user question for this turn.
    messages     : Sliding window conversation history (max 6 messages).
    route        : Routing decision: 'retrieve' | 'tool' | 'skip'.
    retrieved    : Formatted retrieved context from ChromaDB (empty if tool/skip).
    sources      : List of topic labels returned by retrieval.
    tool_result  : String output from calculator or datetime tool.
    answer       : Generated answer from the answer_node.
    faithfulness : Faithfulness score (0.0–1.0) from eval_node.
    eval_retries : Number of eval retry cycles consumed this turn.
    user_name    : Student name extracted when they say "my name is …".
    """

    question: str
    messages: List[dict]
    route: str
    retrieved: str
    sources: List[str]
    tool_result: str
    answer: str
    faithfulness: float
    eval_retries: int
    user_name: str
