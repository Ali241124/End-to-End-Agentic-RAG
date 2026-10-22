
from typing import Literal

from langgraph.graph import StateGraph, START, END

from app.config import MAX_RETRIEVAL_ATTEMPTS, TOP_K
from app.agent.state import AgentState
from app.agent.evaluator import evaluate_relevance
from app.rag.retriever import retrieve_documents
from app.rag.generator import generate_answer


def retrieve_node(state: AgentState) -> dict:
    """Retrieve passages for the current search query."""

    query = state["search_query"]
    documents = retrieve_documents(query, top_k=TOP_K)
    attempts = state["attempts"] + 1

    print(f"\n[Agent] Retrieval attempt {attempts}")
    print(f"[Agent] Search query: {query}")
    print(f"[Agent] Retrieved {len(documents)} passages")

    return {
        "documents": documents,
        "attempts": attempts,
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate whether the retrieved passages are useful."""

    relevant = evaluate_relevance(
        question=state["question"],
        documents=state["documents"],
    )

    print(f"[Agent] Evidence relevant: {relevant}")

    return {"relevant": relevant}


def rewrite_query_node(state: AgentState) -> dict:
    """Make a simple alternative query for the next retrieval attempt."""

    original_query = state["search_query"]

    # Add a little more retrieval context without changing the question.
    rewritten_query = (
        f"{state['question']} relevant concepts definitions explanations"
    )

    # Avoid repeating the exact same query if it already contains this suffix.
    if rewritten_query == original_query:
        rewritten_query = f"{state['question']} related information"

    print(f"[Agent] Trying an alternative search query")

    return {"search_query": rewritten_query}


def generate_node(state: AgentState) -> dict:
    """Generate the final answer from the retrieved evidence."""

    answer = generate_answer(
        question=state["question"],
        retrieved_chunks=state["documents"],
    )

    print("[Agent] Answer generated")

    return {"answer": answer}


def decide_next_step(
    state: AgentState,
) -> Literal["generate", "rewrite", "finish"]:
    """Choose the next node based on relevance and remaining attempts."""

    if state["relevant"]:
        return "generate"

    if state["attempts"] < MAX_RETRIEVAL_ATTEMPTS:
        return "rewrite"

    return "finish"


def build_agent():
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("rewrite", rewrite_query_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "evaluate")

    workflow.add_conditional_edges(
        "evaluate",
        decide_next_step,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "finish": "generate",
        },
    )

    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", END)

    return workflow.compile()


agent = build_agent()


def ask_agent(question: str) -> dict:
    """Run the agent and return its answer and retrieval details."""

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    initial_state: AgentState = {
        "question": question.strip(),
        "search_query": question.strip(),
        "documents": [],
        "attempts": 0,
        "relevant": False,
        "answer": "",
    }

    return agent.invoke(initial_state)
