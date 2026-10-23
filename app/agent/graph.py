
from typing import Literal

from langgraph.graph import StateGraph, START, END

from app.config import MAX_RETRIEVAL_ATTEMPTS, TOP_K
from app.agent.state import AgentState
from app.agent.evaluator import (
    evaluate_relevance,
    evaluate_answer_grounding,
)
from app.rag.retriever import retrieve_documents
from app.rag.generator import generate_answer
from app.agent.query_contextualizer import contextualize_question


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
    """Make an alternative query for another retrieval attempt."""

    rewritten_query = (
        f"{state['search_query']} "
        "relevant concepts definitions explanations"
    )

    if rewritten_query == state["search_query"]:
        rewritten_query = (
            f"{state['question']} related information"
        )

    print("[Agent] Trying an alternative search query")

    return {"search_query": rewritten_query}

def generate_node(state: AgentState) -> dict:
    """Generate an answer using retrieved evidence and conversation history."""

    answer = generate_answer(
        question=state["question"],
        retrieved_chunks=state["documents"],
        conversation_history=state["conversation_history"],
    )

    print("[Agent] Answer generated")

    return {"answer": answer}

def verify_answer_node(state: AgentState) -> dict:
    """Verify the generated answer against retrieved evidence."""

    try:
        grounded = evaluate_answer_grounding(
            question=state["question"],
            answer=state["answer"],
            documents=state["documents"],
        )

        print(f"[Agent] Answer grounded: {grounded}")

        if not grounded:
            return {
                "grounded": False,
                "verification_error": None,
                "answer": (
                    "I couldn't verify that this answer is fully "
                    "supported by the retrieved documents. Please "
                    "consult the source documents or rephrase your "
                    "question."
                ),
            }

        return {
            "grounded": True,
            "verification_error": None,
        }

    except Exception as exc:
        print(f"[Agent] Grounding verification failed: {exc}")

        return {
            "grounded": False,
            "verification_error": str(exc),
            "answer": (
                "I couldn't verify the answer because the verification "
                "service encountered an error. Please try again."
            ),
        }

def no_evidence_node(state: AgentState) -> dict:
    """Respond safely when retrieval finds no relevant evidence."""

    answer = (
        "I couldn't find sufficiently relevant information in the "
        "uploaded documents to answer this question confidently. "
        "Try rephrasing your question or uploading a document that "
        "covers this topic."
    )

    print("[Agent] No relevant evidence found. Skipping generation.")

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
    workflow.add_node("no_evidence", no_evidence_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "evaluate")

    workflow.add_conditional_edges(
        "evaluate",
        decide_next_step,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "finish": "no_evidence",
        },
    )

    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", "verify")
    workflow.add_node("verify", verify_answer_node)
    workflow.add_edge("verify", END)
    workflow.add_edge("no_evidence", END)

    return workflow.compile()


agent = build_agent()



def ask_agent(
    question: str,
    conversation_history: list[dict] | None = None,
) -> dict:
    """Run the agent using conversation-aware retrieval."""

    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    history = conversation_history or []

    # Convert a follow-up into a standalone retrieval query.
    search_query = contextualize_question(
        question=question,
        conversation_history=history,
    )

    print(f"[Agent] Original question: {question}")
    print(f"[Agent] Contextualized search query: {search_query}")

    initial_state: AgentState = {
        "question": question,
        "search_query": search_query,
        "conversation_history": history,
        "documents": [],
        "attempts": 0,
        "relevant": False,
        "answer": "",
        "grounded": None,
        "verification_error": None,
    }

    return agent.invoke(initial_state)
