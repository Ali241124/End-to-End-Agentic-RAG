
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.config import LLM_MODEL, OLLAMA_BASE_URL


def contextualize_question(
    question: str,
    conversation_history: list[dict] | None = None,
) -> str:
    """Rewrite a follow-up question as a standalone retrieval query."""

    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    history = conversation_history or []

    # A standalone question needs no contextualization.
    if not history:
        return question

    messages = [
        {
            "role": "system",
            "content": """
You are a search-query rewriting tool, NOT a question-answering assistant.

Your only task is to rewrite the latest user question into a standalone
search query using the conversation history.

Rules:
- NEVER answer the user's question.
- NEVER say "I couldn't find", "I don't know", or provide explanations.
- Return exactly one search query and nothing else.
- If the latest question mentions "it", "they", "that", or "the method",
  resolve the reference using the conversation history.
- Preserve all named concepts in the latest question.
- Do not claim facts or invent details.
- Keep the query concise.

Example:
Conversation:
User: What is RAG?
Assistant: RAG retrieves relevant information from external documents.
Latest question: How is it different from GraphRAG?

Correct output:
How does Retrieval-Augmented Generation (RAG) differ from GraphRAG?

Incorrect output:
I couldn't find any information on GraphRAG.
""".strip()

        }
    ]

    # Include only recent user and assistant messages.
    for message in history[-6:]:
        role = message.get("role")
        content = message.get("content", "")

        if role in ("user", "assistant") and isinstance(content, str):
            if content.strip():
                messages.append(
                    {
                        "role": role,
                        "content": content.strip(),
                    }
                )

    messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0,
        },
    }

    request = Request(
        url=f"{OLLAMA_BASE_URL}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))

        rewritten_query = (
            result.get("message", {}).get("content", "").strip()
        )

        return rewritten_query or question

    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        # Contextualization is an enhancement, not a reason to
        # prevent the original question from being processed.
        print(f"[Agent] Query contextualization failed: {exc}")
        print("[Agent] Falling back to the original question.")
        return question
