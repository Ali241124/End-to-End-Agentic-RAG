
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.config import LLM_MODEL, OLLAMA_BASE_URL


def generate_answer(
    question: str,
    retrieved_chunks: list[dict],
    conversation_history: list[dict] | None = None,
) -> str:
    """Generate an evidence-grounded answer using recent conversation history."""

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not retrieved_chunks:
        return "I couldn't find relevant information in the available documents."

    # Build the evidence context from retrieved document passages.
    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Source {i}: {chunk['source']}]\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    # Give the model instructions that prioritize retrieved evidence.
    system_prompt = """
You are a helpful, evidence-grounded question-answering assistant.

Use the supplied document context as the primary source of factual
information. Conversation history helps you understand references and
follow-up questions, but it is not proof that a factual claim is true.

If the documents do not contain enough information to answer a question
about their contents, clearly say so. Do not invent facts.

Use conversation history to understand references such as "it",
"that method", or "the previous approach".
Answer the user's latest question, not an earlier question.
Mention source filenames when appropriate.
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt.strip(),
        }
    ]

    # Include only recent user/assistant messages to limit context size.
    recent_history = (conversation_history or [])[-6:]

    for message in recent_history:
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

    # Add the current question and retrieved evidence last.
    current_prompt = f"""
Retrieved document context:

{context}

Latest question:
{question}

Answer the latest question using the instructions and document context.
"""

    messages.append(
        {
            "role": "user",
            "content": current_prompt.strip(),
        }
    )

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": False,
    }

    request = Request(
        url=f"{OLLAMA_BASE_URL}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))

    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama returned HTTP {exc.code}: {details}"
        ) from exc

    except URLError as exc:
        raise RuntimeError(
            "Cannot connect to Ollama. Check that Ollama is running."
        ) from exc

    answer = result.get("message", {}).get("content", "").strip()

    if not answer:
        raise RuntimeError("Ollama returned an empty answer.")

    return answer
