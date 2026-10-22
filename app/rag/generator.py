
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.config import LLM_MODEL, OLLAMA_BASE_URL


def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:
    """Generate an answer using the retrieved document passages."""

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not retrieved_chunks:
        return "I couldn't find relevant information in the available documents."

    # Combine retrieved passages into a single context.
    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Source {i}: {chunk['source']}]\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    # Tell the LLM how to answer.
    prompt = f"""
You are a helpful question-answering assistant.

Answer the question using only the provided context.
If the context does not contain enough information, say so.
Do not invent facts.
Mention the source filename when appropriate.

Context:
{context}

Question:
{question}

Answer:
"""

    # Prepare the request for Ollama's chat API.
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
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
