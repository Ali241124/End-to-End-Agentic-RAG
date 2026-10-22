
import json
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from app.config import LLM_MODEL, OLLAMA_BASE_URL


def evaluate_relevance(question: str, documents: list[dict]) -> bool:
    """Ask the LLM whether retrieved passages help answer the question."""

    if not documents:
        return False

    context = "\n\n".join(
        document["text"] for document in documents
    )

    prompt = f"""
Decide whether the context contains useful information for answering
the question.

Return only a JSON object in this exact format:
{{"relevant": true}}
or
{{"relevant": false}}

Question:
{question}

Context:
{context}
"""

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
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

    content = result.get("message", {}).get("content", "")

    try:
        decision = json.loads(content)
        return decision.get("relevant") is True
    except (json.JSONDecodeError, AttributeError):
        raise RuntimeError(
            f"Ollama returned an invalid relevance decision: {content}"
        )
