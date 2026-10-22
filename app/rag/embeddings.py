
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.config import EMBEDDING_MODEL, OLLAMA_BASE_URL


def create_embeddings(texts: list[str]) -> list[list[float]]:
    """Convert a list of texts into embedding vectors using Ollama."""

    if not texts:
        return []

    payload = {
        "model": EMBEDDING_MODEL,
        "input": texts,
    }

    request = Request(
        url=f"{OLLAMA_BASE_URL}/api/embed",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
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

    embeddings = result.get("embeddings", [])

    if len(embeddings) != len(texts):
        raise RuntimeError(
            "Ollama returned an unexpected number of embeddings."
        )

    if any(not vector for vector in embeddings):
        raise RuntimeError("Ollama returned an empty embedding vector.")

    dimensions = len(embeddings[0])

    if any(len(vector) != dimensions for vector in embeddings):
        raise RuntimeError("Embedding vectors have inconsistent dimensions.")

    return embeddings
