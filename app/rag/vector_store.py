
import chromadb

from app.config import CHROMA_DIR

# Create a persistent ChromaDB client.
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

client = chromadb.PersistentClient(path=str(CHROMA_DIR))

# Use cosine distance to compare embedding vectors.
collection = client.get_or_create_collection(
    name="agentic_rag_documents",
    metadata={"hnsw:space": "cosine"},
)


def store_chunks(
    chunks: list[str],
    embeddings: list[list[float]],
    source: str,
) -> None:
    """Store text chunks, vectors, and source metadata."""

    if len(chunks) != len(embeddings):
        raise ValueError("Each chunk must have one embedding.")

    if not chunks:
        return

    # Remove older chunks from this source before re-ingesting it.
    collection.delete(where={"source": source})

    ids = [f"{source}_{i}" for i in range(len(chunks))]
    metadatas = [
        {"source": source, "chunk_index": i}
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def search_chunks(
    query_embedding: list[float],
    top_k: int = 4,
) -> list[dict]:
    """Find the closest document chunks to a query vector."""

    if not query_embedding:
        raise ValueError("Query embedding cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if collection.count() == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    matches = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for document, metadata, distance in zip(
        documents, metadatas, distances
    ):
        matches.append(
            {
                "text": document,
                "source": metadata["source"],
                "chunk_index": metadata["chunk_index"],
                "distance": distance,
            }
        )

    return matches
