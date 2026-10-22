
from app.config import TOP_K
from app.rag.loader import load_documents
from app.rag.chunker import split_text
from app.rag.embeddings import create_embeddings
from app.rag.vector_store import store_chunks, search_chunks


def ingest_documents() -> int:
    """Load documents, chunk them, embed them, and store them."""

    documents = load_documents()
    total_chunks = 0

    for document in documents:
        chunks = split_text(document["text"])

        if not chunks:
            continue

        embeddings = create_embeddings(chunks)

        store_chunks(
            chunks=chunks,
            embeddings=embeddings,
            source=document["source"],
        )

        total_chunks += len(chunks)

        print(
            f"Ingested {document['source']}: "
            f"{len(chunks)} chunks"
        )

    return total_chunks


def retrieve_documents(
    question: str,
    top_k: int = TOP_K,
) -> list[dict]:
    """Retrieve relevant passages for a question."""

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    query_embedding = create_embeddings([question])[0]

    return search_chunks(
        query_embedding=query_embedding,
        top_k=top_k,
    )
