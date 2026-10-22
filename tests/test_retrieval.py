
from app.rag.retriever import ingest_documents, retrieve_documents


def main():
    print("Ingesting documents...")

    total_chunks = ingest_documents()

    if total_chunks == 0:
        raise RuntimeError(
            "No chunks were stored. Check data/documents."
        )

    print(f"\nTotal chunks stored: {total_chunks}")

    question = "What is text chunking used for?"

    print(f"\nQuestion: {question}")
    print("\nRetrieved passages:")

    results = retrieve_documents(question, top_k=4)

    if not results:
        raise RuntimeError("No passages were retrieved.")

    for i, result in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print(f"Source: {result['source']}")
        print(f"Chunk: {result['chunk_index']}")
        print(f"Distance: {result['distance']:.4f}")
        print(result["text"])

    print("\nRetrieval test passed!")


if __name__ == "__main__":
    main()
