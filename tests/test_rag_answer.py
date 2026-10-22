
from app.rag.retriever import ingest_documents, retrieve_documents
from app.rag.generator import generate_answer


def main():
    print("Step 1: Ingesting documents...")
    total_chunks = ingest_documents()
    print(f"Chunks processed: {total_chunks}")

    question = "What is text chunking used for?"

    print(f"\nStep 2: Retrieving passages for: {question}")
    passages = retrieve_documents(question, top_k=4)

    if not passages:
        raise RuntimeError("No relevant passages were retrieved.")

    print(f"Retrieved {len(passages)} passages.")

    print("\nStep 3: Generating the answer with Ollama...")
    answer = generate_answer(question, passages)

    print("\n" + "=" * 50)
    print("QUESTION")
    print(question)

    print("\nANSWER")
    print(answer)

    print("\nSOURCES")
    for source in sorted({p["source"] for p in passages}):
        print(f"- {source}")

    print("=" * 50)
    print("\nBasic RAG test passed!")


if __name__ == "__main__":
    main()
