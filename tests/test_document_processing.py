
from app.rag.loader import load_documents
from app.rag.embeddings import create_embeddings


def main():
    documents = load_documents()

    if not documents:
        raise RuntimeError(
            "No TXT documents found in data/documents."
        )

    print(f"Documents loaded: {len(documents)}")
    print("First source:", documents[0]["source"])
    print("First 100 characters:", documents[0]["text"][:100])

    vectors = create_embeddings(
        [documents[0]["text"][:500]]
    )

    print("Embedding dimensions:", len(vectors[0]))
    print("Embedding test passed!")


if __name__ == "__main__":
    main()
