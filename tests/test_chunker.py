from app.rag.chunker import split_text


def test_split_text_returns_chunks():
    text = "Artificial intelligence. " * 200

    chunks = split_text(text)

    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)


if __name__ == "__main__":
    test_split_text_returns_chunks()
    print("Chunker test passed!")