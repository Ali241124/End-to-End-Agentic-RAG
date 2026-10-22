
from app.agent.graph import ask_agent


def main():
    question = "What is text chunking used for?"

    print("=" * 50)
    print("AGENTIC RAG TEST")
    print("=" * 50)
    print(f"Question: {question}")

    result = ask_agent(question)

    print("\nFinal answer:")
    print(result["answer"])

    print("\nAgent details:")
    print(f"Retrieval attempts: {result['attempts']}")
    print(f"Evidence judged relevant: {result['relevant']}")
    print(f"Retrieved passages: {len(result['documents'])}")

    assert result["answer"].strip(), "The agent returned no answer."
    assert 1 <= result["attempts"] <= 2

    print("\nAgent test passed!")


if __name__ == "__main__":
    main()
