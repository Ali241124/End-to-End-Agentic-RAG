
from app.database.chat_history import (
    create_conversation,
    add_message,
    get_conversation,
    list_conversations,
    delete_conversation,
)


def main():
    # Create a temporary test conversation in the project database.
    chat = create_conversation("Database Test")
    chat_id = chat["id"]

    try:
        add_message(
            chat_id,
            "user",
            "What is RAG?",
        )

        add_message(
            chat_id,
            "assistant",
            "RAG combines retrieval with language generation.",
            sources=["ai_notes.txt"],
            retrieval_attempts=1,
        )

        saved_chat = get_conversation(chat_id)

        assert saved_chat is not None
        assert len(saved_chat["messages"]) == 2
        assert saved_chat["messages"][0]["role"] == "user"
        assert saved_chat["messages"][1]["sources"] == ["ai_notes.txt"]

        chats = list_conversations()
        assert any(chat["id"] == chat_id for chat in chats)

        print("Conversation creation: passed")
        print("Message persistence: passed")
        print("Source persistence: passed")
        print("Conversation listing: passed")

    finally:
        delete_conversation(chat_id)

    print("\nChat history database test passed!")


if __name__ == "__main__":
    main()
