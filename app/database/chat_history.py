
import sqlite3
from datetime import datetime, timezone
from uuid import uuid4

from app.config import BASE_DIR


DATABASE_PATH = BASE_DIR / "data" / "chat_history.db"


def get_connection():
    """Open a connection to the chat database."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    """Create the conversation and message tables if needed."""
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT NOT NULL DEFAULT '[]',
                retrieval_attempts INTEGER,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(id)
                    ON DELETE CASCADE
            )
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_conversation
            ON messages(conversation_id, id)
        """)


def create_conversation(title: str = "New Chat") -> dict:
    """Create a conversation and return its details."""
    conversation_id = str(uuid4())
    now = datetime.now(timezone.utc).isoformat()

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO conversations (id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, title, now, now),
        )

    return {
        "id": conversation_id,
        "title": title,
        "created_at": now,
        "updated_at": now,
    }


def list_conversations() -> list[dict]:
    """Return conversations, newest activity first."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM conversations
            ORDER BY updated_at DESC
            """
        ).fetchall()

    return [dict(row) for row in rows]


def get_conversation(conversation_id: str) -> dict | None:
    """Return one conversation and its messages."""
    with get_connection() as connection:
        conversation = connection.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM conversations
            WHERE id = ?
            """,
            (conversation_id,),
        ).fetchone()

        if conversation is None:
            return None

        messages = connection.execute(
            """
            SELECT id, role, content, sources,
                   retrieval_attempts, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id
            """,
            (conversation_id,),
        ).fetchall()

    result = dict(conversation)
    result["messages"] = []

    for row in messages:
        message = dict(row)
        message["sources"] = __import__("json").loads(message["sources"])
        result["messages"].append(message)

    return result


def add_message(
    conversation_id: str,
    role: str,
    content: str,
    sources: list[str] | None = None,
    retrieval_attempts: int | None = None,
) -> dict:
    """Save a message and update the conversation's activity time."""
    import json

    if role not in {"user", "assistant"}:
        raise ValueError("Role must be 'user' or 'assistant'.")

    now = datetime.now(timezone.utc).isoformat()

    with get_connection() as connection:
        conversation = connection.execute(
            "SELECT id, title FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()

        if conversation is None:
            raise ValueError("Conversation not found.")

        connection.execute(
            """
            INSERT INTO messages (
                conversation_id, role, content, sources,
                retrieval_attempts, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                role,
                content,
                json.dumps(sources or []),
                retrieval_attempts,
                now,
            ),
        )

        # Use the first user question as the conversation title.
        if role == "user" and conversation["title"] == "New Chat":
            title = content.strip().replace("\n", " ")
            if len(title) > 50:
                title = title[:47] + "..."
            connection.execute(
                "UPDATE conversations SET title = ? WHERE id = ?",
                (title or "New Chat", conversation_id),
            )

        connection.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )

    return {
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
        "sources": sources or [],
        "retrieval_attempts": retrieval_attempts,
        "created_at": now,
    }


def delete_conversation(conversation_id: str) -> bool:
    """Delete one conversation and its messages."""
    with get_connection() as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        cursor = connection.execute(
            "DELETE FROM conversations WHERE id = ?",
            (conversation_id,),
        )
        return cursor.rowcount > 0


# Create the database tables when this module is imported.
initialize_database()
