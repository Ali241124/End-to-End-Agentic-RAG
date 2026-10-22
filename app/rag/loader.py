
from pathlib import Path

from app.config import DOCUMENTS_DIR


def load_text_file(file_path: Path) -> str:
    """Read a text file and return its contents."""

    return file_path.read_text(encoding="utf-8-sig").strip()


def load_documents() -> list[dict[str, str]]:
    """Load all TXT documents from the documents directory."""

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    documents = []

    for file_path in DOCUMENTS_DIR.glob("*.txt"):
        text = load_text_file(file_path)

        if not text:
            continue

        documents.append(
            {
                "text": text,
                "source": file_path.name,
            }
        )

    return documents
