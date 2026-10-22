
from pathlib import Path

import fitz  # PyMuPDF

from app.config import DOCUMENTS_DIR


def load_text_file(file_path: Path) -> str:
    """Read a UTF-8 text file and return its contents."""

    return file_path.read_text(encoding="utf-8-sig").strip()


def load_pdf_file(file_path: Path) -> str:
    """Extract text from every page of a PDF file."""

    pages = []

    with fitz.open(file_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            page_text = page.get_text("text").strip()

            if page_text:
                pages.append(page_text)

    return "\n\n".join(pages).strip()


def load_document(file_path: Path) -> str:
    """Load a supported document based on its file extension."""

    extension = file_path.suffix.lower()

    if extension == ".txt":
        return load_text_file(file_path)

    if extension == ".pdf":
        return load_pdf_file(file_path)

    raise ValueError(
        f"Unsupported document format: {extension}"
    )


def load_documents() -> list[dict[str, str]]:
    """Load all supported TXT and PDF documents."""

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    documents = []

    supported_extensions = {".txt", ".pdf"}

    for file_path in sorted(DOCUMENTS_DIR.iterdir()):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in supported_extensions:
            continue

        try:
            text = load_document(file_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load document '{file_path.name}': {exc}"
            ) from exc

        if not text:
            print(
                f"[Loader] Skipping '{file_path.name}': "
                "no extractable text found."
            )
            continue

        documents.append(
            {
                "text": text,
                "source": file_path.name,
            }
        )

        print(f"[Loader] Loaded: {file_path.name}")

    return documents
