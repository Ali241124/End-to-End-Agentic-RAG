
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from app.agent.graph import ask_agent
from app.rag.retriever import ingest_documents
from pathlib import Path
from app.config import DOCUMENTS_DIR
from app.rag.retriever import ingest_single_document
from app.rag.vector_store import (
    list_indexed_documents,
    delete_document_chunks,
)
from app.database.chat_history import (
    create_conversation,
    list_conversations,
    get_conversation,
    add_message,
)


app = FastAPI(
    title="Agentic RAG API",
    description=(
        "A document question-answering API powered by "
        "LangGraph, ChromaDB, and Ollama."
    ),
    version="1.0.0",
)


# -----------------------------
# Request and response models
# -----------------------------

class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="The question to ask the Agentic RAG system.",
    )
    conversation_id: str | None = None


class QuestionResponse(BaseModel):
    conversation_id: str
    question: str
    answer: str
    sources: list[str]
    retrieval_attempts: int
    evidence_relevant: bool
    grounded: bool | None = None
    verification_error: str | None = None


class IngestResponse(BaseModel):
    message: str
    total_chunks: int


# -----------------------------
# Basic endpoints
# -----------------------------

@app.get("/")
def home():
    """Confirm that the API is running."""
    return {
        "message": "Agentic RAG API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    """Basic API health check."""
    return {"status": "healthy"}


# -----------------------------
# Conversation endpoints
# -----------------------------

@app.post("/conversations")
def new_conversation():
    """Create a new conversation."""
    return create_conversation()


@app.get("/conversations")
def recent_conversations():
    """List conversations, newest activity first."""
    return list_conversations()


@app.get("/conversations/{conversation_id}")
def load_conversation(conversation_id: str):
    """Load a conversation and all its saved messages."""
    conversation = get_conversation(conversation_id)

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    return conversation


# -----------------------------
# Agentic RAG question endpoint
# -----------------------------

@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):
    """Answer a question and save the conversation messages."""

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty or whitespace.",
        )

    conversation_id = request.conversation_id

    # Create a new conversation if no ID was provided.
    if conversation_id is None:
        conversation = create_conversation()
        conversation_id = conversation["id"]

    # Reject IDs that do not exist.
    elif get_conversation(conversation_id) is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Conversation not found. "
                "Create a new conversation first."
            ),
        )

    try:
        # Save the user's question.
        
        # Save the current user message first.
        add_message(conversation_id, "user", question)

        # Load this conversation's saved history.
        conversation = get_conversation(conversation_id)

        if conversation is None:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found.",
            )

        # Exclude the current question, which was just saved.
        previous_messages = conversation["messages"][:-1]

        # Pass only valid user/assistant messages to the agent.
        conversation_history = [
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in previous_messages
            if message.get("role") in ("user", "assistant")
            and message.get("content", "").strip()
        ]

        # Run the agent with the previous conversation context.
        result = ask_agent(
            question,
            conversation_history=conversation_history,
        )


        # Collect unique source filenames.
        sources = sorted({
            document["source"]
            for document in result["documents"]
        })

        # Save the assistant's response.
        add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=result["answer"],
            sources=sources,
            retrieval_attempts=result["attempts"],
        )

        return QuestionResponse(
            conversation_id=conversation_id,
            question=question,
            answer=result["answer"],
            sources=sources,
            retrieval_attempts=result["attempts"],
            evidence_relevant=result["relevant"],
            grounded=result.get("grounded"),
            verification_error=result.get("verification_error"),
        )


    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process the question: {exc}",
        ) from exc


# -----------------------------
# Document ingestion endpoint
# -----------------------------

@app.post("/ingest", response_model=IngestResponse)
def ingest():
    """Load documents and store their chunks in ChromaDB."""

    try:
        total_chunks = ingest_documents()

        if total_chunks == 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No text chunks found. "
                    "Add TXT files to data/documents."
                ),
            )

        return IngestResponse(
            message="Documents ingested successfully",
            total_chunks=total_chunks,
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Document ingestion failed: {exc}",
        ) from exc

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    replace_existing: bool = Form(False),
):
    """Save and index one uploaded PDF or TXT document."""

    allowed_extensions = {".pdf", ".txt"}
    max_file_size = 10 * 1024 * 1024  # 10 MB

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Please select a file to upload.",
        )

    safe_name = Path(file.filename).name
    extension = Path(safe_name).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported.",
        )

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    destination = DOCUMENTS_DIR / safe_name
    already_exists = destination.exists()

    if already_exists and not replace_existing:
        raise HTTPException(
            status_code=409,
            detail=(
                f"'{safe_name}' already exists. "
                "Enable replacement to update it."
            ),
        )

    content = await file.read(max_file_size + 1)
    await file.close()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(content) > max_file_size:
        raise HTTPException(
            status_code=413,
            detail="The maximum upload size is 10 MB.",
        )

    # Keep the existing file contents in case replacement fails.
    old_content = destination.read_bytes() if already_exists else None

    try:
        destination.write_bytes(content)
        total_chunks = ingest_single_document(destination)

        action = "replaced and re-indexed" if already_exists else "uploaded and indexed"

        return {
            "message": f"'{safe_name}' {action} successfully.",
            "filename": safe_name,
            "total_chunks": total_chunks,
        }

    except Exception as exc:
        if already_exists and old_content is not None:
            # Restore the original file and attempt to restore its index.
            try:
                destination.write_bytes(old_content)
                ingest_single_document(destination)
            except Exception as recovery_error:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        f"Replacement failed: {exc}. "
                        "The original file was restored, but index recovery "
                        f"also failed: {recovery_error}"
                    ),
                ) from exc
        else:
            destination.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail=f"Could not process '{safe_name}': {exc}",
        ) from exc

@app.get("/documents")
def get_documents():
    """List supported document files and their indexing status."""

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    indexed_documents = {
        document["filename"]: document["chunks"]
        for document in list_indexed_documents()
    }

    documents = []

    for file_path in sorted(DOCUMENTS_DIR.iterdir()):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in {".pdf", ".txt"}:
            continue

        chunk_count = indexed_documents.get(file_path.name, 0)

        documents.append(
            {
                "filename": file_path.name,
                "indexed": chunk_count > 0,
                "chunks": chunk_count,
                "size_bytes": file_path.stat().st_size,
            }
        )

    return {
        "total_documents": len(documents),
        "documents": documents,
    }


@app.delete("/documents")
def delete_document(filename: str):
    """Delete one document and its indexed chunks."""

    safe_name = Path(filename).name

    if safe_name != filename or safe_name in {"", ".", ".."}:
        raise HTTPException(
            status_code=400,
            detail="Invalid document filename.",
        )

    if Path(safe_name).suffix.lower() not in {".pdf", ".txt"}:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT documents can be deleted.",
        )

    file_path = DOCUMENTS_DIR / safe_name

    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Document '{safe_name}' was not found.",
        )

    try:
        # Remove indexed chunks first so deleted documents
        # cannot continue appearing in retrieval results.
        delete_document_chunks(safe_name)

        file_path.unlink()

        return {
            "message": f"'{safe_name}' was deleted successfully.",
            "filename": safe_name,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not delete '{safe_name}': {exc}",
        ) from exc

