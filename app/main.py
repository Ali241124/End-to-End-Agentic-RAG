
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.agent.graph import ask_agent
from app.rag.retriever import ingest_documents


app = FastAPI(
    title="Agentic RAG API",
    description="A document question-answering API powered by LangGraph, ChromaDB, and Ollama.",
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="The question to ask the Agentic RAG system.",
    )


class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    retrieval_attempts: int
    evidence_relevant: bool


class IngestResponse(BaseModel):
    message: str
    total_chunks: int


@app.get("/")
def home():
    """Confirm that the API is running."""
    return {
        "message": "Agentic RAG API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    """Basic service health check."""
    return {"status": "healthy"}


@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):
    """Run the Agentic RAG workflow for a question."""
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty or whitespace.",
        )

    try:
        result = ask_agent(request.question.strip())

        sources = sorted({
            document["source"]
            for document in result["documents"]
        })

        return QuestionResponse(
            question=request.question.strip(),
            answer=result["answer"],
            sources=sources,
            retrieval_attempts=result["attempts"],
            evidence_relevant=result["relevant"],
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process the question: {exc}",
        ) from exc


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    """Load documents and store their chunks in ChromaDB."""
    try:
        total_chunks = ingest_documents()

        if total_chunks == 0:
            raise HTTPException(
                status_code=400,
                detail="No text chunks found. Add TXT files to data/documents.",
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
