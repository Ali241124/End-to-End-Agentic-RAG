
import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"


# --------------------------------
# Page configuration
# --------------------------------

st.set_page_config(
    page_title="Agentic RAG",
    page_icon="🤖",
    layout="wide",
)


# --------------------------------
# API helper functions
# --------------------------------

def create_chat() -> dict:
    """Create a conversation through FastAPI."""
    response = requests.post(
        f"{API_URL}/conversations",
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def fetch_chats() -> list[dict]:
    """Fetch saved conversations from SQLite through FastAPI."""
    response = requests.get(
        f"{API_URL}/conversations",
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def fetch_chat(conversation_id: str) -> dict:
    """Load one conversation and its messages."""
    response = requests.get(
        f"{API_URL}/conversations/{conversation_id}",
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def ask_question(question: str, conversation_id: str) -> dict:
    """Send a question to the Agentic RAG backend."""
    response = requests.post(
        f"{API_URL}/ask",
        json={
            "question": question,
            "conversation_id": conversation_id,
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()

def upload_document(uploaded_file) -> dict:
    """Upload one PDF or TXT file to FastAPI for indexing."""

    response = requests.post(
        f"{API_URL}/upload",
        files={
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "application/octet-stream",
            )
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()

def display_message(message: dict):
    """Display a saved message, including answer sources."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            sources = message.get("sources", [])

            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.write(f"- {source}")

            attempts = message.get("retrieval_attempts")

            if attempts is not None:
                st.caption(f"Retrieval attempts: {attempts}")


# --------------------------------
# Session state initialization
# --------------------------------

if "active_conversation_id" not in st.session_state:
    st.session_state.active_conversation_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------
# Sidebar: new chat and history
# --------------------------------

st.sidebar.title("🤖 Agentic RAG")

if st.sidebar.button("＋ New Chat", use_container_width=True):
    try:
        new_chat = create_chat()
        st.session_state.active_conversation_id = new_chat["id"]
        st.session_state.messages = []
        st.rerun()
    except requests.RequestException as exc:
        st.sidebar.error(f"Could not create chat: {exc}")


st.sidebar.divider()
st.sidebar.subheader("Recent Chats")

try:
    conversations = fetch_chats()

    for chat in conversations:
        chat_id = chat["id"]
        title = chat["title"]

        # Visually distinguish the active conversation.
        if chat_id == st.session_state.active_conversation_id:
            button_label = f"🟢 {title}"
        else:
            button_label = f"💬 {title}"

        if st.sidebar.button(
            button_label,
            key=f"conversation_{chat_id}",
            use_container_width=True,
        ):
            try:
                selected_chat = fetch_chat(chat_id)

                st.session_state.active_conversation_id = chat_id
                st.session_state.messages = selected_chat["messages"]

                st.rerun()

            except requests.RequestException as exc:
                st.sidebar.error(f"Could not load chat: {exc}")

except requests.RequestException:
    st.sidebar.error(
        "Cannot reach FastAPI. Start the backend and refresh this page."
    )


st.sidebar.divider()

st.sidebar.subheader("Upload a Document")

with st.sidebar.form("document_upload_form", clear_on_submit=True):
    uploaded_file = st.file_uploader(
        "Choose a PDF or TXT file",
        type=["pdf", "txt"],
        help="Maximum file size: 10 MB.",
    )

    upload_submitted = st.form_submit_button(
        "Upload and Index",
        use_container_width=True,
    )

if upload_submitted:
    if uploaded_file is None:
        st.sidebar.warning("Please select a PDF or TXT file first.")
    else:
        try:
            with st.spinner("Uploading and indexing document..."):
                result = upload_document(uploaded_file)

            st.sidebar.success(result["message"])
            st.sidebar.write(
                f"Chunks indexed: {result['total_chunks']}"
            )

        except requests.HTTPError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            st.sidebar.error(f"Upload failed: {detail}")

        except requests.RequestException as exc:
            st.sidebar.error(f"Could not reach the backend: {exc}")

if st.sidebar.button("📚 Ingest Documents", use_container_width=True):
    try:
        with st.spinner("Ingesting documents..."):
            response = requests.post(
                f"{API_URL}/ingest",
                timeout=300,
            )
            response.raise_for_status()
            result = response.json()

        st.sidebar.success(result["message"])
        st.sidebar.write(
            f"Chunks processed: {result['total_chunks']}"
        )

    except requests.RequestException as exc:
        st.sidebar.error(f"Ingestion failed: {exc}")


st.sidebar.caption("Powered by LangGraph, FastAPI, ChromaDB and Ollama")


# --------------------------------
# Ensure an active conversation
# --------------------------------

if st.session_state.active_conversation_id is None:
    try:
        new_chat = create_chat()

        st.session_state.active_conversation_id = new_chat["id"]
        st.session_state.messages = []

    except requests.RequestException:
        st.error(
            "The backend is unavailable. Start FastAPI in another "
            "terminal, then refresh this page."
        )
        st.stop()


# --------------------------------
# Main chat interface
# --------------------------------

st.title("Agentic RAG")
st.caption(
    "Ask questions about your documents. "
    "Your conversations are saved automatically."
)

for message in st.session_state.messages:
    display_message(message)


# --------------------------------
# Handle a new question
# --------------------------------

question = st.chat_input("Ask a question about your documents...")

if question:
    conversation_id = st.session_state.active_conversation_id

    # Show the user's question immediately.
    user_message = {
        "role": "user",
        "content": question,
    }

    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(question)

    # Ask the backend and display its response.
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documents and generating an answer..."):
                result = ask_question(question, conversation_id)

            answer_message = {
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get("sources", []),
                "retrieval_attempts": result.get("retrieval_attempts"),
            }

            st.markdown(answer_message["content"])

            if answer_message["sources"]:
                with st.expander("Sources"):
                    for source in answer_message["sources"]:
                        st.write(f"- {source}")

            attempts = answer_message["retrieval_attempts"]

            if attempts is not None:
                st.caption(f"Retrieval attempts: {attempts}")

            st.session_state.messages.append(answer_message)

        except requests.RequestException as exc:
            error_message = (
                "I couldn't get a response from the backend. "
                "Check that FastAPI and Ollama are running."
            )

            st.error(error_message)
            st.caption(str(exc))

            # The backend may already have saved the user message.
            # Do not save a fabricated assistant answer to SQLite.
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                }
            )
