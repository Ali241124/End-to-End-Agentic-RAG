
import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="Agentic RAG",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Agentic RAG")
st.caption("Ask questions about your documents using AI.")

# Store chat messages in this browser session.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar controls.
with st.sidebar:
    st.header("Knowledge Base")
    st.write("Manage the documents used by your AI assistant.")

    if st.button("📚 Ingest Documents", use_container_width=True):
        try:
            with st.spinner("Processing documents..."):
                response = requests.post(
                    f"{API_URL}/ingest",
                    timeout=300,
                )
                response.raise_for_status()
                result = response.json()

            st.success(result["message"])
            st.write(f"Chunks processed: {result['total_chunks']}")

        except requests.RequestException as exc:
            st.error(f"Could not ingest documents: {exc}")

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Backend: FastAPI")
    st.caption("Agent: LangGraph")
    st.caption("Vector database: ChromaDB")


# Display existing conversation messages.
for message in st.session_state.messages:
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


# Input box for a new question.
question = st.chat_input("Ask a question about your documents...")

if question:
    # Display and save the user's question.
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Ask the FastAPI backend.
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating an answer..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question},
                    timeout=300,
                )
                response.raise_for_status()
                result = response.json()

                answer = result["answer"]
                sources = result.get("sources", [])
                attempts = result.get("retrieval_attempts")

                st.markdown(answer)

                if sources:
                    with st.expander("Sources"):
                        for source in sources:
                            st.write(f"- {source}")

                if attempts is not None:
                    st.caption(f"Retrieval attempts: {attempts}")

                # Save the complete assistant response.
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "retrieval_attempts": attempts,
                    }
                )

            except requests.RequestException as exc:
                error_message = (
                    "I couldn't connect to the backend or get an answer. "
                    "Check that FastAPI and Ollama are running."
                )
                st.error(error_message)
                st.caption(str(exc))

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )
