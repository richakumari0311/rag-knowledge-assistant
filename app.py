import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
BASE_DIR = Path(__file__).parent
SRC_PATH = BASE_DIR / "src"
sys.path.insert(0, str(SRC_PATH))

from rag_pipeline import RAGAssistant


# PAGE CONFIG
st.set_page_config(
    page_title="RAG Knowledge Assistant",
    layout="wide",
)

# SESSION STATE
if "assistant" not in st.session_state:
    st.session_state.assistant = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "ready" not in st.session_state:
    st.session_state.ready = False


# INIT ASSISTANT
def initialize_assistant(load_existing: bool = False) -> None:
    """Initialize or load the RAG assistant."""
    try:
        assistant = RAGAssistant()

        if load_existing:
            assistant.load()
        else:
            num_chunks = assistant.ingest()
            st.success(f"Indexed {num_chunks} chunks!")

        assistant.setup_chain()
        st.session_state.assistant = assistant
        st.session_state.ready = True

        if load_existing:
            st.success("Index loaded!")

    except Exception as error:
        st.error(f"Error: {error}")


# SIDEBAR
with st.sidebar:
    st.title("Settings")

    st.subheader("Ollama Model")
    model = st.selectbox("Choose model", ["llama3.2", "mistral", "phi3"])
    os.environ["OLLAMA_MODEL"] = model

    st.subheader("Retrieval")
    top_k = st.slider("Top-K chunks", 1, 8, 4)
    os.environ["TOP_K_RESULTS"] = str(top_k)

    st.divider()

    # Ingest button
    if st.button("Ingest Documents", type="primary"):
        with st.spinner("Loading and indexing documents..."):
            initialize_assistant(load_existing=False)

    # Load existing
    if st.button("Load Existing Index"):
        with st.spinner("Loading index..."):
            initialize_assistant(load_existing=True)

    st.divider()
    st.caption("RAG Assistant | LangChain + FAISS")


# MAIN
st.title("RAG Knowledge Assistant")
st.caption("Ask questions grounded in your own documents")

if not st.session_state.ready:
    st.info(
        "Click Ingest Documents or Load Existing Index in the sidebar to get started."
    )

else:
        # PDF Upload section
    with st.expander("Upload a new document"):
        uploaded_file = st.file_uploader(
            "Upload a PDF or TXT file",
            type=["pdf", "txt"],
        )
        if uploaded_file:
            from pathlib import Path
            save_path = Path("data/sample_docs") / uploaded_file.name
            save_path.write_bytes(uploaded_file.read())
            st.success(f"Saved {uploaded_file.name}")

            if st.button("Re-ingest with new document"):
                with st.spinner("Re-ingesting all documents..."):
                    try:
                        import shutil
                        shutil.rmtree("data/vectorstore", ignore_errors=True)
                        assistant = RAGAssistant()
                        n = assistant.ingest()
                        assistant.setup_chain()
                        st.session_state.assistant = assistant
                        st.session_state.ready = True
                        st.success(f"Done! Indexed {n} chunks.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

    # Chat history display
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

            if msg.get("sources"):
                with st.expander("Sources"):
                    for src in msg["sources"]:
                        st.write(f"- {src}")

    # Chat input
    question = st.chat_input("Ask something about your documents...")

    if question:
        # User message
        with st.chat_message("user"):
            st.write(question)

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        # Assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.assistant.ask(question)

                    answer = result.get("answer", "")
                    sources = result.get("sources", [])

                    st.write(answer)

                    if sources:
                        with st.expander("Sources"):
                            for src in sources:
                                st.write(f"- {src}")

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )

                except Exception as error:
                    st.error(f"Error: {error}")

    # Clear chat
    if st.session_state.chat_history:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()