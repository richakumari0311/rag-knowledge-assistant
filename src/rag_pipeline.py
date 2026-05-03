import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings,ChatOpenAI


load_dotenv()

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

# LOAD DOCUMENTS
def load_documents(data_dir: str):
    """Load all PDF and TXT files from a folder."""
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Folder not found: {data_dir}")

    docs = []

    try:
        txt_loader = DirectoryLoader(
            str(path),
            glob="**/*.txt",
            loader_cls=TextLoader,
        )
        docs.extend(txt_loader.load())
        logger.info("Loaded TXT files")
    except Exception as exc:
        logger.warning(f"TXT issue: {exc}")

    try:
        pdf_loader = DirectoryLoader(
            str(path),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
        )
        docs.extend(pdf_loader.load())
        logger.info("Loaded PDF files")
    except Exception as exc:
        logger.warning(f"PDF issue: {exc}")

    logger.info(f"Total documents: {len(docs)}")
    return docs


# CHUNK DOCUMENTS
def chunk_documents(docs):
    """Split documents into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200)),
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    logger.info(
        "Created %d chunks from %d documents",
        len(chunks),
        len(docs),
    )
    return chunks


# EMBEDDINGS
def get_embeddings():
    """Return the embedding model based on .env config."""
    backend = os.getenv("EMBEDDING_BACKEND", "huggingface")

    if backend == "openai":
        logger.info("Using OpenAI embeddings")
        return OpenAIEmbeddings()

    else:
        model = os.getenv("HF_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        logger.info(f"Using HuggingFace embeddings: {model}")
        return HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

# VECTOR STORE
def build_vector_store(chunks, embeddings, persist_dir: str):
    """Build and save FAISS or ChromaDB vector store."""
    backend = os.getenv("VECTOR_STORE", "faiss").lower()
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    if backend == "chroma":
        from langchain_community.vectorstores import Chroma
        logger.info("Building ChromaDB vector store...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir,
        )
    else:
        from langchain_community.vectorstores import FAISS
        logger.info("Building FAISS vector store...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(os.path.join(persist_dir, "faiss_index"))

    logger.info(f"Saved to: {persist_dir}")
    return vectorstore


def load_vector_store(embeddings, persist_dir: str):
    """Load existing FAISS or ChromaDB vector store."""
    backend = os.getenv("VECTOR_STORE", "faiss").lower()

    if backend == "chroma":
        from langchain_community.vectorstores import Chroma
        logger.info("Loading ChromaDB vector store...")
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )
    else:
        from langchain_community.vectorstores import FAISS
        logger.info("Loading FAISS vector store...")
        faiss_path = os.path.join(persist_dir, "faiss_index")
        return FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)

# RAG CHAIN
PROMPT_TEMPLATE = """You are a helpful assistant. Use the context below to answer the question.
If the answer is not in the context, say "I don't have that information in the documents."
Always mention which document your answer comes from.

Context:
{context}

Question: {question}

Answer:"""


def build_rag_chain(vectorstore, llm):
    """Build the retrieval chain."""
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": int(os.getenv("TOP_K_RESULTS", 4))},
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


# LLM
def get_llm():
    """Return the LLM based on .env config."""
    backend = os.getenv("LLM_BACKEND", "ollama")

    if backend == "openai":
        logger.info("Using OpenAI LLM")
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    else:
        model = os.getenv("OLLAMA_MODEL", "llama3.2")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        logger.info(f"Using Ollama: {model}")
        return OllamaLLM(model=model, base_url=base_url)

# MAIN ASSISTANT CLASS
class RAGAssistant:
    def __init__(self):
        self.data_dir = os.getenv("DATA_DIR", "data/sample_docs")
        self.vectorstore_dir = os.getenv(
            "VECTORSTORE_DIR",
            "data/vectorstore",
        )

        self.embeddings = None
        self.vectorstore = None
        self.chain = None

    def ingest(self):
        """Load documents → chunk → embed → store."""
        self.embeddings = get_embeddings()

        docs = load_documents(self.data_dir)
        chunks = chunk_documents(docs)

        self.vectorstore = build_vector_store(
            chunks,
            self.embeddings,
            self.vectorstore_dir,
        )

        return len(chunks)

    def load(self):
        """Load existing vector store from disk."""
        self.embeddings = get_embeddings()
        self.vectorstore = load_vector_store(
            self.embeddings,
            self.vectorstore_dir,
        )

    def setup_chain(self):
        """Set up LLM + retrieval chain."""
        llm = get_llm()
        self.chain = build_rag_chain(self.vectorstore, llm)

    def ask(self, question: str):
        """Ask a question, get answer + sources."""
        chain, retriever = self.chain

        answer = chain.invoke(question)
        source_docs = retriever.invoke(question)

        sources = list(
            dict.fromkeys(
                doc.metadata.get("source", "Unknown")
                for doc in source_docs
            )
        )

        return {"answer": answer, "sources": sources}