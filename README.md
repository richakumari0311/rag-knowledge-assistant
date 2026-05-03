# RAG Knowledge Assistant

A full end-to-end Retrieval-Augmented Generation (RAG) pipeline built with LangChain.
Ask questions over your own documents with source attribution.

## Tech Stack

| Component    | Tool                              |
|--------------|-----------------------------------|
| Framework    | LangChain                         |
| Vector Store | FAISS                             |
| Embeddings   | HuggingFace all-MiniLM-L6-v2 (free) |
| LLM          | Ollama llama3.2 (free, local)     |
| UI           | Streamlit                         |

## Default Stack - 100% Free, No API Key Needed

- Embeddings run locally via HuggingFace sentence-transformers
- LLM runs locally via Ollama
- No OpenAI or paid API required

Want better quality answers? You can optionally switch to OpenAI.
See the "Using OpenAI Instead" section below.

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/richakumari0311/rag-knowledge-assistant.git
cd rag-knowledge-assistant
```

**2. Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**3. Configure**
```bash
cp .env.example .env
```

**4. Install and start Ollama**
```bash
brew install ollama
ollama serve
ollama pull llama3.2
```

**5. Add your documents**

Put .pdf or .txt files in data/sample_docs/

**6. Run**
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Project Structure

```
rag-knowledge-assistant/
├── app.py                  # Streamlit UI
├── src/
│   └── rag_pipeline.py     # Core RAG pipeline
├── data/
│   └── sample_docs/        # Your documents go here
├── requirements.txt
├── .env.example
└── .env                    # your local config (gitignored)
```

## How It Works

```
Documents -> Chunking -> Embeddings -> FAISS Vector Store
                                             |
User Question -> Retriever (Top-K search) -> LLM -> Answer + Sources
```

## Using OpenAI Instead (Optional)

The pipeline supports OpenAI as a drop-in replacement for both the LLM
and embeddings. No code changes needed - just update your .env file.

**1. Install the OpenAI package**
```bash
pip install langchain-openai openai
```

**2. Update .env**
```
LLM_BACKEND=openai
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx

EMBEDDING_BACKEND=openai
```

**3. Re-ingest your documents** (needed because embeddings change)
```bash
rm -rf data/vectorstore
streamlit run app.py
```

Note: OpenAI charges per API call. For most personal projects the cost
is very small, but keep that in mind.

## All Config Options

| Variable           | Options                        | Default       |
|--------------------|--------------------------------|---------------|
| LLM_BACKEND        | ollama / openai                | ollama        |
| OLLAMA_MODEL       | llama3.2, mistral, phi3        | llama3.2      |
| EMBEDDING_BACKEND  | huggingface / openai           | huggingface   |
| VECTOR_STORE       | faiss                          | faiss         |
| CHUNK_SIZE         | any number                     | 1000          |
| CHUNK_OVERLAP      | any number                     | 200           |
| TOP_K_RESULTS      | any number                     | 4             |


<!-- source venv/bin/activate
ollama serve    # in a separate terminal
streamlit run app.py -->