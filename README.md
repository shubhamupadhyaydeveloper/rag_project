# Personal Knowledge Base RAG System

A production-style Retrieval-Augmented Generation (RAG) pipeline that answers questions from a personal knowledge base using Google Gemini, hybrid search, and cross-encoder reranking — with full observability via Langfuse and automated evaluation via RAGAS.

---

## Architecture

```
Documents (Markdown)
        │
        ▼
┌─────────────────────┐
│  Intelligent Chunker │  ← Gemini 2.5 Flash Lite (via LiteLLM)
│  headline + summary  │    25% overlap, semantic boundaries
└─────────────────────┘
        │
        ▼
┌────────────────────────────────────────┐
│            Dual Index                  │
│  ChromaDB (dense, 768-dim embeddings)  │
│  BM25     (sparse, keyword-based)      │
└────────────────────────────────────────┘
        │
        ▼  Query time
┌─────────────────────────────────────────────────────┐
│  1. HyDE  — generate hypothetical answer with Gemini │
│  2. Hybrid Retrieval — BM25 (k=20) + Dense (k=20+10) │
│  3. Reciprocal Rank Fusion — merge both result lists  │
│  4. CrossEncoder Reranker — keep top 7               │
│  5. Generation — Gemini 2.5 Flash Lite + context     │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────┐
│  Streamlit Chat UI           │
│  Langfuse Observability      │
│  RAGAS Automated Evaluation  │
└──────────────────────────────┘
```

---

## Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| LLM | Google Gemini 2.5 Flash Lite | Answer generation, HyDE hypothesis, chunking |
| Embeddings | Gemini Embedding 2 Preview (768-dim) | Semantic vector search |
| Vector DB | ChromaDB | Persistent dense retrieval |
| Keyword Search | BM25 (rank_bm25) | Sparse keyword-based retrieval |
| Fusion | Reciprocal Rank Fusion | Merge and re-rank BM25 + dense results |
| Reranker | BAAI/bge-reranker-base (CrossEncoder) | Precision re-scoring of top candidates |
| Chunking LLM | LiteLLM → Gemini | Intelligent semantic chunking with structured output |
| Observability | Langfuse | Per-stage latency tracing (HyDE, retrieval, rerank, generation) |
| Evaluation | RAGAS | Faithfulness, Answer Relevancy, Context Precision, Context Recall |
| UI | Streamlit | Chat interface with source attribution |

---

## Pipeline Deep Dive

### 1. Ingestion (`ingest.py`)
- Loads markdown files from `knowledge-base/`
- Uses Gemini to chunk each document into semantically coherent pieces with a `headline`, `summary`, and `original_text`
- Stores chunks in **ChromaDB** (dense) and a **BM25 index** (sparse, pickled to `bm25_index.pkl`)

### 2. Retrieval (`answer.py`)
- **HyDE**: Given a user question, first generate a hypothetical answer using Gemini — this makes the embedding search more effective
- **Hybrid search**: BM25 keyword search + two dense searches (original question + HyDE answer), k=20 each
- **RRF**: Reciprocal Rank Fusion merges all three result lists into a single ranked list
- **Reranking**: CrossEncoder (`BAAI/bge-reranker-base`) scores every (query, chunk) pair and keeps top 7

### 3. Generation
- Top 7 chunks are concatenated as context
- Gemini generates the final answer using a system prompt
- All stages are traced in Langfuse with latency metrics

---

## Evaluation Results

### Retrieval Quality
| Metric | Score |
|---|---|
| MRR (Mean Reciprocal Rank) | **0.91** |
| nDCG (Normalized Discounted Cumulative Gain) | **0.89** |

### RAGAS (12 test cases — support tickets, product info, financial data, technical topics)

| Metric | Score | What it measures |
|---|---|---|
| Faithfulness | **1.0000** | Are answers grounded in the retrieved context? |
| Answer Relevancy | **0.9707** | Is the answer relevant to the question asked? |
| Context Precision | **0.6932** | Are retrieved chunks actually useful for answering? |
| Context Recall | — | Are all ground-truth facts present in retrieved context? |
| **Average** | **0.8880** | Mean over 3 scored metrics |

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_google_api_key
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
HF_TOKEN=your_huggingface_token
```

### 3. Add your documents
Place markdown files in the `knowledge-base/` directory.

### 4. Ingest documents
```bash
python ingest.py
```
This builds the ChromaDB vector store and `bm25_index.pkl`.

### 5. Run the Streamlit app
```bash
streamlit run app.py
```

### 6. Run RAGAS evaluation
```bash
python ragas_eval.py
```

---

## Project Structure

```
rag_project/
├── knowledge-base/      # Source markdown documents
├── vector_db/           # ChromaDB persistent store
├── prompts/
│   ├── system.yaml      # Answer generation system prompt
│   └── chunking.yaml    # Document chunking prompt
├── ingest.py            # Document ingestion pipeline
├── answer.py            # RAG query pipeline
├── app.py               # Streamlit chat UI
├── ragas_eval.py        # RAGAS automated evaluation
├── langfuse_tracker.py  # Observability helpers
├── bm25_index.pkl       # Serialized BM25 index
└── requirements.txt
```
