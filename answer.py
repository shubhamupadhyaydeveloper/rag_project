import torch
import os
import time
import pickle
import logging
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from langfuse_tracker import langfuse, log_hyde, log_retrieval, log_rerank, log_generation, log_rag_trace


load_dotenv(override=True)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

MODEL = "gemini-2.5-flash-lite"
EMBEDDING_MODEL = "gemini-embedding-2-preview"
DB_NAME = str(Path(__file__).parent / "vector_db")
BM25_PATH = str(Path(__file__).parent / "bm25_index.pkl")

# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    output_dimensionality=768,
    task_type="RETRIEVAL_QUERY",
)

RETRIEVAL_K = 20  # pehle 20 laao dono se
RERANK_TOP_N = 7  # reranker ke baad sirf 5 rakhne hain

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vectorstore.as_retriever()
llm = ChatGoogleGenerativeAI(temperature=0, model=MODEL)

# Cross-encoder — same as week 5 reranker, free model
device = "mps" if torch.backends.mps.is_available() else "cpu"
reranker = CrossEncoder("BAAI/bge-reranker-base", device=device)

# BM25 load karo
with open(BM25_PATH, "rb") as f:
    data = pickle.load(f)
    bm25 = data["bm25"]
    bm25_chunks = data["chunks"]


def bm25_search(query: str, k: int) -> list[Document]:
    """BM25 keyword search"""
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    # top-k indices by score
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [bm25_chunks[i] for i in top_indices]


def dense_search(query: str, k: int) -> list[Document]:
    """ChromaDB semantic search — same as before"""
    return vectorstore.similarity_search(query, k=k)


def reciprocal_rank_fusion(
    bm25_docs: list[Document], dense_docs: list[Document], k: int = 60
) -> list[Document]:
    """
    RRF — dono lists ko merge karo.
    k=60 is standard constant, don't worry about it.
    Higher score = better rank.
    """
    scores = {}
    # page_content as key (same doc dono lists mein aa sakta hai)
    for rank, doc in enumerate(bm25_docs):
        key = doc.page_content
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)

    for rank, doc in enumerate(dense_docs):
        key = doc.page_content
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)

    # unique docs, sorted by combined score
    all_docs = {doc.page_content: doc for doc in bm25_docs + dense_docs}
    sorted_keys = sorted(scores, key=scores.get, reverse=True)
    return [all_docs[key] for key in sorted_keys]


def rerank(query: str, docs: list[Document], top_n: int) -> list[Document]:
    """Cross-encoder — query + chunk ek saath score karo"""
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_n]]


def hyde_query(question: str) -> str:
    """Real search se pehle fake answer generate karo"""
    response = llm.invoke(
        [
            SystemMessage(
                content="""You are answering questions about Insurellm — 
an insurance company with products like AutoSure, HomeSure, LifeSure.
Generate a short answer as if you know the specifics."""
            ),
            HumanMessage(content=question),
        ]
    )
    return response.content


# def expand_query(question: str) -> list[str]:
#     """Ek query ke 3 variations banao"""
#     response = llm.invoke([
#         SystemMessage(content="""Generate 3 different versions of this question.
#         Return only the questions, one per line, nothing else."""),
#         HumanMessage(content=question)
#     ])
#     variants = response.content.strip().split("\n")
#     return [question] + variants[:3]  # original + 3 variants


def fetch_context(question: str) -> list[Document]:
    hyde_start = time.time()
    with langfuse.start_as_current_observation(name="hyde", as_type="span") as hyde_span:
        hyde = hyde_query(question)
        log_hyde(hyde_span, hyde, round((time.time() - hyde_start) * 1000))

    retrieval_start = time.time()
    with langfuse.start_as_current_observation(name="hybrid-retrieval", as_type="retriever") as retrieval_span:
        all_bm25_docs = bm25_search(question, k=RETRIEVAL_K)
        all_dense_docs = dense_search(question, k=RETRIEVAL_K)
        all_dense_docs.extend(dense_search(hyde, k=10))
        fused = reciprocal_rank_fusion(all_bm25_docs, all_dense_docs)
        log_retrieval(retrieval_span, all_bm25_docs, all_dense_docs, fused, round((time.time() - retrieval_start) * 1000))

    rerank_start = time.time()
    with langfuse.start_as_current_observation(name="reranking", as_type="span") as rerank_span:
        final = rerank(question, fused, top_n=RERANK_TOP_N)
        log_rerank(rerank_span, final, round((time.time() - rerank_start) * 1000))

    return final


def combined_question(question: str, history: list[dict] = None) -> str:
    """
    Combine all the user's messages into a single string.
    """
    if history is None:
        history = []
    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    return prior + "\n" + question

def build_context_with_citations(docs: list) -> str:
    context_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        context_parts.append(
            f"[{i}] Source: {source}\n{doc.page_content}"
        )
    return "\n\n".join(context_parts)

def answer_question(
    question: str, history: list[dict] = None
) -> tuple[str, list[Document]]:
    if history is None:
        history = []
    total_start = time.time()

    with langfuse.start_as_current_observation(
        name="rag-query", as_type="span",
        input={"question": question},
        metadata={"model": MODEL, "rerank_top_n": RERANK_TOP_N}
    ) as trace:
        combined = combined_question(question, history)
        docs = fetch_context(combined)

        gen_start = time.time()
        context = build_context_with_citations(docs)
        system_prompt = SYSTEM_PROMPT.format(context=context)
        messages = [SystemMessage(content=system_prompt)]
        messages.extend(convert_to_messages(history))
        messages.append(HumanMessage(content=question))

        with langfuse.start_as_current_observation(name="generation", as_type="generation") as gen_span:
            response = llm.invoke(messages)
            answer = response.content
            ttft_ms = round((time.time() - gen_start) * 1000)
            output_tokens = len(answer.split())
            log_generation(gen_span, answer, ttft_ms, output_tokens)

        total_ms = round((time.time() - total_start) * 1000)
        log_rag_trace(trace, answer, ttft_ms, total_ms, docs, system_prompt, question)

    return answer, docs
