import os
from langfuse import Langfuse

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
)


def log_hyde(span, hypothesis: str, latency_ms: int):
    span.update(output={"hypothesis": hypothesis[:300], "latency_ms": latency_ms})


def log_retrieval(span, bm25_docs, dense_docs, fused_docs, latency_ms: int):
    span.update(output={
        "bm25_count": len(bm25_docs),
        "dense_count": len(dense_docs),
        "fused_count": len(fused_docs),
        "latency_ms": latency_ms,
    })


def log_rerank(span, final_docs, latency_ms: int):
    span.update(output={
        "chunks_kept": len(final_docs),
        "sources": [d.metadata.get("source", "") for d in final_docs],
        "latency_ms": latency_ms,
    })


def log_generation(span, answer: str, ttft_ms: int, output_tokens: int):
    span.update(output={"answer": answer[:300], "ttft_ms": ttft_ms, "output_tokens_approx": output_tokens})


def log_rag_trace(trace, answer: str, ttft_ms: int, total_ms: int, docs, system_prompt: str, question: str):
    prompt_tokens = len(system_prompt.split()) + len(question.split())
    trace.update(
        output={"answer": answer},
        metadata={
            "ttft_ms": ttft_ms,
            "total_latency_ms": total_ms,
            "chunks_used": len(docs),
            "prompt_tokens_approx": prompt_tokens,
            "output_tokens_approx": len(answer.split()),
            "sources": [d.metadata.get("source", "") for d in docs],
        }
    )
