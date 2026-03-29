import os
import glob
# new stuff start
import pickle
from rank_bm25 import BM25Okapi
# new stuff end
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential
from tqdm import tqdm
from litellm import completion
from dotenv import load_dotenv
from dataclasses import dataclass, field


@dataclass
class Document:
    page_content: str
    metadata: dict = field(default_factory=dict)

load_dotenv(override=True)

MODEL = "gemini/gemini-2.5-flash-lite"
DB_NAME = str(Path(__file__).parent / "vector_db")
KNOWLEDGE_BASE = str(Path(__file__).parent / "knowledge-base")
BM25_PATH = str(Path(__file__).parent / "bm25_index.pkl")
AVERAGE_CHUNK_SIZE = 200

wait = wait_exponential(multiplier=1, min=10, max=240)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    output_dimensionality=768,
    task_type="RETRIEVAL_DOCUMENT"
)
class Chunk(BaseModel):
    headline: str = Field(
        description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query"
    )
    summary: str = Field(
        description="A few sentences summarizing the content of this chunk to answer common questions"
    )
    original_text: str = Field(
        description="The original text of this chunk from the provided document, exactly as is"
    )

    def as_document(self, doc_type: str, source: str):
        from langchain_core.documents import Document
        return Document(
            page_content=self.headline + "\n\n" + self.summary + "\n\n" + self.original_text,
            metadata={"doc_type": doc_type, "source": source}
        )

class Chunks(BaseModel):
    chunks: list[Chunk]


def make_prompt(document: Document) -> str:
    how_many = (len(document.page_content) // AVERAGE_CHUNK_SIZE) + 1
    return f"""
You take a document and split it into overlapping chunks for a KnowledgeBase.
The document is of type: {document.metadata['doc_type']}
The document source: {document.metadata.get('source', 'unknown')}

A chatbot will use these chunks to answer questions about Insurellm.
Split into at least {how_many} chunks with ~25% overlap (~50 words).

Here is the document:
{document.page_content}

Respond with valid JSON in this exact format:
{{
  "chunks": [
    {{
      "headline": "Brief heading here",
      "summary": "Few sentences summarizing this chunk",
      "original_text": "Exact original text from document"
    }}
  ]
}}
"""

@retry(wait=wait, reraise=True, before_sleep=lambda retry_state: print(f"Retrying after error: {retry_state.outcome.exception()}"))
def process_document(document):
    response = completion(
        model=MODEL,  # cheap model for chunking
        messages=[{"role": "user", "content": make_prompt(document)}],
        response_format=Chunks
    )
    raw_chunks = Chunks.model_validate_json(
        response.choices[0].message.content
    ).chunks
    return [
        chunk.as_document(
            document.metadata["doc_type"],
            document.metadata.get("source", "")
        )
        for chunk in raw_chunks
    ]

def fetch_documents() -> list:
    documents = []
    for path in glob.glob(str(Path(KNOWLEDGE_BASE) / "*")):
        if os.path.isdir(path):
            doc_type = os.path.basename(path)
            loader = DirectoryLoader(path, glob="**/*.md", loader_cls=TextLoader,
                                     loader_kwargs={"encoding": "utf-8"})
            for doc in loader.load():
                doc.metadata["doc_type"] = doc_type
                documents.append(doc)
        elif path.endswith(".md"):
            doc_type = os.path.splitext(os.path.basename(path))[0]
            loader = TextLoader(path, encoding="utf-8")
            for doc in loader.load():
                doc.metadata["doc_type"] = doc_type
                documents.append(doc)
    return documents

def create_chunks(documents):
    chunks = []
    for doc in tqdm(documents):
        result = process_document(doc)
        chunks.extend(result)
    print(f"Total chunks created: {len(chunks)}")
    return chunks

def create_embeddings(chunks):
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_NAME,
    )

    collection = vectorstore._collection
    count = collection.count()
    sample = collection.get(limit=1, include=["embeddings"])["embeddings"]
    dimensions = len(sample[0]) if len(sample) > 0 else 0
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
    return vectorstore

def create_bm25_index(chunks):  # NEW — bas yahi add hua
    tokenized = [chunk.page_content.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized)
    # BM25 + original chunks dono save karo — retrieval mein chunks chahiye
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
    print(f"BM25: {len(chunks)} chunks indexed")


if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    if not chunks:
        print("No chunks created — check knowledge-base path and contents.")
    else:
        create_embeddings(chunks)
        create_bm25_index(chunks)
        print("Ingestion complete")
