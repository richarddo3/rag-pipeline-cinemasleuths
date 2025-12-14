from rag_pipeline.ingest import load_csv_documents, load_directory_texts
from rag_pipeline.vector_store import build_faiss_index
from rag_pipeline.retriever import retrieve_top_k
from rag_pipeline.embeddings import get_embedding_model

from langchain_text_splitters import RecursiveCharacterTextSplitter


# -----------------------------
# CHUNKING FUNCTION (no separate file)
# -----------------------------
def chunk_documents(documents, chunk_size=800, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    texts = [doc["text"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]

    chunks = splitter.create_documents(texts, metadatas=metadatas)
    return chunks


# -----------------------------
# BUILD THE FULL RAG PIPELINE
# -----------------------------
def build_rag_pipeline():
    csv_path = "data/etl_cleaned_dataset.csv"
    extra_docs_dir = "data/additional_documents"

    print("Loading CSV documents...")
    csv_docs = load_csv_documents(csv_path)

    print("Loading additional .txt documents...")
    extra_docs = load_directory_texts(extra_docs_dir)

    all_docs = csv_docs + extra_docs
    print(f"Loaded {len(all_docs)} total raw documents.")

    print("Chunking documents...")
    chunks = chunk_documents(all_docs)
    print(f"Created {len(chunks)} chunks.")

    print("Building FAISS index...")
    index, embeddings, metadata = build_faiss_index(chunks)

    return index, embeddings, metadata


# -----------------------------
# ASK FUNCTION – combines retrieval + LLM
# -----------------------------
from openai import OpenAI
client = OpenAI()


def answer_question(index, embeddings, metadata, question, k=4):
    results = retrieve_top_k(index, question, metadata, k=k)

    context = ""
    for r in results:
        context += f"Source: {r['metadata']}\n"
        context += f"Distance: {r['distance']}\n\n"

    prompt = f"""
You are a domain-specific assistant.
Answer ONLY using the context below.
If the answer is not present, say "The dataset does not contain that information."

CONTEXT:
{context}

QUESTION: {question}
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    answer = completion.choices[0].message["content"]

    return {
        "answer": answer,
        "sources": [r["metadata"] for r in results]
    }
