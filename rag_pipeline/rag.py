from .ingest import load_csv_documents, load_directory_texts, chunk_documents
from .embeddings import embed_texts
from .vector_store import build_faiss_index
from .retriever import retrieve_top_k


def build_rag_pipeline(
    csv_path="data/etl_cleaned_dataset.csv",
    extra_docs_dir="data/additional_documents"
):
    print("Loading CSV...")
    docs = load_csv_documents(csv_path)

    print("Loading extra documents...")
    if extra_docs_dir:
        docs += load_directory_texts(extra_docs_dir)

    print("Chunking...")
    docs = chunk_documents(docs, chunk_size=5000, chunk_overlap=200)

    print("Embedding...")
    texts = [d["text"] for d in docs]
    embeddings = embed_texts(texts)

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    return {
        "index": index,
        "documents": docs
    }


def answer_question(pipeline, question, k=4):
    docs = pipeline["documents"]
    index = pipeline["index"]

    retrieved = retrieve_top_k(index, docs, question, k=k)

    context = "\n\n".join(r["text"] for r in retrieved)

    prompt = f"""You are a movie data assistant.
Answer the user's question **using ONLY the context below**.
If the answer is not in the context, say "I cannot answer from the provided data."

Question:
{question}

Context:
{context}

Answer:
"""

    # use small LLM (Qwen, Phi, etc)
    from openai import OpenAI
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message["content"]

    return {
        "answer": answer,
        "sources": retrieved
    }
