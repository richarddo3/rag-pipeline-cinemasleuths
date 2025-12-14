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
    docs = chunk_documents(docs)

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

    context = "\n\n".join([r["text"] for r in retrieved])

    answer = f"ANSWER BASED ON RETRIEVED DOCUMENTS:\n\n{context}"

    return {
        "answer": answer,
        "sources": [
            {"id": r["id"], "distance": r["distance"]} for r in retrieved
        ]
    }
