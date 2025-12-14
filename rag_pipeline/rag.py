from .ingest import load_csv_documents, load_directory_texts, chunk_documents
from .embeddings import get_embedding_model, embed_texts
from .vector_store import build_faiss_index
from .retriever import retrieve_top_k

def build_rag_pipeline(csv_path, extra_docs_dir=None):
    docs = load_csv_documents(csv_path)

    if extra_docs_dir:
        docs += load_directory_texts(extra_docs_dir)

    docs = chunk_documents(docs)

    model = get_embedding_model()
    texts = [d["text"] for d in docs]
    embeddings = embed_texts(model, texts)

    index = build_faiss_index(embeddings)

    return {
        "model": model,
        "index": index,
        "documents": docs
    }


def answer_question(pipeline, question, k=3):
    model = pipeline["model"]
    index = pipeline["index"]
    docs = pipeline["documents"]

    query_emb = model.encode([question])[0]
    retrieved = retrieve_top_k(index, query_emb, docs, k=k)

    context = "\n\n".join([d["text"] for d in retrieved])

    response = f"ANSWER BASED ON DOCUMENTS:\n\n{context}"
    return response, retrieved
