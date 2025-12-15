from .embeddings import embed_texts
from .vector_store import search_faiss

def retrieve_top_k(index, docs, query, k=5):
    q_emb = embed_texts([query])[0]
    distances, idxs = search_faiss(index, q_emb, k=k)

    results = []
    for dist, idx in zip(distances, idxs):

        # Skip FAISS padding (-1)
        if idx < 0 or idx >= len(docs):
            continue

        doc = docs[idx]

        # Skip corrupted docs
        if not isinstance(doc, dict):
            print("⚠️ WARNING: Skipping bad doc:", doc)
            continue

        results.append({
            "id": doc["id"],
            "text": doc["text"],
            "metadata": doc["metadata"],
            "distance": float(dist)
        })

    return results
