from .embeddings import embed_texts
from .vector_store import search_faiss

def retrieve_top_k(index, docs, query, k=5):

    q_emb = embed_texts([query])[0]
    distances, idxs = search_faiss(index, q_emb, k=k)

    results = []
    for dist, i in zip(distances, idxs):
        d = docs[i]
        results.append({
            "id": d["id"],
            "text": d["text"],
            "metadata": d["metadata"],
            "distance": float(dist)
        })

    return results
