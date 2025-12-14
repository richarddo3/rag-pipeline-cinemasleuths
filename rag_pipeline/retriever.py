from .embeddings import embed_texts
from .vector_store import search_faiss

def retrieve(query, docs, index, k=5):
    q_emb = embed_texts([query])[0]
    distances, idxs = search_faiss(index, q_emb, k=k)
    
    results = []
    for dist, i in zip(distances, idxs):
        d = docs[i]
        results.append({
            "text": d["text"],
            "source": d["source"],
            "distance": float(dist)
        })
    return results
