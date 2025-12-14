from .embeddings import embed_texts
from .vector_store import search_faiss

def retrieve_top_k(index, docs, query, k=5):

    # Embed the query
    q_emb = embed_texts([query])[0]

    # Perform vector search
    distances, idxs = search_faiss(index, q_emb, k=k)

    results = []
    for dist, i in zip(distances, idxs):
        doc = docs[i]  # ← CORRECT: pull the chunked document
        
        results.append({
            "id": doc["id"],
            "text": doc["text"],          # ← THIS is the important part
            "metadata": doc["metadata"],
            "distance": float(dist),
        })

    return results
