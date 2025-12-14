from .embeddings import embed_texts
from .vector_store import search_faiss


def retrieve_top_k(index, query, docs, k=5):
    """
    Retrieve the top-k closest document chunks.

    Parameters:
        index  – FAISS index
        query  – user input question (string)
        docs   – list of document chunks
        k      – number of results to return
    """

    # Embed user question
    q_emb = embed_texts([query])[0]

    # Search index
    distances, idxs = search_faiss(index, q_emb, k=k)

    results = []
    for dist, i in zip(distances, idxs):
        doc = docs[i]  # doc contains id, text, metadata

        results.append({
            "id": doc["id"],
            "text": doc["text"],
            "metadata": doc["metadata"],
            "distance": float(dist)
        })

    return results
