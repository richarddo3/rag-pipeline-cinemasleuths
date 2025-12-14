from .embeddings import embed_texts
from .vector_store import search_faiss

def retrieve_top_k(index, query, metadata, k=5):
    """
    Returns the top-k closest documents to the query.
    
    Parameters:
        index     – FAISS index
        query     – user question as a string
        metadata  – list of metadata entries (same order as chunks)
        k         – number of results

    Returns:
        List of dictionaries containing:
            - metadata
            - distance
    """
    
    # Embed the query
    q_emb = embed_texts([query])[0]

    # Search FAISS for k-nearest vectors
    distances, idxs = search_faiss(index, q_emb, k=k)

    results = []
    for dist, i in zip(distances, idxs):
        results.append({
            "metadata": metadata[i],   # matches your rag.py usage
            "distance": float(dist)
        })

    return results
