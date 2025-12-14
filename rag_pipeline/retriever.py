import numpy as np
from .embeddings import get_embedding_model

# Load embedding model once
embedder = get_embedding_model()

def embed_query(text):
    """Return embedding vector for a single query."""
    return embedder.encode([text])[0]

def retrieve_top_k(index, query, metadata, k=4):
    """
    index: FAISS index
    query: user question (string)
    metadata: list of metadata dicts corresponding to FAISS IDs
    k: number of neighbors
    """

    # Get embedding for the query
    q_emb = embed_query(query)
    q_emb = np.array([q_emb]).astype("float32")

    # FAISS search
    distances, indices = index.search(q_emb, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({
            "metadata": metadata[idx],
            "distance": float(dist)
        })

    return results
