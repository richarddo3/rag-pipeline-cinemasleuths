import faiss
import numpy as np
from rag_pipeline.embeddings import get_embedding_model

def build_faiss_index(chunks):
    """
    chunks: list of Document-like dicts:
        { "text": "...", "metadata": {...} }
    Returns: (faiss_index, embeddings_matrix, metadata_list)
    """
    embedder = get_embedding_model()

    # Embed each chunk
    embeddings = [embedder.embed_query(chunk["text"]) for chunk in chunks]
    embeddings = np.array(embeddings).astype("float32")

    # FAISS index (L2 or cosine normalized)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    metadata = [chunk["metadata"] for chunk in chunks]

    return index, embeddings, metadata


def search_faiss(index, embeddings, metadata, query, k=5):
    embedder = get_embedding_model()
    q_vec = np.array([embedder.embed_query(query)]).astype("float32")

    distances, indices = index.search(q_vec, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({
            "score": float(dist),
            "metadata": metadata[idx]
        })

    return results
