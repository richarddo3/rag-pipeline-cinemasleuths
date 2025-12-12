import faiss
import numpy as np
from rag_pipeline.embeddings import get_embedding_model

def build_faiss_index(chunks):
    """
    Build a FAISS index from LangChain Document chunks.

    chunks: list of Document objects with:
        - chunk.page_content
        - chunk.metadata

    Returns: index, embeddings_matrix, metadata_list
    """
    embedder = get_embedding_model()

    # Embed each chunk's text
    embeddings = [
        embedder.embed_query(chunk.page_content)
        for chunk in chunks
    ]
    embeddings = np.array(embeddings).astype("float32")

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    metadata = [chunk.metadata for chunk in chunks]

    return index, embeddings, metadata


def search_faiss(index, embeddings, metadata, query, k=5):
    """
    Query FAISS and return top-k metadata results.
    """
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
