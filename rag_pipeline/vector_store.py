import faiss
import numpy as np
from rag_pipeline.embeddings import get_embedding_model

def build_faiss_index(chunks):
    """
    Build a FAISS index from LangChain Document objects.
    Each chunk is a Document with:
        chunk.page_content  (text)
        chunk.metadata      (metadata)
    """

    embedder = get_embedding_model()

    # Embed all chunk texts
    embeddings = []
    metadata_list = []

    for chunk in chunks:
        text = chunk.page_content
        metadata = chunk.metadata

        emb = embedder.embed_query(text)
        embeddings.append(emb)
        metadata_list.append(metadata)

    embeddings = np.array(embeddings).astype("float32")

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings, metadata_list


def search_faiss(index, embeddings, metadata, query, k=5):
    """
    Search FAISS index and return top-k metadata entries.
    """

    embedder = get_embedding_model()
    query_vec = np.array([embedder.embed_query(query)]).astype("float32")

    distances, indices = index.search(query_vec, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({
            "distance": float(dist),
            "metadata": metadata[idx]
        })

    return results
