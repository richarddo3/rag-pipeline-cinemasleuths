import faiss
import numpy as np

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))
    return index


def search_faiss(index, query_embedding, k=5):
    distances, indices = index.search(
        np.array([query_embedding]).astype("float32"), k
    )
    return distances[0], indices[0]
