from rag_pipeline.vector_store import search_faiss


class Retriever:
    """
    Thin wrapper around FAISS search that stores:
        - index
        - embeddings matrix
        - metadata list
    """

    def __init__(self, index, embeddings, metadata):
        self.index = index
        self.embeddings = embeddings
        self.metadata = metadata

    def retrieve(self, query, k=5):
        """
        Run similarity search and return top-k chunks + metadata
        """
        results = search_faiss(
            self.index,
            self.embeddings,
            self.metadata,
            query,
            k=k
        )
        return results
