from rag_pipeline.vector_store import search_faiss

class Retriever:
    def __init__(self, index, embeddings, metadata):
        self.index = index
        self.embeddings = embeddings
        self.metadata = metadata

    def get_relevant_docs(self, query, k=5):
        return search_faiss(
            self.index,
            self.embeddings,
            self.metadata,
            query,
            k,
        )
