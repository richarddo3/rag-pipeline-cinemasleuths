from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Loads a HuggingFace embedding model for use in vector stores.
    Returns a LangChain-compatible embedding function.
    """
    embedder = HuggingFaceEmbeddings(model_name=model_name)
    return embedder
