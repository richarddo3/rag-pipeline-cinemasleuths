# embeddings.py

from sentence_transformers import SentenceTransformer
import numpy as np

# Cache model so it only loads once
_model = None

def load_embedding_model(model_name="BAAI/bge-small-en-v1.5"):
    """
    Loads a SentenceTransformer embedding model.
    Reuses the same model instance for speed.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model


def embed_texts(text_list):
    """
    Takes a list of strings and returns their embeddings as numpy arrays.
    This is what the vector_store and retriever rely on.
    """
    model = load_embedding_model()
    embeddings = model.encode(text_list, convert_to_numpy=True)
    return embeddings
