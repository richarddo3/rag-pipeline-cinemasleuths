# embeddings.py

from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model ONCE (fast)
_model = None

def load_embedding_model():
    """
    Loads the BAAI/bge-small-en-v1.5 embedding model.
    Only loads once globally.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _model


def embed_texts(text_list):
    """
    Takes a list of strings and returns a list of embeddings (numpy vectors).
    """
    model = load_embedding_model()
    embeddings = model.encode(text_list, convert_to_numpy=True)
    return embeddings
