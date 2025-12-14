from sentence_transformers import SentenceTransformer

_model = None

def load_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _model


def embed_texts(texts):
    model = load_embedding_model()
    return model.encode(texts, convert_to_numpy=True)
