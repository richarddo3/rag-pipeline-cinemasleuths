from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(documents, chunk_size=800, chunk_overlap=150):
    """
    Takes a list of documents in the form:
    {"text": "...", "metadata": {...}}

    Returns a list of chunked documents.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    texts = [doc["text"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]

    chunks = splitter.create_documents(texts, metadatas=metadatas)
    return chunks
