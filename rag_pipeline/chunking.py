from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_documents(documents, chunk_size=800, chunk_overlap=150):
    """
    Takes a list of documents in the form:
    { "text": "...", "metadata": {...} }

    Returns a list of chunked documents:
    {
        "text": chunk_text,
        "metadata": original_metadata
    }
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunked_output = []

    for doc in documents:
        chunks = splitter.split_text(doc["text"])
        for chunk in chunks:
            chunked_output.append({
                "text": chunk,
                "metadata": doc["metadata"]
            })

    return chunked_output
