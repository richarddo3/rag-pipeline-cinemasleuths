import pandas as pd
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_csv_documents(csv_path, text_column=None):
    df = pd.read_csv(csv_path)
    if text_column is None:
        text_column = df.columns[0]

    docs = []
    for i, row in df.iterrows():
        docs.append({
            "id": f"row-{i}",
            "text": str(row[text_column]),
            "metadata": row.to_dict()
        })
    return docs


def load_directory_texts(dir_path, extensions=(".txt", ".md", ".csv")):
    docs = []
    for root, dirs, files in os.walk(dir_path):
        for fname in files:
            if fname.lower().endswith(extensions):
                full = os.path.join(root, fname)
                try:
                    with open(full, "r", encoding="utf-8") as f:
                        text = f.read()
                    docs.append({
                        "id": fname,
                        "text": text,
                        "metadata": {"source": fname}
                    })
                except:
                    pass
    return docs


def chunk_documents(documents, chunk_size=800, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunked_docs = []
    for doc in documents:
        chunks = splitter.split_text(doc["text"])
        for i, c in enumerate(chunks):
            chunked_docs.append({
                "id": f"{doc['id']}_chunk_{i}",
                "text": c,
                "metadata": doc["metadata"]
            })
    return chunked_docs
