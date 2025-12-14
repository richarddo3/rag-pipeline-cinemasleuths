import os
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_csv_documents(path):
    df = pd.read_csv(path)
    documents = []
    for _, row in df.iterrows():
        documents.append({
            "text": row["text"],
            "metadata": {"source": path}
        })
    return documents


def load_directory_texts(directory):
    """Load all .txt files in a directory as documents."""
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            full_path = os.path.join(directory, filename)
            with open(full_path, "r", encoding="utf-8") as f:
                documents.append({
                    "text": f.read(),
                    "metadata": {"source": filename}
                })
    return documents
