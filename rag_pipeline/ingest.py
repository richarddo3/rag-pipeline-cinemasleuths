import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_csv_documents(path):
    df = pd.read_csv(path)

    docs = []
    for _, row in df.iterrows():
        docs.append({
            "id": str(row.get("id", "")),
            "source": "etl_cleaned_dataset.csv",
            "text": row["text"]
        })
    return docs


def chunk_documents(docs, chunk_size=800, overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )

    chunks = []
    for d in docs:
        pieces = splitter.split_text(d["text"])
        for i, c in enumerate(pieces):
            chunks.append({
                "id": f"{d['id']}_{i}",
                "text": c,
                "source": d["source"]
            })

    return chunks
