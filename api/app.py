from flask import Flask, request, jsonify
from rag_pipeline.ingest import load_csv_documents, chunk_documents
from rag_pipeline.embeddings import embed_texts
from rag_pipeline.vector_store import build_faiss_index
from rag_pipeline.retriever import retrieve
from rag_pipeline.rag import generate_answer

import numpy as np

app = Flask(__name__)

print("Loading data...")
docs_raw = load_csv_documents("data/etl_cleaned_dataset.csv")
chunks = chunk_documents(docs_raw)
texts = [c["text"] for c in chunks]
embeddings = embed_texts(texts)
index = build_faiss_index(embeddings)
print("Ready.")


@app.post("/api/ask")
def ask():
    data = request.get_json()
    question = data.get("question")

    retrieved = retrieve(question, chunks, index, k=5)
    answer = generate_answer(question, retrieved)

    return jsonify({
        "answer": answer,
        "sources": retrieved
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
