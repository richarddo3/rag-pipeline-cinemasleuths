from .ingest import load_csv_documents, load_directory_texts, chunk_documents
from .embeddings import embed_texts
from .vector_store import build_faiss_index
from .retriever import retrieve_top_k

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

DATA_PATH = "data/etl_cleaned_dataset.csv"


SYSTEM_PROMPT = """
You are a movie data assistant.

Answer using ONLY the context below.
Compare values carefully and choose the correct answer.
Cite the source ID(s) you used.

If the answer cannot be found in the context, say:
"I cannot answer from the provided data."
""".strip()


# -----------------------------
# Local LLM (Qwen 1.5 – 4B)
# -----------------------------
_tokenizer = None
_model = None

def get_llm():
    global _tokenizer, _model

    if _model is None:
        model_name = "Qwen/Qwen1.5-4B-Chat"
        _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

    return _tokenizer, _model


# -----------------------------
# Build RAG Pipeline
# -----------------------------
def build_rag_pipeline(
    csv_path="data/etl_cleaned_dataset.csv",
    extra_docs_dir="data/additional_documents"
):
    print("📄 Loading CSV documents...")
    docs = load_csv_documents(csv_path)

    if extra_docs_dir:
        print("📂 Loading additional documents...")
        docs += load_directory_texts(extra_docs_dir)

    print("✂️ Chunking documents...")
    docs = chunk_documents(docs, chunk_size=1000, chunk_overlap=200)
    docs = [d for d in docs if isinstance(d, dict)]

    print("🧠 Generating embeddings...")
    texts = [d["text"] for d in docs]
    embeddings = embed_texts(texts)

    # Normalize for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    print("📦 Building FAISS index...")
    index = build_faiss_index(embeddings)

    print("✅ RAG pipeline ready.")
    return {
        "index": index,
        "documents": docs
    }


# -----------------------------
# Answer Question
# -----------------------------
def answer_question(pipeline, question, k=4):
    docs = pipeline["documents"]
    index = pipeline["index"]

    # --- Retrieve ---
    retrieved = retrieve_top_k(index, docs, question, k=k)

    if not retrieved:
        return {
            "answer": "I cannot answer from the provided data.",
            "sources": []
        }

    # Build context
    context = "\n\n".join(
        f"[SOURCE: {r['id']}]\n{r['text']}"
        for r in retrieved
    )

    # --- Prompt ---
    prompt = f"""
    {SYSTEM_PROMPT}
    
    Question:
    {question}
    
    Context:
    {context}
    
    Answer (be concise and factual):
    """.strip()

    # --- Generate ---
    tokenizer, model = get_llm()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "Answer:" in decoded:
        answer = decoded.split("Answer:", 1)[1].strip()
    else:
        answer = decoded.strip()



    # --- Format sources ---
    sources = []
    for r in retrieved:
        snippet = r["text"][:200].replace("\n", " ")
        sources.append({
            "id": r["id"],
            "source": "etl_cleaned_dataset.csv",
            "snippet": snippet
        })

    return {
        "answer": answer.strip(),
        "sources": sources
    }
