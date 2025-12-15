from .ingest import load_csv_documents, load_directory_texts, chunk_documents
from .embeddings import embed_texts
from .vector_store import build_faiss_index
from .retriever import retrieve_top_k


def build_rag_pipeline(
    csv_path="data/etl_cleaned_dataset.csv",
    extra_docs_dir="data/additional_documents"
):
    print("Loading CSV...")
    docs = load_csv_documents(csv_path)

    print("Loading extra documents...")
    if extra_docs_dir:
        docs += load_directory_texts(extra_docs_dir)

    print("Chunking...")
    docs = chunk_documents(docs, chunk_size=1000, chunk_overlap=200)
    docs = [d for d in docs if isinstance(d, dict)] 

    print("Embedding...")
    texts = [d["text"] for d in docs]
    embeddings = embed_texts(texts)

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    return {
        "index": index,
        "documents": docs
    }


def answer_question(pipeline, question, k=4):

    # ALWAYS use docs created by build_rag_pipeline
    docs = pipeline["documents"]
    index = pipeline["index"]

    # --- Step 1: Retrieve context ---
    retrieved = retrieve_top_k(index, docs, question, k=k)

    # Build readable context for the prompt
    context = "\n\n".join(
        f"[SOURCE: {r['id']}]\n{r['text']}"
        for r in retrieved
    )

    # --- Step 2: Build prompt ---
    prompt = f"""
You are a movie data assistant.

Answer using ONLY the context below.
If the answer is not present, say:
'I cannot answer from the provided data.'

Question:
{question}

Context:
{context}

Answer:
"""

    # --- Step 3: Call LLM ---
    from openai import OpenAI
    client = OpenAI()

    response = client.chat.completions.create(
        model="Qwen/Qwen1.5-4B-Chat",
        messages=[{"role": "user", "content": prompt}],
    )

    answer = response.choices[0].message.get("content", "").strip()

    # --- Step 4: Return assignment-compliant output ---
    sources_output = []
    for r in retrieved:
        snippet = r["text"][:200].replace("\n", " ")
        sources_output.append({
            "id": r["id"],
            "distance": r["distance"],
            "snippet": snippet
        })

    return {"answer": answer, "sources": sources_output}

