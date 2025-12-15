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
    docs = chunk_documents(docs, chunk_size=1200, chunk_overlap=200)
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
    """
    Answers a question using ONLY retrieved context.
    Returns a dict containing:
        - answer: grounded LLM answer
        - sources: list of source chunks with ids + snippet + distance
    """
    
    docs = pipeline["documents"]
    index = pipeline["index"]

    # --- Step 1: Retrieve Top-k Chunks ---
    retrieved = retrieve_top_k(index, docs, question, k=k)

    # Create readable context block sent to the LLM
    ccontext = "\n\n".join(
        f"[SOURCE: {r['id']}]\n{r['text']}"
        for r in retrieved
    )


    # --- Step 2: Build grounded prompt ---
    prompt = f"""
You are a movie data assistant.

Answer the user's question **using ONLY the context below**.
If the answer is not in the context, say:
"I cannot answer from the provided data."

Question:
{question}

Context:
{context}

Answer:
"""

    # --- Step 3: Call Local/OpenAI LLM ---
    from openai import OpenAI
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",     # Replace with Qwen/Phi/Gemma on the VM
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.get("content", "").strip()

    # --- Step 4: Build Assignment-Compliant Output Format ---
    sources_output = []
    for r in retrieved:
        snippet = r["text"][:200].replace("\n", " ")  # 200-char preview

        sources_output.append({
            "id": r["id"],
            "distance": r["distance"],
            "snippet": snippet
        })

    return {
        "answer": answer,
        "sources": sources_output
    }
