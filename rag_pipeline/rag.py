%%writefile rag-pipeline-cinemasleuths/rag_pipeline/rag.py
def build_prompt(query, docs):
    context = "\n\n".join([d.get("text", "") for d in docs])
    return f"""
Answer ONLY using the context below.
If the answer is not found, say that you cannot find it.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
""".strip()


def generate_answer(query, retriever, llm=None, k=5):
    docs = retriever.get_relevant_docs(query, k=k)
    prompt = build_prompt(query, docs)

    if llm is None:
        # Basic testing mode
        return {
            "answer": "(LLM disabled — pipeline test successful)",
            "sources": docs,
            "prompt": prompt
        }

    # Real LLM execution
    result = llm(prompt)
    return {"answer": result, "sources": docs}
