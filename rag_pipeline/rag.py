%%writefile rag_pipeline/rag.py
def build_prompt(query, docs):
    """
    Build a context-grounded prompt for the LLM.
    """
    context = "\n\n".join([d.get("text", "") for d in docs])

    return f"""
Answer ONLY using the context below. 
If the answer is not available, say so.

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
        # Debug mode
        return {
            "answer": "(LLM disabled — pipeline test successful)",
            "sources": docs,
            "prompt": prompt
        }

    response_text = llm(prompt)
    return {"answer": response_text, "sources": docs}
