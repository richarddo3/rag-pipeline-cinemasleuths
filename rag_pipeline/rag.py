def build_prompt(query, docs):
    context_blocks = []
    for d in docs:
        snippet = d.get("text", "")
        context_blocks.append(snippet)

    context = "\n\n".join(context_blocks)

    return f"""
You are a domain-specific assistant.
Answer using ONLY the available context.
If the answer is not in the context, say "I don't know based on the available documents."

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
        return {"answer": "(LLM not connected yet)", "sources": docs}

    result = llm(prompt)
    return {"answer": result, "sources": docs}
