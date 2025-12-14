from rag_pipeline.retriever import Retriever
from openai import OpenAI

client = OpenAI()

SYSTEM_PROMPT = """
You are a domain-specific assistant. 
Always answer ONLY using the context provided.
If the answer is not in the retrieved context, say:
'I could not find that information in the dataset.'

Always cite the sources.
"""

def generate_answer(query, retriever, k=5):
    # 1. Retrieve context
    docs = retriever.get_relevant_docs(query, k=k)

    # Build context string
    context_text = ""
    sources = []

    for d in docs:
        meta = d["metadata"]
        snippet = meta.get("chunk", meta)  # fallback
        context_text += f"- {snippet}\n"
        sources.append(meta)

    # 2. Build final prompt
    prompt = f"""
SYSTEM:
{SYSTEM_PROMPT}

CONTEXT:
{context_text}

USER QUESTION:
{query}
"""

    # 3. Call local model
    response = client.responses.create(
        model="gpt-4o-mini",  # Replace with Qwen, Gemma, Phi later on VM
        input=prompt
    )

    # Extract text
    answer = response.output_text

    return {
        "answer": answer,
        "sources": sources
    }
