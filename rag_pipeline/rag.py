"""
rag.py

This file handles:
- Prompt construction
- Retrieval -> LLM generation
- Local model loading (Phi-3 Mini recommended)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ----------------------------------------
# 1. Load Local LLM
# ----------------------------------------
def load_llm():
    """
    Loads a lightweight open-source LLM that fits in a 16GB VM.
    Modify here if you want to switch models.
    """
    model_name = "microsoft/Phi-3-mini-4k-instruct"

    print("Loading LLM... (this may take ~15–25 seconds)")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    print("LLM Loaded Successfully.")
    return tokenizer, model


# ----------------------------------------
# 2. Build Prompt
# ----------------------------------------
def build_prompt(query, docs):
    """
    Creates a grounded prompt for the LLM using retrieved context chunks.
    """

    context_blocks = []
    for d in docs:
        snippet = d.get("text", "")
        context_blocks.append(snippet)

    context = "\n\n".join(context_blocks)

    return f"""
You are a domain-specific assistant.
Answer ONLY using the context below. 
If the answer is not present in the context, say:
"I don't know based on the available documents."

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
""".strip()


# ----------------------------------------
# 3. Generate Answer Using RAG Pipeline
# ----------------------------------------
def generate_answer(query, retriever, llm=None, k=5):
    """
    Retrieves context and generates an answer using an LLM.
    """

    # Step 1: Retrieve top-k documents
    docs = retriever.get_relevant_docs(query, k=k)

    # Step 2: Build grounded prompt
    prompt = build_prompt(query, docs)

    # Step 3: Load LLM if not provided
    if llm is None:
        tokenizer, model = load_llm()
    else:
        tokenizer, model = llm

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Step 4: Generate answer
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.2,
            do_sample=False,
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "answer": answer,
        "sources": docs,
    }
