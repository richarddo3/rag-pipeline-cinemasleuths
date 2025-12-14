import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------
# Load Local LLM
# ---------------------------
def load_llm():
    """
    Loads a small local LLM that fits on a 16GB VM.
    Modify model_name if you'd rather use Qwen, Gemma, etc.
    """
    model_name = "microsoft/Phi-3-mini-4k-instruct"

    print("Loading LLM:", model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    print("LLM loaded successfully.")
    return tokenizer, model


# ---------------------------
# Generate Answer
# ---------------------------
def generate_answer(question, retriever, llm=None):
    """
    Retrieves context and (optionally) generates an answer using an LLM.
    If llm is None, returns retrieved chunks only.
    """
    docs = retriever.get_relevant_docs(question)

    # Convert docs into text
    context_blocks = []
    for d in docs:
        context_blocks.append(f"[source={d['metadata']}] {d['text']}")

    context_text = "\n\n".join(context_blocks)

    # If no LLM provided → return retrieval results only
    if llm is None:
        return {
            "answer": "(LLM not connected yet)",
            "sources": docs
        }

    tokenizer, model = llm

    prompt = f"""
Answer the following question using ONLY the provided context.
If the answer is not in the context, say "I cannot answer from the given data."

### QUESTION:
{question}

### CONTEXT:
{context_text}

### ANSWER:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.2,
    )

    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    return {
        "answer": answer,
        "sources": docs
    }
