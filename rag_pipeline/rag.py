from .retriever import retrieve
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load local LLM (pick one from project list)
MODEL_NAME = "Qwen/Qwen1.5-4B-Chat"

_tokenizer = None
_model = None

def load_llm():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto"
        )
    return _tokenizer, _model


def generate_answer(question, retrieved_chunks):
    tokenizer, model = load_llm()

    context = "\n\n".join([c["text"] for c in retrieved_chunks])

    prompt = f"""
You are a grounded assistant. Use ONLY the context below.
If the answer is not in the context, say "I do not know".

Context:
{context}

Question: {question}

Answer:
"""

    tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**tokens, max_new_tokens=250)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    return answer
