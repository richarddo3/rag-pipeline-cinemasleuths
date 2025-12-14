from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def build_prompt(query, docs):
    context_blocks = []
    for d in docs:
        snippet = d.get("text", "")
        context_blocks.append(snippet)

    context = "\n\n".join(context_blocks)

    return f"""
You are a domain-specific assistant.
You must answer ONLY using the context provided.
If the answer is not in the context, say:
"I don't know based on the available documents."

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
""".strip()


# Load LLM only once (fast)
_llm_pipeline = None

def load_llm():
    global _llm_pipeline
    if _llm_pipeline is None:
        model_name = "Qwen/Qwen1.5-4B-Chat"  # runs on 16GB VM
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto"
        )
        _llm_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=300
        )
    return _llm_pipeline


def generate_answer(query, retriever, k=5):
    docs = retriever.get_relevant_docs(query, k=k)

    prompt = build_prompt(query, docs)

    llm = load_llm()
    raw_output = llm(prompt)[0]["generated_text"]

    answer = raw_output.replace(prompt, "").strip()

    return {
        "answer": answer,
        "sources": docs
    }
