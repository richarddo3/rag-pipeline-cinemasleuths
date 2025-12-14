from flask import Flask, request, jsonify
from rag_pipeline import build_rag_pipeline, answer_question

app = Flask(__name__)

print("Building RAG pipeline...")
pipeline = build_rag_pipeline()
print("RAG pipeline ready.")

@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")

    result = answer_question(pipeline, question, k=4)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
