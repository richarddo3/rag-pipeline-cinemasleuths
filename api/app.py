from flask import Flask, request, jsonify
from rag_pipeline.rag import build_rag_pipeline, answer_question
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

print("🔧 Building RAG pipeline... (this may take 1–2 minutes)")
pipeline = build_rag_pipeline()
print("✅ Pipeline ready.")

@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({"error": "Request must contain a 'question' field"}), 400

    question = data["question"]

    result = answer_question(pipeline, question, k=4)

    return jsonify(result)

@app.route("/")
def serve_ui():
    return send_from_directory("ui", "index.html")

if __name__ == "__main__":
    # Runs on VM
    app.run(host="0.0.0.0", port=8000)

