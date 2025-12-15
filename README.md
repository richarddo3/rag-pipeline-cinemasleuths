# CinemaSleuths: A Domain-Specific RAG Chatbot for Movie Box Office Analysis
Course: DS2002 - Data Systems, Fall 2025
Instructor: Jason Williamson

Group Members:

-Richard Do, msq2zu
-Nadia Kamal, xne8bh
-Krishna Bhamidipati, uue8zg

##Overview: CinemaSleuths is a domain-specific **Retrieval-Augmented Generation (RAG)** system designed to answer questions about movie box office performance, audience reception, and production metadata. The system retrieves relevant information from a curated dataset and generates grounded responses using a local large language model.

Unlike general-purpose chatbots, this system is intentionally constrained to answer questions **only from the provided data**, ensuring factual grounding and transparency. The project integrates ETL processing, vector search, and a Flask API into a complete, end-to-end RAG pipeline.

This project demonstrates how structured datasets (CSV-based ETL outputs) can be transformed into searchable knowledge sources for conversational AI systems.

##Data & ETL Integration: The primary dataset is a cleaned CSV generated from an earlier ETL assignment: etl_cleaned_dataset.csv

## Notebooks
The `notebooks/` directory contains exploratory analysis and early experimentation used during dataset inspection and pipeline development. These notebooks are not required to run the final system.

##Running the Project:

-Install dependencies: pip install -r api/requirements.txt
-Set PYTHONPATH: export PYTHONPATH=$(pwd)
-Start the Flask server: python3 api/app.py
-Example API Call: curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Which movie had the highest worldwide gross?"}'

  Web Interface (Chat UI):

After starting the Flask server, open a browser and navigate to:
http://<VM-IP>:8000/

Example:
http://136.107.19.147:8000/

This launches a simple chat interface where users can ask natural-language questions about movie box office performance and metadata. All responses are generated using only the project dataset and include source citations.

Expected Behavior:

- The assistant answers questions using ONLY the provided dataset.
- Each response is grounded in retrieved document chunks.
- Source IDs and text snippets are displayed with each answer.
- If a question cannot be answered from the data, the system responds:
  "I cannot answer from the provided data."

System Architecture:
1. ETL-cleaned CSV data is loaded and chunked.
2. Sentence embeddings are generated and stored in a FAISS vector index.
3. User queries retrieve the most relevant chunks via vector similarity.
4. A local large language model generates answers conditioned on retrieved context.
5. A Flask API serves both the chat UI and the /api/ask endpoint.
