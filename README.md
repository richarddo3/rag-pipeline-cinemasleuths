# CinemaSleuths: A Domain-Specific RAG Chatbot for Movie Box Office Analysis
Course: DS2002 - Data Systems, Fall 2025
Instructor: Jason Williamson

Group Memebers:
-Richard Do, msq2zu
-Nadia Kamal, xne8bh
-Krishna Bhamidipati, uue8zg

Overview: CinemaSleuths is a domain-specific **Retrieval-Augmented Generation (RAG)** system designed to answer questions about movie box office performance, audience reception, and production metadata. The system retrieves relevant information from a curated dataset and generates grounded responses using a local large language model.

Unlike general-purpose chatbots, this system is intentionally constrained to answer questions **only from the provided data**, ensuring factual grounding and transparency. The project integrates ETL processing, vector search, and a Flask API into a complete, end-to-end RAG pipeline.

This project demonstrates how structured datasets (CSV-based ETL outputs) can be transformed into searchable knowledge sources for conversational AI systems.

Data & ETL Integration: The primary dataset is a cleaned CSV generated from an earlier ETL assignment: etl_cleaned_dataset.csv
