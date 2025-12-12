import pandas as pd
import json
import os

def build_documents():
    df = pd.read_csv("data/etl_cleaned_dataset.csv")

    documents = []

    for idx, row in df.iterrows():
        text = f"""
Title: {row.get('title', '')}
Year: {row.get('year', '')}
Genres: {row.get('genres_y', '')}
Worldwide Gross: {row.get('worldwide_gross', '')}
Domestic Gross: {row.get('domestic_gross', '')}
Foreign Gross: {row.get('foreign_gross', '')}
IMDb Rating: {row.get('imdb_avg_rating', '')}
IMDb Votes: {row.get('imdb_num_votes', '')}

Summary:
This film earned {row.get('worldwide_gross', '')} worldwide and received an IMDb rating of {row.get('imdb_avg_rating', '')}.
""".strip()

        documents.append({
            "id": f"movie_{idx}",
            "text": text,
            "metadata": {
                "title": row.get("title"),
                "year": row.get("year"),
                "source": "etl_cleaned_dataset.csv"
            }
        })

    os.makedirs("data", exist_ok=True)
    with open("data/rag_documents.jsonl", "w") as f:
        for d in documents:
            f.write(json.dumps(d) + "\n")

    print("Created", len(documents), "documents.")


if __name__ == "__main__":
    build_documents()
