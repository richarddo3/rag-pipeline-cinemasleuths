import pandas as pd
from pathlib import Path

def load_etl_documents(csv_path="data/etl_cleaned_dataset.csv"):
    """
    Loads your cleaned ETL dataset and converts each row into a RAG-ready document.

    OUTPUT FORMAT:
    [
        {
            "text": "full text string here…",
            "metadata": {
                "title": "...",
                "year": 2009,
                "genres": "Action, Sci-Fi",
                ...
            }
        },
        ...
    ]
    """

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"ETL CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    # OPTIONAL: adjust these if your ETL columns are named differently
    text_columns = [
        "Title",
        "Worldwide Gross",
        "IMDb Rating",
        "IMDb Vote Count",
        "Genres",
        "Original Language",
        "Production Countries"
    ]

    documents = []

    for _, row in df.iterrows():

        # Build the "text" field (this is what will be embedded)
        text_parts = []
        for col in text_columns:
            if col in df.columns:
                text_parts.append(f"{col}: {row[col]}")
        
        full_text = " | ".join(text_parts)

        # Build metadata
        metadata = {
            "title": row.get("Title", ""),
            "year": int(row["Year"]) if "Year" in row and not pd.isna(row["Year"]) else None,
            "genres": row.get("Genres", ""),
            "language": row.get("Original Language", "")
        }

        documents.append({
            "text": full_text,
            "metadata": metadata
        })

    return documents
