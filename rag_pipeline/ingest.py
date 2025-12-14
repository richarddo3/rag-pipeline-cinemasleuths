import pandas as pd
from pathlib import Path

def load_csv_documents(csv_path: str):
    """
    Load ETL-cleaned CSV and convert each row into a text document
    with clear metadata so retrieval can cite the CSV correctly.
    """
    df = pd.read_csv(csv_path)

    documents = []

    for idx, row in df.iterrows():
        # Build readable text for this row
        text_parts = []
        for col, val in row.items():
            if pd.notnull(val):
                text_parts.append(f"{col}: {val}")

        text = "\n".join(text_parts)

        metadata = {
            "source": Path(csv_path).name,
            "row": int(idx)
        }

        documents.append({
            "text": text,
            "metadata": metadata
        })

    return documents


def load_text_file(path: str):
    """
    Load raw text from a .txt file.
    """
    with open(path, "r") as f:
        text = f.read()

    return [{
        "text": text,
        "metadata": {
            "source": Path(path).name
        }
    }]


def load_directory_texts(directory: str):
    """
    Load all .txt files from additional_documents directory.
    Useful for scraped pages, notes, etc.
    """
    directory = Path(directory)
    documents = []

    for file in directory.glob("*.txt"):
        with open(file, "r") as f:
            text = f.read()

        documents.append({
            "text": text,
            "metadata": {"source": file.name}
        })

    return documents
