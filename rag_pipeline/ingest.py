import pandas as pd
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

def load_csv_documents(csv_path):
    df = pd.read_csv(csv_path)

    docs = []
    for i, row in df.iterrows():

        text = f"""
        Title: {row['Release Group']}
        Rank: {row['Rank']}
        Year: {row['Year']}

        Genres: {row['Genres']}

        Worldwide Gross: {row['$Worldwide']}
        Domestic Gross: {row['$Domestic']} ({row['Domestic %']}%)
        Foreign Gross: {row['$Foreign']} ({row['Foreign %']}%)

        Rating: {row['Rating']}
        IMDb Avg Rating: {row['imdb_avg_rating']}
        IMDb Votes: {row['imdb_num_votes']}
        Vote Count: {row['Vote_Count']}

        Original Language: {row['Original_Language']}
        Production Countries: {row['Production_Countries']}
        """

        docs.append({
            "id": f"row-{i}",
            "text": text,
            "metadata": row.to_dict()
        })

    return docs



def load_directory_texts(dir_path, extensions=(".txt", ".md", ".csv")):
    docs = []
    
    for root, dirs, files in os.walk(dir_path):
        for fname in files:
            if fname.lower().endswith(extensions):
                full_path = os.path.join(root, fname)
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        text = f.read()
                    docs.append({
                        "id": fname,
                        "text": text,
                        "metadata": {"source": fname}
                    })
                except:
                    pass
    return docs


def chunk_documents(documents, chunk_size=800, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunked = []
    
    for doc in documents:
        chunks = splitter.split_text(doc["text"])
        for i, c in enumerate(chunks):
            chunked.append({
                "id": f"{doc['id']}_chunk_{i}",
                "text": c,
                "metadata": doc["metadata"]
            })
    
    return chunked
