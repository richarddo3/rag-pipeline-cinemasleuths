import pandas as pd
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

def load_csv_documents(csv_path):
    df = pd.read_csv(csv_path)

    docs = []
    for i, row in df.iterrows():

        text = f"""
        Movie Title: {row['Release Group']}
        This movie was released in {row['Year']} and belongs to the genres: {row['Genres']}.
        
        It earned a worldwide gross of ${row['$Worldwide']:,}.
        It made ${row['$Domestic']:,} domestically ({row['Domestic %']}%) 
        and ${row['$Foreign']:,} in foreign markets ({row['Foreign %']}%).
        
        Audience reception:
        - Rating: {row['Rating']}
        - IMDb Average Rating: {row['imdb_avg_rating']}
        - IMDb Votes: {row['imdb_num_votes']:,}
        
        Production details:
        - Original Language: {row['Original_Language']}
        - Production Countries: {row['Production_Countries']}
        
        Summary:
        This film was one of the top-performing movies of its release year.
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
            metadata = doc.get("metadata", {})

            # ---- FIX: ensure metadata is always a dict ----
            if not isinstance(metadata, dict):
                metadata = {"source": str(metadata)}

            chunked.append({
                "id": f"{doc['id']}_chunk_{i}",
                "text": c,
                "metadata": metadata
            })
    
    return chunked
