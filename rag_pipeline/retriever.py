def retrieve_top_k(index, docs, query, k=5):
    q_emb = embed_texts([query])[0]
    distances, idxs = search_faiss(index, q_emb, k=k)

    results = []
    for dist, idx in zip(distances, idxs):

        # Skip invalid FAISS returns (e.g., -1 or strings)
        if isinstance(idx, str):
            try:
                idx = int(idx)
            except:
                continue

        if idx < 0 or idx >= len(docs):
            continue  # FAISS pads with -1 when not enough results

        doc = docs[idx]
        results.append({
            "id": doc["id"],
            "text": doc["text"],
            "metadata": doc["metadata"],
            "distance": float(dist)
        })

    return results
