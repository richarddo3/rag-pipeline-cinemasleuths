def retrieve_top_k(index, docs, query, k=5):
    q_emb = embed_texts([query])[0]
    distances, idxs = search_faiss(index, q_emb, k=k)

    results = []
    for dist, idx in zip(distances, idxs):

        # ---- FIX: ignore invalid FAISS padded indices ----
        if idx == -1 or idx < 0 or idx >= len(docs):
            continue

        doc = docs[idx]

        # ---- FIX: ensure doc is a dict, not a stray string ----
        if not isinstance(doc, dict):
            continue

        results.append({
            "id": doc["id"],
            "text": doc["text"],
            "metadata": doc["metadata"],
            "distance": float(dist)
        })

    return results
