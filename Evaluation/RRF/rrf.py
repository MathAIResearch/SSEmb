def reciprocal_rank_fusion(rankings, k=60):
    """
    Compute Reciprocal Rank Fusion (RRF) rankings.
    :param rankings: A dictionary where each key is a model name, and the value is a list of ranked documents from that model
    :param k: Smoothing parameter, default is 60
    :return: Combined ranking based on RRF
    """
    doc_scores = {}
    for model, ranking in rankings.items():
        for rank, doc in enumerate(ranking, start=1):
            score = 1 / (k + rank)
            if doc not in doc_scores:
                doc_scores[doc] = score
            else:
                doc_scores[doc] += score

    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in sorted_docs],[score for doc, score in sorted_docs]

