from soft_matching import get_soft_scores

FIELDS = ["title", "author", "bibliography", "body"]


def calculate_soft_score(soft_scores, weights):
    """Calculate the weighted sum of the four fractional field scores."""
    return sum(weights[field] * soft_scores[field] for field in FIELDS)


def score_document(query_tokens, document, weights):
    soft_scores = get_soft_scores(query_tokens, document)
    return calculate_soft_score(soft_scores, weights)


def rank_documents(query_tokens, documents, weights):
    results = []

    for document in documents:
        score = score_document(query_tokens, document, weights)
        if score > 0:
            results.append({"doc_id": document["doc_id"], "score": score})

    results.sort(key=lambda item: (-item["score"], item["doc_id"]))
    return results
