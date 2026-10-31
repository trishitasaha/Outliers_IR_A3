
def calculate_precision_recall_f1(
    retrieved_docs,
    relevant_docs
):
    """
    Calculate precision, recall and F1
    for one query.

    retrieved_docs: ranked list of document IDs
    relevant_docs: set of relevant document IDs
    """

    retrieved_set = set(retrieved_docs)
    relevant_set = set(relevant_docs)

    true_positives = len(
        retrieved_set & relevant_set
    )

    if len(retrieved_set) == 0:
        precision = 0.0
    else:
        precision = (
            true_positives / len(retrieved_set)
        )

    if len(relevant_set) == 0:
        recall = 0.0
    else:
        recall = (
            true_positives / len(relevant_set)
        )

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (
            2 * precision * recall
            / (precision + recall)
        )

    return precision, recall, f1


def evaluate_query(
    ranked_docs,
    relevant_docs,
    k=10
):
    """
    Evaluate the top-k documents for one query.
    """

    retrieved_docs = [
        doc["doc_id"]
        for doc in ranked_docs[:k]
    ]

    return calculate_precision_recall_f1(
        retrieved_docs,
        relevant_docs
    )


def evaluate_queries(
    rankings,
    relevance,
    query_ids,
    k=10
):
    """
    Calculate macro precision, recall and F1
    across multiple queries.

    rankings:
        Dictionary mapping query ID to ranked documents.

    relevance:
        Dictionary mapping query ID to relevant document IDs.
    """

    precisions = []
    recalls = []
    f1_scores = []

    for query_id in query_ids:

        relevant_docs = relevance.get(
            query_id,
            set()
        )

        if len(relevant_docs) == 0:
            continue

        ranked_docs = rankings.get(
            query_id,
            []
        )

        precision, recall, f1 = evaluate_query(
            ranked_docs,
            relevant_docs,
            k
        )

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    if len(precisions) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }

    return {
        "precision": sum(precisions) / len(precisions),
        "recall": sum(recalls) / len(recalls),
        "f1": sum(f1_scores) / len(f1_scores)
    }
