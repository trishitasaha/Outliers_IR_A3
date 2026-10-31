
from boolean_matching import get_boolean_scores


FIELDS = [
    "title",
    "author",
    "bibliography",
    "body"
]


def calculate_score(boolean_scores, weights):
    """
    Calculate the weighted Boolean score
    for a query-document pair.

    boolean_scores: dictionary of four Boolean scores
    weights: dictionary of four field weights
    """

    score = 0

    for field in FIELDS:
        score += weights[field] * boolean_scores[field]

    return score


def score_document(query_tokens, document, weights):
    """
    Calculate the final score for one document.
    """

    boolean_scores = get_boolean_scores(
        query_tokens,
        document
    )

    final_score = calculate_score(
        boolean_scores,
        weights
    )

    return final_score



def rank_documents(query_tokens, documents, weights):

    results = []

    for document in documents:

        score = score_document(
            query_tokens,
            document,
            weights
        )

        # Retrieve only documents with a positive score
        if score > 0:

            results.append({
                "doc_id": document["doc_id"],
                "score": score
            })

    results.sort(
        key=lambda x: (-x["score"], x["doc_id"])
    )

    return results
