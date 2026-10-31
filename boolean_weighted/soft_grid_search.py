import itertools
import json
import math

import numpy as np

from evaluation import evaluate_queries
from grid_search import load_data
from soft_matching import get_soft_scores
from soft_scoring import calculate_soft_score

FIELDS = ["title", "author", "bibliography", "body"]


def precompute_soft_score_matrices(documents, query_map, query_ids):
    """Cache one (number_of_documents x 4) score matrix per query."""
    doc_ids = np.asarray([doc["doc_id"] for doc in documents], dtype=np.int64)
    cached = {}

    for query_id in query_ids:
        query_tokens = json.loads(query_map[query_id]["query_stemmed_tokens"])
        matrix = np.zeros((len(documents), len(FIELDS)), dtype=np.float64)

        for doc_index, document in enumerate(documents):
            field_scores = get_soft_scores(query_tokens, document)
            matrix[doc_index] = [field_scores[field] for field in FIELDS]

        cached[query_id] = matrix

    return doc_ids, cached


def select_top_k_indices(scores, doc_ids, k=10):
    """Select positive-score documents, with deterministic doc-ID tie-breaking."""
    positive = np.flatnonzero(scores > 0.0)

    if positive.size == 0:
        return np.asarray([], dtype=np.int64)

    if positive.size <= k:
        selected = positive
    else:
        threshold = np.partition(scores[positive], positive.size - k)[positive.size - k]
        better = positive[scores[positive] > threshold]
        tied = positive[scores[positive] == threshold]

        # Match the original ranking policy: lower document ID wins a score tie.
        tied = tied[np.argsort(doc_ids[tied], kind="stable")]
        needed = k - better.size
        selected = np.concatenate((better, tied[:needed]))

    order = np.lexsort((doc_ids[selected], -scores[selected]))
    return selected[order]


def rank_from_cached_scores(cached_scores, query_id, weights, doc_ids, k=None):
    """Rank a query; k=None returns every positive-score document."""
    weight_vector = np.asarray([weights[field] for field in FIELDS], dtype=np.float64)
    scores = cached_scores[query_id] @ weight_vector

    if k is None:
        indices = np.flatnonzero(scores > 0.0)
        order = np.lexsort((doc_ids[indices], -scores[indices]))
        indices = indices[order]
    else:
        indices = select_top_k_indices(scores, doc_ids, k)

    return [
        {"doc_id": int(doc_ids[index]), "score": float(scores[index])}
        for index in indices
    ]


def evaluate_weight_combinations(
    weight_matrix,
    doc_ids,
    cached_scores,
    relevance,
    query_ids,
    k=10,
):
    """Vectorized grid search evaluation using the project's existing metric definition."""
    number_of_combinations = len(weight_matrix)
    precision_sums = np.zeros(number_of_combinations, dtype=np.float64)
    recall_sums = np.zeros(number_of_combinations, dtype=np.float64)
    f1_sums = np.zeros(number_of_combinations, dtype=np.float64)
    evaluated_queries = 0

    for query_id in query_ids:
        relevant_docs = relevance.get(query_id, set())
        if not relevant_docs:
            continue

        evaluated_queries += 1
        scores_for_combinations = weight_matrix @ cached_scores[query_id].T
        relevant_set = relevant_docs

        for combo_index in range(number_of_combinations):
            top_indices = select_top_k_indices(
                scores_for_combinations[combo_index], doc_ids, k
            )
            retrieved_count = len(top_indices)
            true_positives = sum(
                int(doc_ids[index]) in relevant_set
                for index in top_indices
            )

            precision = true_positives / retrieved_count if retrieved_count else 0.0
            recall = true_positives / len(relevant_set)
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if precision + recall > 0.0
                else 0.0
            )

            precision_sums[combo_index] += precision
            recall_sums[combo_index] += recall
            f1_sums[combo_index] += f1

    if evaluated_queries == 0:
        raise ValueError("No queries with relevance judgments were available for evaluation.")

    metrics = {
        "precision": precision_sums / evaluated_queries,
        "recall": recall_sums / evaluated_queries,
        "f1": f1_sums / evaluated_queries,
    }
    return metrics


def grid_search(doc_ids, cached_scores, relevance, train_query_ids, weight_values, k=10):
    combinations = [
        combination
        for combination in itertools.product(weight_values, repeat=len(FIELDS))
        if any(combination)
    ]
    weight_matrix = np.asarray(combinations, dtype=np.float64)

    print(f"Searching {len(combinations)} non-zero weight combinations.")
    metrics = evaluate_weight_combinations(
        weight_matrix,
        doc_ids,
        cached_scores,
        relevance,
        train_query_ids,
        k,
    )

    best_index = int(np.argmax(metrics["f1"]))
    best_weights = {
        field: int(weight_matrix[best_index, position])
        for position, field in enumerate(FIELDS)
    }
    best_metrics = {
        metric: float(values[best_index])
        for metric, values in metrics.items()
    }
    return best_weights, best_metrics


def normalize_weights(weights):
    total = sum(weights.values())
    if total == 0:
        return {field: 0.0 for field in FIELDS}
    return {field: weights[field] / total for field in FIELDS}


def count_top_k_ties(rankings, query_ids, k=10):
    tie_pairs = 0
    possible_pairs = 0

    for query_id in query_ids:
        top_docs = rankings.get(query_id, [])[:k]
        for index in range(1, len(top_docs)):
            possible_pairs += 1
            if math.isclose(
                top_docs[index - 1]["score"],
                top_docs[index]["score"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                tie_pairs += 1

    return tie_pairs, possible_pairs


if __name__ == "__main__":
    DATA_DIR = "../preprocessing/data/processed"

    (
        documents,
        query_map,
        relevance,
        train_query_ids,
        val_query_ids,
    ) = load_data(DATA_DIR)

    print("Documents:", len(documents))
    print("Training queries:", len(train_query_ids))
    print("Validation queries:", len(val_query_ids))

    all_query_ids = list(dict.fromkeys(train_query_ids + val_query_ids))
    doc_ids, cached_scores = precompute_soft_score_matrices(
        documents, query_map, all_query_ids
    )
    print("Soft field scores precomputed.")

    weight_values = [0, 1, 2, 3, 4, 5]
    best_weights, train_metrics = grid_search(
        doc_ids,
        cached_scores,
        relevance,
        train_query_ids,
        weight_values,
        k=10,
    )

    print("\nBest raw weights:")
    print(best_weights)
    print("Normalized weights:")
    print(normalize_weights(best_weights))
    print("\nTraining metrics:")
    print(train_metrics)

    val_rankings = {
        query_id: rank_from_cached_scores(
            cached_scores, query_id, best_weights, doc_ids, k=10
        )
        for query_id in val_query_ids
    }
    val_metrics = evaluate_queries(
        val_rankings, relevance, val_query_ids, k=10
    )

    print("\nValidation metrics:")
    print(val_metrics)

    tie_pairs, possible_pairs = count_top_k_ties(
        val_rankings, val_query_ids, k=10
    )
    tie_rate = tie_pairs / possible_pairs if possible_pairs else 0.0
    print(
        "\nValidation adjacent tie pairs in top 10:",
        f"{tie_pairs}/{possible_pairs} ({tie_rate:.3f})",
    )