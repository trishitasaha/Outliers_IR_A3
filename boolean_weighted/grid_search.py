import csv
import json
import itertools
import heapq
import os

from scoring import calculate_score
from evaluation import evaluate_queries


FIELDS = ["title", "author", "bibliography", "body"]


def load_csv(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def load_data(data_dir):
    documents = load_csv(os.path.join(data_dir, "documents_processed.csv"))
    queries = load_csv(os.path.join(data_dir, "queries_processed.csv"))
    relevance_rows = load_csv(os.path.join(data_dir, "relevance.csv"))
    train_rows = load_csv(os.path.join(data_dir, "train_queries.csv"))
    val_rows = load_csv(os.path.join(data_dir, "val_queries.csv"))
    test_rows = load_csv(os.path.join(data_dir, "test_queries.csv"))

    for document in documents:
        document["doc_id"] = int(document["doc_id"])

    for query in queries:
        query["query_id"] = int(query["query_id"])

    relevance = {}
    for row in relevance_rows:
        if int(row["binary_relevance"]) == 1:
            query_id = int(row["query_id"])
            doc_id = int(row["doc_id"])
            relevance.setdefault(query_id, set()).add(doc_id)

    train_query_ids = [int(row["query_id"]) for row in train_rows]
    val_query_ids = [int(row["query_id"]) for row in val_rows]
    test_query_ids = [int(row["query_id"]) for row in test_rows]

    query_map = {query["query_id"]: query for query in queries}

    return (
        documents,
        query_map,
        relevance,
        train_query_ids,
        val_query_ids,
        test_query_ids,
    )


def token_set(tokens):
    """Convert CSV JSON tokens or token lists to a lowercase set."""
    if isinstance(tokens, str):
        tokens = json.loads(tokens)
    return {str(token).lower() for token in tokens}


def precompute_match_count_groups(documents, query_map, query_ids):
    """Precompute the number of query-term matches in each field once."""
    prepared_documents = []

    for document in documents:
        field_sets = [
            token_set(document[field + "_stemmed_tokens"])
            for field in FIELDS
        ]
        prepared_documents.append((document["doc_id"], field_sets))

    cached_counts = {}

    for query_id in query_ids:
        query_tokens = token_set(
            query_map[query_id]["query_stemmed_tokens"]
        )
        count_groups = {}

        for doc_id, field_sets in prepared_documents:
            match_counts = tuple(
                len(query_tokens & field_tokens)
                for field_tokens in field_sets
            )
            count_groups.setdefault(match_counts, []).append(doc_id)

        for doc_ids in count_groups.values():
            doc_ids.sort()

        cached_counts[query_id] = count_groups

    return cached_counts


def build_boolean_pattern_cache(cached_counts, min_matches):
    """Convert cached field match counts to Boolean 0/1 field patterns."""
    cached_patterns = {}

    for query_id, count_groups in cached_counts.items():
        boolean_groups = {}

        for match_counts, doc_ids in count_groups.items():
            pattern = tuple(
                int(count >= min_matches)
                for count in match_counts
            )
            boolean_groups.setdefault(pattern, []).extend(doc_ids)

        for doc_ids in boolean_groups.values():
            doc_ids.sort()

        cached_patterns[query_id] = boolean_groups

    return cached_patterns


def rank_from_cached_patterns(cached_patterns, query_id, weights, k=None):
    """Rank documents; equal scores are ordered by ascending document ID."""
    score_buckets = {}
    groups = cached_patterns[query_id]

    for pattern, doc_id_groups in groups.items():
        boolean_scores = {
            field: pattern[index]
            for index, field in enumerate(FIELDS)
        }
        score = calculate_score(boolean_scores, weights)

        if score > 0:
            score_buckets.setdefault(score, []).append(doc_id_groups)

    ranked = []

    for score in sorted(score_buckets, reverse=True):
        # Merge already sorted ID lists, so ties are resolved by document ID.
        for doc_id in heapq.merge(*score_buckets[score]):
            ranked.append({"doc_id": doc_id, "score": score})
            if k is not None and len(ranked) >= k:
                return ranked

    return ranked


def grid_search(cached_patterns, relevance, train_query_ids, weight_values, k=10):
    best_weights = None
    best_metrics = None
    best_f1 = -1.0

    total = len(weight_values) ** len(FIELDS) - 1
    checked = 0

    for combination in itertools.product(weight_values, repeat=len(FIELDS)):
        if not any(combination):
            continue

        checked += 1
        weights = {
            field: combination[index]
            for index, field in enumerate(FIELDS)
        }

        rankings = {
            query_id: rank_from_cached_patterns(
                cached_patterns,
                query_id,
                weights,
                k=k,
            )
            for query_id in train_query_ids
        }

        metrics = evaluate_queries(
            rankings,
            relevance,
            train_query_ids,
            k=k,
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_weights = weights.copy()
            best_metrics = metrics.copy()

        if checked % 100 == 0:
            print(f"Checked {checked}/{total} weight combinations")

    return best_weights, best_metrics


def evaluate_split(cached_patterns, weights, relevance, query_ids, k=10):
    rankings = {
        query_id: rank_from_cached_patterns(
            cached_patterns,
            query_id,
            weights,
            k=k,
        )
        for query_id in query_ids
    }
    metrics = evaluate_queries(rankings, relevance, query_ids, k=k)
    return rankings, metrics


if __name__ == "__main__":
    DATA_DIR = "../preprocessing/data/processed"
    K = 10
    WEIGHT_VALUES = [0, 1, 2, 3, 4, 5]
    THRESHOLDS = [1, 2, 3]

    (
        documents,
        query_map,
        relevance,
        train_query_ids,
        val_query_ids,
        test_query_ids,
    ) = load_data(DATA_DIR)

    print("Documents:", len(documents))
    print("Training queries:", len(train_query_ids))
    print("Validation queries:", len(val_query_ids))
    print("Test queries:", len(test_query_ids))

    all_query_ids = list(dict.fromkeys(
        train_query_ids + val_query_ids + test_query_ids
    ))

    best_configuration = None
    best_validation_f1 = -1.0
    print("Precomputing query-term match counts once...")
    cached_counts = precompute_match_count_groups(
        documents,
        query_map,
        all_query_ids,
    )
    print("Match-count precomputation complete.")

    best_cache = None

    for threshold in THRESHOLDS:
        print("\n" + "=" * 55)
        print(f"BOOLEAN MATCH THRESHOLD: {threshold}")
        print("=" * 55)
        cached_patterns = build_boolean_pattern_cache(
            cached_counts,
            min_matches=threshold,
        )
        print("Learning field weights on training queries...")

        best_weights, train_metrics = grid_search(
            cached_patterns,
            relevance,
            train_query_ids,
            WEIGHT_VALUES,
            k=K,
        )

        _, val_metrics = evaluate_split(
            cached_patterns,
            best_weights,
            relevance,
            val_query_ids,
            k=K,
        )

        print("Best training weights:", best_weights)
        print("Training metrics:", train_metrics)
        print("Validation metrics:", val_metrics)

        if val_metrics["f1"] > best_validation_f1:
            best_validation_f1 = val_metrics["f1"]
            best_configuration = {
                "min_matches": threshold,
                "weights": best_weights.copy(),
                "train_metrics": train_metrics.copy(),
                "validation_metrics": val_metrics.copy(),
            }
            best_cache = cached_patterns

    print("\n" + "=" * 55)
    print("SELECTED CONFIGURATION (based on validation F1)")
    print("=" * 55)
    print("Minimum matches per field:", best_configuration["min_matches"])
    print("Field weights:", best_configuration["weights"])
    print("Training metrics:", best_configuration["train_metrics"])
    print("Validation metrics:", best_configuration["validation_metrics"])

    # The test set is evaluated only after threshold and weights are selected.
    test_rankings, test_metrics = evaluate_split(
        best_cache,
        best_configuration["weights"],
        relevance,
        test_query_ids,
        k=K,
    )

    print("\nFinal test metrics:")
    print(test_metrics)

    if test_query_ids:
        sample_query_id = test_query_ids[0]
        sample_query = json.loads(
            query_map[sample_query_id]["query_stemmed_tokens"]
        )
        relevant_docs = relevance.get(sample_query_id, set())

        print("\nSample ranked results")
        print("Query ID:", sample_query_id)
        print("Query tokens:", sample_query)
        print("Top", K, "documents:")

        for result in test_rankings[sample_query_id]:
            status = "RELEVANT" if result["doc_id"] in relevant_docs else "not judged relevant"
            print(
                f"Document {result['doc_id']} | "
                f"Score {result['score']} | {status}"
            )
