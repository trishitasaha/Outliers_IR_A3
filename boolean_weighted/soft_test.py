
import csv
from pathlib import Path

from grid_search import load_csv
from soft_grid_search import (
    precompute_soft_score_matrices,
    rank_from_cached_scores
)


FIELDS = ["title", "author", "bibliography", "body"]

# Frozen weights learned using training queries
BEST_WEIGHTS = {
    "title": 4,
    "author": 0,
    "bibliography": 0,
    "body": 3
}

K = 10


# --------------------------------------------------
# Load relevance judgments
# --------------------------------------------------

def load_relevance(relevance_file):

    relevance = {}

    rows = load_csv(str(relevance_file))

    for row in rows:

        if int(row["binary_relevance"]) == 1:

            query_id = int(row["query_id"])
            doc_id = int(row["doc_id"])

            relevance.setdefault(
                query_id, set()
            ).add(doc_id)

    return relevance


# --------------------------------------------------
# Calculate evaluation metrics
# --------------------------------------------------

def calculate_fixed_cutoff_metrics(
    rankings,
    relevance,
    query_ids,
    k=10
):

    precision_values = []
    recall_values = []
    f1_values = []

    evaluated_queries = 0
    short_result_queries = 0

    for query_id in query_ids:

        relevant_docs = relevance.get(
            query_id, set()
        )

        if not relevant_docs:
            continue

        evaluated_queries += 1

        top_docs = rankings.get(
            query_id, []
        )[:k]

        retrieved_ids = {
            item["doc_id"]
            for item in top_docs
        }

        true_positives = len(
            retrieved_ids & relevant_docs
        )

        if len(top_docs) < k:
            short_result_queries += 1

        # Fixed denominator for Precision@K
        precision = true_positives / k

        recall = (
            true_positives / len(relevant_docs)
        )

        f1 = (
            2 * precision * recall /
            (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

    if evaluated_queries == 0:
        raise ValueError(
            "No test queries have relevance judgments."
        )

    return {
        "queries_evaluated": evaluated_queries,
        "queries_with_fewer_than_k_results": short_result_queries,
        "precision_at_k": sum(precision_values) / evaluated_queries,
        "recall_at_k": sum(recall_values) / evaluated_queries,
        "f1_at_k": sum(f1_values) / evaluated_queries
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    # Project directory
    project_root = Path(__file__).resolve().parent.parent

    data_dir = (
        project_root /
        "preprocessing" /
        "data" /
        "processed"
    )

    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------
    # Check required files
    # ----------------------------------------------

    required_files = [
        "documents_processed.csv",
        "queries_processed.csv",
        "train_queries.csv",
        "val_queries.csv",
        "test_queries.csv",
        "relevance.csv"
    ]

    missing_files = [
        name
        for name in required_files
        if not (data_dir / name).is_file()
    ]

    if missing_files:

        raise FileNotFoundError(
            "Missing processed data files: "
            + ", ".join(missing_files)
            + f"\nExpected directory: {data_dir}"
        )

    # ----------------------------------------------
    # Load documents and queries
    # ----------------------------------------------

    documents = load_csv(
        str(data_dir / "documents_processed.csv")
    )

    queries = load_csv(
        str(data_dir / "queries_processed.csv")
    )

    train_rows = load_csv(
        str(data_dir / "train_queries.csv")
    )

    val_rows = load_csv(
        str(data_dir / "val_queries.csv")
    )

    test_rows = load_csv(
        str(data_dir / "test_queries.csv")
    )

    for document in documents:
        document["doc_id"] = int(document["doc_id"])

    for query in queries:
        query["query_id"] = int(query["query_id"])

    query_map = {
        query["query_id"]: query
        for query in queries
    }

    train_ids = {
        int(row["query_id"])
        for row in train_rows
    }

    val_ids = {
        int(row["query_id"])
        for row in val_rows
    }

    test_query_ids = [
        int(row["query_id"])
        for row in test_rows
    ]

    test_ids = set(test_query_ids)

    # ----------------------------------------------
    # Verify test split
    # ----------------------------------------------

    if len(test_ids) != len(test_query_ids):

        raise ValueError(
            "Duplicate query IDs in test_queries.csv"
        )

    overlap = test_ids & (train_ids | val_ids)

    if overlap:

        raise ValueError(
            f"Train/validation and test overlap: {sorted(overlap)}"
        )

    missing_queries = [
        query_id
        for query_id in test_query_ids
        if query_id not in query_map
    ]

    if missing_queries:

        raise KeyError(
            f"Missing test queries: {missing_queries}"
        )

    print("Documents:", len(documents))
    print("Test queries:", len(test_query_ids))

    print("\nFrozen weights:")
    print(BEST_WEIGHTS)

    weight_total = sum(BEST_WEIGHTS.values())

    if weight_total <= 0:
        raise ValueError("At least one weight must be positive.")

    normalized_weights = {
        field: BEST_WEIGHTS[field] / weight_total
        for field in FIELDS
    }

    print("\nNormalized weights:")
    print(normalized_weights)

    # ----------------------------------------------
    # Precompute soft scores for test queries
    # ----------------------------------------------

    doc_ids, cached_scores = precompute_soft_score_matrices(
        documents,
        query_map,
        test_query_ids
    )

    print("\nSoft field scores precomputed.")

    # ----------------------------------------------
    # Rank test documents
    # ----------------------------------------------

    test_rankings = {}

    for query_id in test_query_ids:

        test_rankings[query_id] = rank_from_cached_scores(
            cached_scores,
            query_id,
            BEST_WEIGHTS,
            doc_ids,
            k=None
        )

    print("Test rankings generated.")

    # ----------------------------------------------
    # Save ranked documents
    # ----------------------------------------------

    # Load relevance judgments only after rankings
    # have been generated.
    relevance = load_relevance(
        data_dir / "relevance.csv"
    )

    rankings_path = (
        output_dir / "soft_test_rankings.csv"
    )

    with rankings_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "query_id",
                "rank",
                "doc_id",
                "score",
                "binary_relevance"
            ]
        )

        writer.writeheader()

        for query_id in test_query_ids:

            relevant_docs = relevance.get(
                query_id, set()
            )

            for rank, result in enumerate(
                test_rankings[query_id],
                start=1
            ):

                writer.writerow({
                    "query_id": query_id,
                    "rank": rank,
                    "doc_id": result["doc_id"],
                    "score": f'{result["score"]:.10f}',
                    "binary_relevance": int(
                        result["doc_id"] in relevant_docs
                    )
                })

    # ----------------------------------------------
    # Evaluate test rankings
    # ----------------------------------------------

    test_metrics = calculate_fixed_cutoff_metrics(
        test_rankings,
        relevance,
        test_query_ids,
        k=K
    )

    print("\nTEST RESULTS")

    print("Precision@10:", test_metrics["precision_at_k"])
    print("Recall@10:", test_metrics["recall_at_k"])
    print("F1@10:", test_metrics["f1_at_k"])

    print(
        "Queries with fewer than 10 results:",
        test_metrics["queries_with_fewer_than_k_results"]
    )

    # ----------------------------------------------
    # Save metrics
    # ----------------------------------------------

    metrics_path = (
        output_dir / "soft_test_metrics.txt"
    )

    with metrics_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "Soft Weighted Zone Scoring - Test Results\n\n"
        )

        file.write(
            f"Weights: {BEST_WEIGHTS}\n"
        )

        file.write(
            f"Normalized weights: {normalized_weights}\n"
        )

        file.write(
            f"Cutoff: {K}\n\n"
        )

        for name, value in test_metrics.items():

            file.write(
                f"{name}: {value}\n"
            )

    print("\nFiles saved successfully:")
    print(rankings_path)
    print(metrics_path)

    print(
        "\nTest relevance judgments were used only "
        "for evaluation, not model training or ranking."
    )


if __name__ == "__main__":
    main()
