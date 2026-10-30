"""Reproducible query-level train / validation / test split.

Splitting is done over QUERIES, never over query-document pairs, so no query
appears in more than one split. The document collection is not partitioned:
every split is searched against the full 1,400-document corpus, and the global
relevance table is filtered by the split's query IDs when needed.

To keep a similar spread of queries with many and few relevant documents in
every split, queries are stratified into quantile bins of their number of
graded judgments (rows of relevance.csv, grades 1-5) before random assignment.
That count does not depend on the binary relevance threshold, so changing the
threshold never changes the split.
"""

from __future__ import annotations

import logging
import math
import random

import pandas as pd

from src.config import RANDOM_SEED, SPLIT_NAMES, SPLIT_RATIOS

logger = logging.getLogger(__name__)

SPLIT_COLUMNS = ["query_id", "original_query_id", "query_text"]


def compute_split_sizes(n: int, ratios: dict[str, float] = SPLIT_RATIOS) -> dict[str, int]:
    """Number of queries per split; validation and test are rounded, train takes the rest.

    For n = 225 and 70/15/15 this gives train=157, val=34, test=34.
    """
    if not math.isclose(sum(ratios.values()), 1.0):
        raise ValueError(f"Split ratios must sum to 1, got {ratios}")
    val = int(round(n * ratios["val"]))
    test = int(round(n * ratios["test"]))
    return {"train": n - val - test, "val": val, "test": test}


def judged_counts(queries: pd.DataFrame, relevance: pd.DataFrame) -> pd.Series:
    """Number of graded (non-special) judgments per query (0 if none); policy-independent."""
    counts = relevance.groupby("query_id").size()
    return queries["query_id"].map(counts).fillna(0).astype(int)


def _allocate(stratum_sizes: list[int], available: list[int], total: int, share: float) -> list[int]:
    """Largest-remainder allocation of ``total`` items across strata.

    Each stratum ideally receives ``size * share`` items; floors are taken first
    and the remainder goes to strata with the largest fractional parts (ties
    broken by stratum order). No stratum receives more than ``available``.
    """
    ideal = [s * share for s in stratum_sizes]
    alloc = [min(int(math.floor(x)), a) for x, a in zip(ideal, available)]
    order = sorted(range(len(ideal)), key=lambda i: (-(ideal[i] - math.floor(ideal[i])), i))
    remaining = total - sum(alloc)
    while remaining > 0:
        progressed = False
        for i in order:
            if remaining and alloc[i] < available[i]:
                alloc[i] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise ValueError("Cannot allocate split sizes across strata")
    return alloc


def split_queries(
    queries: pd.DataFrame,
    relevance: pd.DataFrame,
    seed: int = RANDOM_SEED,
    ratios: dict[str, float] = SPLIT_RATIOS,
    n_strata: int = 4,
) -> dict[str, pd.DataFrame]:
    """Assign every query to exactly one of train / val / test.

    Args:
        queries: Table with ``query_id, original_query_id, query_text``.
        relevance: Ordinary relevance table (relevance.csv).
        seed: Random seed; the same seed always gives the same split.
        ratios: Split proportions.
        n_strata: Number of quantile bins of per-query judgment counts.

    Returns:
        Mapping split name -> DataFrame (columns ``SPLIT_COLUMNS``), sorted by query_id.
    """
    sizes = compute_split_sizes(len(queries), ratios)
    n = len(queries)

    table = queries[SPLIT_COLUMNS].sort_values("query_id").reset_index(drop=True)
    counts = judged_counts(table, relevance)
    # Rank first so that ties do not produce empty or unequal quantile bins.
    n_bins = max(1, min(n_strata, n))
    table["stratum"] = pd.qcut(counts.rank(method="first"), q=n_bins, labels=False) if n > 1 else 0

    rng = random.Random(seed)
    strata: list[list[int]] = []
    for _, group in table.groupby("stratum", sort=True):
        ids = sorted(group["query_id"].tolist())
        rng.shuffle(ids)
        strata.append(ids)

    stratum_sizes = [len(s) for s in strata]
    val_alloc = _allocate(stratum_sizes, stratum_sizes, sizes["val"], sizes["val"] / n)
    after_val = [s - v for s, v in zip(stratum_sizes, val_alloc)]
    test_alloc = _allocate(stratum_sizes, after_val, sizes["test"], sizes["test"] / n)

    assignment: dict[int, str] = {}
    for ids, n_val, n_test in zip(strata, val_alloc, test_alloc):
        for qid in ids[:n_val]:
            assignment[qid] = "val"
        for qid in ids[n_val:n_val + n_test]:
            assignment[qid] = "test"
        for qid in ids[n_val + n_test:]:
            assignment[qid] = "train"

    table["split"] = table["query_id"].map(assignment)
    splits = {
        name: table.loc[table["split"] == name, SPLIT_COLUMNS].sort_values("query_id").reset_index(drop=True)
        for name in SPLIT_NAMES
    }
    for name in SPLIT_NAMES:
        logger.debug("Split %-5s: %d queries", name, len(splits[name]))
    return splits


def summarize_splits(splits: dict[str, pd.DataFrame], relevance: pd.DataFrame) -> dict[str, dict]:
    """Query counts and usable judgment counts per split (special -1 rows excluded)."""
    summary = {}
    for name, df in splits.items():
        rel = relevance[relevance["query_id"].isin(df["query_id"])]
        pos = rel[rel["binary_relevance"] == 1]
        summary[name] = {
            "num_queries": int(len(df)),
            "num_judgments": int(len(rel)),
            "num_positive_judgments": int(len(pos)),
            "mean_positive_per_query": round(len(pos) / len(df), 3) if len(df) else 0.0,
            "queries_without_positive": int(len(set(df["query_id"]) - set(pos["query_id"]))),
            "grade_counts": {str(k): int(v) for k, v in rel["relevance_grade"].value_counts().sort_index().items()},
        }
    return summary
