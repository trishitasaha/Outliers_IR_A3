"""Task 1 pipeline: parse, preprocess, split and validate the Cranfield collection.

Usage (from the project folder):

    python main.py                          # default: grades 1-4 relevant, seed 42
    python main.py --relevant-max-grade 3   # optional: only grades 1-3 relevant

All outputs in data/processed are regenerated on every run, so it is safe to
rerun. Exit codes: 0 success, 1 data-integrity failure, 2 missing input files.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.config import (
    PROCESSED_DIR,
    RANDOM_SEED,
    RAW_DIR,
    RAW_FILENAMES,
    SPLIT_NAMES,
    DataIntegrityError,
    output_path,
    raw_path,
    write_csv,
)
from src.parsing import (
    RelevancePolicy,
    check_references,
    parse_documents,
    parse_qrels,
    parse_queries,
    split_relevance,
    verify_query_id_mapping,
)
from src.preprocess import PreprocessConfig, preprocess_documents, preprocess_queries
from src.split_dataset import split_queries, summarize_splits
from src.validate_dataset import compute_statistics, validate_outputs

logger = logging.getLogger("task1")


def check_input_files(raw_dir: Path) -> None:
    """Fail early, listing every missing raw file."""
    missing = [name for name in RAW_FILENAMES.values() if not (raw_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing input file(s) in {raw_dir}: {missing}")


def run_pipeline(
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    policy: RelevancePolicy = RelevancePolicy(),
    preprocess_config: PreprocessConfig = PreprocessConfig(),
    seed: int = RANDOM_SEED,
) -> dict:
    """Run every Task 1 step and return a summary dictionary.

    Raises:
        FileNotFoundError: If a raw input file is missing.
        DataIntegrityError: If parsing or validation finds a critical problem
            (the validation report is still written in the latter case).
    """
    raw_dir, processed_dir = Path(raw_dir), Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    # A report from a previous run must never be mistaken for this run's result.
    output_path("report", processed_dir).unlink(missing_ok=True)
    check_input_files(raw_dir)

    logger.info("[1/5] Parsing documents, queries and relevance judgments")
    documents, doc_warnings = parse_documents(raw_path("documents", raw_dir))
    queries, query_warnings = parse_queries(raw_path("queries", raw_dir))
    qrels, qrel_warnings = parse_qrels(raw_path("qrels", raw_dir))
    mapping_info = verify_query_id_mapping(queries, qrels["query_id"])
    check_references(qrels, documents["doc_id"], queries["query_id"])
    relevance, relevance_special = split_relevance(qrels, policy)
    parse_warnings = doc_warnings + query_warnings + qrel_warnings

    logger.info("[2/5] Preprocessing document fields and queries")
    documents_processed = preprocess_documents(documents, preprocess_config)
    queries_processed = preprocess_queries(queries, preprocess_config)

    logger.info("[3/5] Creating the query-level train/val/test split (seed=%d)", seed)
    splits = split_queries(queries, relevance, seed=seed)
    split_summary = summarize_splits(splits, relevance)

    outputs = {
        "documents": documents,
        "queries": queries,
        "relevance": relevance,
        "relevance_special": relevance_special,
        "documents_processed": documents_processed,
        "queries_processed": queries_processed,
        **splits,
    }
    for key, df in outputs.items():
        write_csv(df, output_path(key, processed_dir))

    logger.info("[4/5] Validating the generated files")
    report, mapping_evidence = validate_outputs(processed_dir, raw_dir, policy, preprocess_config, seed, parse_warnings)
    output_path("report", processed_dir).write_text(report.render(), encoding="utf-8")

    logger.info("[5/5] Writing dataset statistics")
    stats = compute_statistics(
        documents, queries, qrels, relevance, relevance_special,
        documents_processed, queries_processed, split_summary, parse_warnings,
        extra={
            "query_id_mapping": mapping_info,
            "relevance_policy": policy.describe(),
            "preprocessing": preprocess_config.describe(),
            "random_seed": seed,
            **mapping_evidence,
        },
    )
    output_path("statistics", processed_dir).write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    if report.critical_failures:
        failed = "\n  ".join(c.name for c in report.critical_failures)
        raise DataIntegrityError(f"{len(report.critical_failures)} critical validation check(s) failed:\n  {failed}")

    return {
        "documents": len(documents),
        "queries": len(queries),
        "qrel_rows": len(qrels),
        "relevance_rows": len(relevance),
        "special_rows": len(relevance_special),
        "splits": split_summary,
        "checks_passed": sum(c.passed for c in report.checks),
        "checks_total": len(report.checks),
        "num_warnings": len(report.warnings),
        "processed_dir": processed_dir,
    }


def print_summary(summary: dict) -> None:
    splits = summary["splits"]
    print()
    print("=" * 60)
    print("TASK 1 PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Documents parsed     : {summary['documents']}")
    print(f"Queries parsed       : {summary['queries']}")
    print(f"Relevance judgments  : {summary['qrel_rows']} "
          f"({summary['relevance_rows']} graded + {summary['special_rows']} special -1)")
    print(f"Query split          : train {splits['train']['num_queries']}, "
          f"val {splits['val']['num_queries']}, test {splits['test']['num_queries']}")
    print(f"Validation           : PASS ({summary['checks_passed']}/{summary['checks_total']} checks; "
          f"{summary['num_warnings']} raw-data warnings, see validation_report.txt)")
    print(f"Output files         : {summary['processed_dir']}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cranfield Task 1: data preparation pipeline")
    parser.add_argument("--relevant-max-grade", type=int, default=4, choices=[1, 2, 3, 4],
                        help="grades 1..N get binary_relevance=1 (default 4)")
    parser.add_argument("--split-hyphens", action="store_true",
                        help="split hyphenated words into parts (default: keep them whole)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help=f"random seed (default {RANDOM_SEED})")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        summary = run_pipeline(
            policy=RelevancePolicy.from_max_grade(args.relevant_max_grade),
            preprocess_config=PreprocessConfig(split_hyphens=args.split_hyphens),
            seed=args.seed,
        )
    except FileNotFoundError as exc:
        logger.error("ERROR: %s\nPut the four Cranfield files into %s and run again.", exc, RAW_DIR)
        return 2
    except DataIntegrityError as exc:
        logger.error("DATA INTEGRITY FAILURE: %s", exc)
        if output_path("report").exists():
            logger.error("See %s for details.", output_path("report"))
        return 1
    print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
