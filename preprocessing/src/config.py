"""Paths, constants, the shared exception type and CSV helpers.

All paths are derived from the location of this file, so the project works on
any computer (macOS, Windows, Linux) without editing paths.

Loading the outputs in Tasks 2-4::

    from src.config import load_output
    documents = load_output("documents_processed", decode_tokens=True)

``load_output`` keeps empty text fields as "" (plain pandas would turn them
into NaN), reads ``original_query_id`` as a string so "001" keeps its zeros,
and with ``decode_tokens=True`` turns the JSON token columns into Python lists.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"

# Raw Cranfield input files (names exactly as distributed in cran.tar.gz).
RAW_FILENAMES = {
    "documents": "cran.all.1400",
    "queries": "cran.qry",
    "qrels": "cranqrel",
    "qrels_readme": "cranqrel.readme",
}

# Generated output files.
OUTPUT_FILENAMES = {
    "documents": "documents.csv",
    "queries": "queries.csv",
    "relevance": "relevance.csv",
    "relevance_special": "relevance_special.csv",
    "documents_processed": "documents_processed.csv",
    "queries_processed": "queries_processed.csv",
    "train": "train_queries.csv",
    "val": "val_queries.csv",
    "test": "test_queries.csv",
    "statistics": "dataset_statistics.json",
    "report": "validation_report.txt",
}

# ---------------------------------------------------------------------------
# Expected properties of the supplied Cranfield collection
# ---------------------------------------------------------------------------
EXPECTED_NUM_DOCUMENTS = 1400
EXPECTED_NUM_QUERIES = 225
EXPECTED_NUM_QREL_ROWS = 1837

# Document field markers in cran.all.1400 -> column names.
DOCUMENT_FIELD_MARKERS = {
    "T": "title",
    "A": "author",
    "B": "bibliography",
    "W": "body",
}
DOCUMENT_FIELDS = tuple(DOCUMENT_FIELD_MARKERS.values())

# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_NAMES = ("train", "val", "test")


class DataIntegrityError(Exception):
    """Raised when the input data violates a critical integrity assumption.

    The pipeline never silently repairs such problems; it stops and reports.
    """


def raw_path(key: str, raw_dir: Path = RAW_DIR) -> Path:
    """Return the path of a raw input file by logical name."""
    return Path(raw_dir) / RAW_FILENAMES[key]


def output_path(key: str, processed_dir: Path = PROCESSED_DIR) -> Path:
    """Return the path of a generated output file by logical name."""
    return Path(processed_dir) / OUTPUT_FILENAMES[key]


# ============================================================================
# CSV helpers (also used by Tasks 2-4 to load the outputs)
# ============================================================================
_DTYPES = {
    "doc_id": "int64",
    "query_id": "int64",
    "original_query_id": "string",
    "relevance_grade": "int64",
    "original_relevance_grade": "int64",
    "binary_relevance": "int64",
}


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame as UTF-8 CSV with standard quoting (overwrites)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        path,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
    )


def read_csv(path: Path, decode_tokens: bool = False) -> pd.DataFrame:
    """Read a CSV produced by this pipeline with the correct dtypes.

    Args:
        path: CSV file path.
        decode_tokens: If True, every column whose name contains ``tokens``
            is decoded from a JSON array string into a Python list.
    """
    header = pd.read_csv(path, nrows=0).columns
    dtypes = {col: _DTYPES.get(col, "string") for col in header}
    df = pd.read_csv(path, dtype=dtypes, keep_default_na=False, encoding="utf-8")
    # Plain Python str objects are friendlier for downstream code than pd.NA-aware strings.
    for col in header:
        if dtypes[col] == "string":
            df[col] = df[col].astype(object)
    if decode_tokens:
        for col in token_columns(df.columns):
            df[col] = df[col].map(json.loads)
    return df


def token_columns(columns: Iterable[str]) -> list[str]:
    """Return the names of token-list columns."""
    return [c for c in columns if "tokens" in c]


def load_output(key: str, processed_dir: Path = PROCESSED_DIR, decode_tokens: bool = False) -> pd.DataFrame:
    """Load one of the generated CSV outputs by logical name (see OUTPUT_FILENAMES)."""
    return read_csv(output_path(key, processed_dir), decode_tokens=decode_tokens)
