"""Parsers for the three Cranfield files: documents, queries and relevance judgments.

File format (cran.all.1400 and cran.qry)
----------------------------------------
Each record starts with ``.I <id>``. Field markers (``.T`` title, ``.A`` author,
``.B`` bibliography, ``.W`` body/query text) stand on a line of their own and
the field text follows on the next lines. A line counts as a marker only if it
starts with ``.`` + one upper-case letter + whitespace/end of line. Cranfield
text is all lower-case, so text such as ``j. ae. scs.`` or ``.5 per cent`` is
never mistaken for a marker.

Repeated markers: documents 240, 576 and 578 contain a second marker inside
their body (e.g. a second ``.W``). The text after it is a continuation of the
body, so a repeated marker does NOT switch fields; the text stays in the open
field and the event is reported as a parse warning.

Query IDs
---------
``cran.qry`` uses sparse, zero-padded IDs (001, 002, 004, 008, ..., 365) while
``cranqrel`` numbers queries 1..225. The k-th query record in ``cran.qry`` is
qrel query k. ``query_id`` (1..225) is the join key for all tasks;
``original_query_id`` keeps the file's label. This correspondence is not
stated in the files; it is inferred from record order and checked in
``validate_dataset.py``.

Relevance codes (cranqrel.readme)
---------------------------------
1 complete answer, 2 high relevance, 3 useful, 4 minimum interest,
5 no interest (absent from the file). Smaller = MORE relevant. Code ``-1`` is
not documented; those rows are kept separately and never given a binary label.
Documents absent from the qrels for a query are unjudged, not non-relevant.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import (
    DOCUMENT_FIELD_MARKERS,
    DOCUMENT_FIELDS,
    EXPECTED_NUM_DOCUMENTS,
    EXPECTED_NUM_QREL_ROWS,
    EXPECTED_NUM_QUERIES,
    DataIntegrityError,
)

logger = logging.getLogger(__name__)

# ``.X`` at line start, optionally followed by whitespace and inline content.
MARKER_RE = re.compile(r"^\.([A-Z])(?:\s+(.*))?$")


# ============================================================================
# Shared helpers
# ============================================================================
@dataclass
class ParseWarning:
    """A non-fatal irregularity found while parsing a raw file."""

    file: str
    record_id: str
    line_number: int
    message: str

    def __str__(self) -> str:
        return f"{self.file} line {self.line_number} (record {self.record_id}): {self.message}"


@dataclass
class _Record:
    """Accumulates the lines of one ``.I`` record while parsing."""

    record_id: str
    line_number: int
    current_field: str | None = None
    seen_fields: list[str] = field(default_factory=list)
    lines: dict[str, list[str]] = field(default_factory=dict)


def read_raw_lines(path: Path) -> list[str]:
    """Read a raw Cranfield file as a list of lines (UTF-8, any line endings)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required input file not found: {path}")
    return path.read_text(encoding="utf-8", errors="strict").splitlines()


def join_field_lines(lines: list[str]) -> str:
    """Join the wrapped lines of a field with single spaces (text itself is unchanged)."""
    return " ".join(stripped for line in lines if (stripped := line.strip()))


def parse_marked_records(
    lines: list[str],
    field_markers: dict[str, str],
    file_label: str,
) -> tuple[list[_Record], list[ParseWarning]]:
    """Split raw lines into ``.I`` records with named fields (used for documents and queries).

    Args:
        lines: Raw file lines.
        field_markers: Marker letter -> field name (``I`` excluded).
        file_label: File name used in warnings and errors.

    Raises:
        DataIntegrityError: For text that cannot be attributed to a record or
            field, or an ``.I`` line without an identifier.
    """
    records: list[_Record] = []
    warnings: list[ParseWarning] = []
    current: _Record | None = None

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip()
        match = MARKER_RE.match(line)
        letter = match.group(1) if match else None
        inline = (match.group(2) or "").strip() if match else ""

        if letter == "I":
            if not inline:
                raise DataIntegrityError(f"{file_label} line {line_number}: '.I' marker without an identifier")
            current = _Record(record_id=inline, line_number=line_number)
            records.append(current)
            continue

        if current is None:
            if line.strip():
                raise DataIntegrityError(f"{file_label} line {line_number}: text before the first '.I' record: {line[:60]!r}")
            continue

        if letter in field_markers:
            field_name = field_markers[letter]
            if field_name in current.seen_fields:
                warnings.append(ParseWarning(
                    file_label, current.record_id, line_number,
                    f"repeated '.{letter}' marker; following text kept in the currently open field '{current.current_field}'",
                ))
            else:
                current.seen_fields.append(field_name)
                current.current_field = field_name
                current.lines.setdefault(field_name, [])
            if inline:
                current.lines.setdefault(current.current_field, []).append(inline)
            continue

        if letter is not None:
            # Looks like a marker but is not a known one: keep it as text and flag it.
            warnings.append(ParseWarning(file_label, current.record_id, line_number, f"unknown marker-like line kept as text: {line!r}"))

        if current.current_field is None:
            if line.strip():
                raise DataIntegrityError(f"{file_label} line {line_number}: text in record {current.record_id} before any field marker")
            continue
        current.lines[current.current_field].append(line)

    return records, warnings


def _check_unique_and_count(df: pd.DataFrame, id_column: str, expected_count: int | None, label: str) -> None:
    """Raise if ``id_column`` has duplicates or the row count is not ``expected_count``."""
    duplicated = df.loc[df[id_column].duplicated(keep=False), id_column].unique().tolist()
    if duplicated:
        raise DataIntegrityError(f"{label}: duplicate {id_column} values: {duplicated}")
    if expected_count is not None and len(df) != expected_count:
        raise DataIntegrityError(f"{label}: expected {expected_count} records, parsed {len(df)}")


# ============================================================================
# Documents (cran.all.1400)
# ============================================================================
def parse_documents(
    path: Path,
    expected_count: int | None = EXPECTED_NUM_DOCUMENTS,
) -> tuple[pd.DataFrame, list[ParseWarning]]:
    """Parse ``cran.all.1400``.

    Args:
        path: Path to the raw document file.
        expected_count: Required number of documents, or None to skip the check
            (used by unit tests with synthetic data).

    Returns:
        ``(documents, warnings)``; ``documents`` has columns
        ``doc_id, title, author, bibliography, body`` in file order. A missing
        field is stored as "" and reported, never dropped.
    """
    label = Path(path).name
    records, warnings = parse_marked_records(read_raw_lines(path), DOCUMENT_FIELD_MARKERS, label)

    rows = []
    for record in records:
        if not record.record_id.isdigit():
            raise DataIntegrityError(f"{label} line {record.line_number}: non-integer identifier {record.record_id!r}")
        missing = [f for f in DOCUMENT_FIELDS if f not in record.seen_fields]
        if missing:
            warnings.append(ParseWarning(label, record.record_id, record.line_number, f"missing field marker(s): {missing}"))
        row = {"doc_id": int(record.record_id)}
        row.update({f: join_field_lines(record.lines.get(f, [])) for f in DOCUMENT_FIELDS})
        rows.append(row)

    documents = pd.DataFrame(rows, columns=["doc_id", *DOCUMENT_FIELDS]).astype({"doc_id": "int64"})
    _check_unique_and_count(documents, "doc_id", expected_count, label)
    logger.debug("Parsed %d documents from %s (%d parse warnings)", len(documents), label, len(warnings))
    return documents, warnings


# ============================================================================
# Queries (cran.qry) and the canonical query-ID mapping
# ============================================================================
QUERY_FIELD_MARKERS = {"W": "query_text"}


def parse_queries(
    path: Path,
    expected_count: int | None = EXPECTED_NUM_QUERIES,
) -> tuple[pd.DataFrame, list[ParseWarning]]:
    """Parse ``cran.qry`` and assign canonical sequential query IDs (1-based record position).

    Returns:
        ``(queries, warnings)``; ``queries`` has columns
        ``query_id, original_query_id, query_text``. ``original_query_id`` is
        the identifier string exactly as written in the file (e.g. ``"004"``).
    """
    label = Path(path).name
    records, warnings = parse_marked_records(read_raw_lines(path), QUERY_FIELD_MARKERS, label)

    rows = []
    for position, record in enumerate(records, start=1):
        if not record.record_id.isdigit():
            raise DataIntegrityError(f"{label} line {record.line_number}: non-numeric query ID {record.record_id!r}")
        text = join_field_lines(record.lines.get("query_text", []))
        if not text:
            warnings.append(ParseWarning(label, record.record_id, record.line_number, "missing or empty query text"))
        rows.append({"query_id": position, "original_query_id": record.record_id, "query_text": text})

    queries = pd.DataFrame(rows, columns=["query_id", "original_query_id", "query_text"]).astype(
        {"query_id": "int64", "original_query_id": object}
    )
    _check_unique_and_count(queries, "original_query_id", expected_count, label)
    if not queries["original_query_id"].astype(int).is_monotonic_increasing:
        # The positional mapping is still applied, but its basis is weaker: report it.
        warnings.append(ParseWarning(label, "-", 0, "original query IDs are not in increasing order in the file"))

    logger.debug("Parsed %d queries from %s (%d parse warnings)", len(queries), label, len(warnings))
    return queries, warnings


def verify_query_id_mapping(queries: pd.DataFrame, qrel_query_ids: Iterable[int]) -> dict:
    """Check that the qrels only use the canonical IDs 1..N (N = number of queries).

    Returns:
        A description of the mapping (stored in dataset_statistics.json).

    Raises:
        DataIntegrityError: If the qrels reference an ID outside 1..N.
    """
    n = len(queries)
    expected = set(range(1, n + 1))
    qrel_ids = {int(q) for q in qrel_query_ids}
    if set(queries["query_id"]) != expected or not qrel_ids <= expected:
        raise DataIntegrityError(
            f"Query ID mapping check failed: qrel IDs outside 1..{n}: {sorted(qrel_ids - expected)[:10]}"
        )
    return {
        "num_queries_in_cran_qry": n,
        "qrel_query_id_range": [min(qrel_ids), max(qrel_ids)] if qrel_ids else None,
        "num_distinct_qrel_query_ids": len(qrel_ids),
        "qrel_ids_contiguous_1_to_n": qrel_ids == expected,
        "original_ids_increasing": bool(queries["original_query_id"].astype(int).is_monotonic_increasing),
        "mapping_rule": "query_id = 1-based position of the record in cran.qry",
    }


# ============================================================================
# Relevance judgments (cranqrel)
# ============================================================================
GRADE_DESCRIPTIONS = {
    1: "complete answer to the question",
    2: "high degree of relevance",
    3: "useful as background or for methods",
    4: "minimum interest",
    5: "no interest",
}
SPECIAL_GRADES = {
    -1: "undocumented special relevance code (-1): not defined in cranqrel.readme; meaning unknown",
}


@dataclass(frozen=True)
class RelevancePolicy:
    """Maps documented grades (1-5) to binary labels.

    ``relevant_grades`` get ``binary_relevance = 1``; every other documented
    grade gets 0. The default {1, 2, 3, 4} counts everything except "no
    interest"; ``RelevancePolicy.from_max_grade(3)`` gives the stricter {1, 2, 3}.
    """

    relevant_grades: frozenset[int] = field(default_factory=lambda: frozenset({1, 2, 3, 4}))

    def __post_init__(self) -> None:
        unknown = set(self.relevant_grades) - set(GRADE_DESCRIPTIONS)
        if unknown:
            raise ValueError(f"relevant_grades contains undocumented grades: {sorted(unknown)}")
        if not self.relevant_grades:
            raise ValueError("relevant_grades must not be empty")

    @classmethod
    def from_max_grade(cls, max_grade: int) -> "RelevancePolicy":
        """Policy where grades 1..max_grade are relevant."""
        return cls(frozenset(range(1, max_grade + 1)))

    def binary_label(self, grade: int) -> int:
        """1 if ``grade`` is relevant under this policy, else 0."""
        if grade not in GRADE_DESCRIPTIONS:
            raise ValueError(f"Grade {grade} is not a documented relevance grade")
        return int(grade in self.relevant_grades)

    def describe(self) -> dict:
        return {
            "relevant_grades": sorted(self.relevant_grades),
            "non_relevant_grades": sorted(set(GRADE_DESCRIPTIONS) - set(self.relevant_grades)),
            "excluded_special_grades": sorted(SPECIAL_GRADES),
        }


def parse_qrels(
    path: Path,
    expected_rows: int | None = EXPECTED_NUM_QREL_ROWS,
) -> tuple[pd.DataFrame, list[ParseWarning]]:
    """Read every row of ``cranqrel`` without interpreting the grades.

    Returns:
        ``(qrels, warnings)``; ``qrels`` has columns
        ``line_number, query_id, doc_id, relevance_grade``.

    Raises:
        DataIntegrityError: Lists every malformed row (not three integer fields,
            or a code outside {-1, 1..5}), duplicate query-document pairs, or a
            wrong row count. Nothing is silently skipped.
    """
    label = Path(path).name
    rows, malformed, warnings = [], [], []
    allowed = set(GRADE_DESCRIPTIONS) | set(SPECIAL_GRADES)

    for line_number, line in enumerate(read_raw_lines(path), start=1):
        if not line.strip():
            continue
        parts = line.split()
        try:
            query_id, doc_id, grade = (int(p) for p in parts)
        except ValueError:
            malformed.append(f"line {line_number}: expected 3 integer fields: {line!r}")
            continue
        if grade not in allowed:
            malformed.append(f"line {line_number}: unknown relevance code {grade}: {line!r}")
            continue
        if line.strip() != " ".join(parts):
            warnings.append(ParseWarning(label, str(query_id), line_number, f"irregular internal whitespace: {line!r}"))
        rows.append({"line_number": line_number, "query_id": query_id, "doc_id": doc_id, "relevance_grade": grade})

    if malformed:
        raise DataIntegrityError(f"{label}: {len(malformed)} malformed row(s):\n  " + "\n  ".join(malformed))

    qrels = pd.DataFrame(rows, columns=["line_number", "query_id", "doc_id", "relevance_grade"]).astype("int64")
    duplicates = qrels[qrels.duplicated(subset=["query_id", "doc_id"], keep=False)]
    if not duplicates.empty:
        raise DataIntegrityError(f"{label}: duplicate query-document judgments: {duplicates.to_dict('records')}")
    if expected_rows is not None and len(qrels) != expected_rows:
        raise DataIntegrityError(f"{label}: expected {expected_rows} judgment rows, parsed {len(qrels)}")

    logger.debug("Parsed %d relevance judgment rows from %s", len(qrels), label)
    return qrels, warnings


def check_references(qrels: pd.DataFrame, doc_ids: Iterable[int], query_ids: Iterable[int]) -> None:
    """Raise if any judgment references an unknown document or canonical query ID."""
    unknown_docs = sorted(set(qrels["doc_id"]) - {int(d) for d in doc_ids})
    unknown_queries = sorted(set(qrels["query_id"]) - {int(q) for q in query_ids})
    if unknown_docs or unknown_queries:
        raise DataIntegrityError(
            f"Relevance judgments reference missing records: unknown doc_ids {unknown_docs[:20]}, "
            f"unknown query_ids {unknown_queries[:20]}"
        )


def split_relevance(
    qrels: pd.DataFrame,
    policy: RelevancePolicy = RelevancePolicy(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate ordinary graded judgments from special-code rows.

    Returns:
        ``relevance``: ``query_id, doc_id, relevance_grade, binary_relevance``
        for documented grades (1-5), in file order; and ``relevance_special``:
        ``query_id, doc_id, original_relevance_grade, reason`` for code -1.
    """
    is_special = qrels["relevance_grade"].isin(list(SPECIAL_GRADES))

    relevance = qrels.loc[~is_special, ["query_id", "doc_id", "relevance_grade"]].copy()
    relevance["binary_relevance"] = relevance["relevance_grade"].map(policy.binary_label).astype("int64")

    special = qrels.loc[is_special, ["query_id", "doc_id", "relevance_grade"]].rename(
        columns={"relevance_grade": "original_relevance_grade"}
    )
    special["reason"] = special["original_relevance_grade"].map(SPECIAL_GRADES)

    return relevance.reset_index(drop=True), special.reset_index(drop=True)
