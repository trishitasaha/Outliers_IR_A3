"""Validation of the generated outputs, and the dataset statistics.

All checks read the files back from ``data/processed`` (not the in-memory
objects that produced them), so CSV round-trip problems are caught too.

A CRITICAL check failing means the data must not be used: ``main.py`` writes
the report and exits with an error. A WARNING marks a known irregularity in the
raw data that is kept unchanged and reported, never silently repaired.

The statistics are descriptive only. They are not ranking features (no IDF,
BM25 or query-document statistics); those belong to Task 3.
"""

from __future__ import annotations

import json
import random
import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import DOCUMENT_FIELDS, SPLIT_NAMES, DataIntegrityError, load_output, raw_path
from src.parsing import SPECIAL_GRADES, ParseWarning, RelevancePolicy, parse_documents, parse_qrels, parse_queries
from src.preprocess import QUERY_PROCESSED_COLUMNS, PreprocessConfig, document_processed_columns, preprocess_text
from src.split_dataset import SPLIT_COLUMNS, compute_split_sizes, split_queries


# ============================================================================
# Report
# ============================================================================
@dataclass
class CheckResult:
    section: str
    name: str
    passed: bool
    critical: bool
    detail: str = ""


@dataclass
class ValidationReport:
    """Collects check results and notes, and renders them as text."""

    checks: list[CheckResult] = field(default_factory=list)
    notes: dict[str, list[str]] = field(default_factory=dict)

    def check(self, section: str, name: str, passed: bool, detail: str = "", critical: bool = True) -> None:
        self.checks.append(CheckResult(section, name, bool(passed), critical, detail))

    def warn_if(self, section: str, name: str, problem_found: bool, detail: str) -> None:
        """Record a non-critical check; the detail is only shown when it fails."""
        self.check(section, name, not problem_found, detail if problem_found else "", critical=False)

    def note(self, section: str, message: str) -> None:
        self.notes.setdefault(section, []).append(message)

    @property
    def critical_failures(self) -> list[CheckResult]:
        return [c for c in self.checks if c.critical and not c.passed]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.critical and not c.passed]

    def render(self) -> str:
        lines = [
            "CRANFIELD TASK 1 - DATA VALIDATION REPORT",
            "=" * 72,
            f"Generated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
            f"Checks run: {len(self.checks)}   passed: {sum(c.passed for c in self.checks)}   "
            f"critical failures: {len(self.critical_failures)}   warnings: {len(self.warnings)}",
            f"OVERALL RESULT: {'FAIL' if self.critical_failures else 'PASS'}",
            "",
            "Legend: [PASS] check satisfied | [FAIL] critical integrity violation |",
            "        [WARN] irregularity in the raw data, kept unchanged",
        ]
        for section in dict.fromkeys([c.section for c in self.checks] + list(self.notes)):
            lines += ["", section.upper(), "-" * 72]
            for c in self.checks:
                if c.section == section:
                    status = "PASS" if c.passed else ("FAIL" if c.critical else "WARN")
                    lines.append(f"[{status}] {c.name}")
                    lines += [f"       {d}" for d in c.detail.splitlines()]
            for message in self.notes.get(section, []):
                first, *rest = message.splitlines()
                lines += [f"  - {first}"] + [f"    {m}" for m in rest]
        return "\n".join(lines) + "\n"


# ============================================================================
# Helpers
# ============================================================================
def _warnings_for(parse_warnings: list[ParseWarning], raw_key: str, raw_dir: Path) -> str:
    """Parse warnings that belong to one raw file, one per line."""
    name = raw_path(raw_key, raw_dir).name
    return "\n".join(str(w) for w in parse_warnings if w.file == name)


def _decode_token_columns(df: pd.DataFrame) -> tuple[dict[str, list[list[str]]], list[str]]:
    """Decode every ``*tokens*`` column; return the lists and any decoding failures."""
    decoded, failures = {}, []
    for col in [c for c in df.columns if "tokens" in c]:
        values = []
        for i, raw in enumerate(df[col]):
            try:
                tokens = json.loads(raw)
                if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
                    raise ValueError("not a list of strings")
            except (ValueError, TypeError) as exc:
                failures.append(f"{col} row {i}: {exc}")
                tokens = []
            values.append(tokens)
        decoded[col] = values
    return decoded, failures


def _wrap(items: list[str], width: int = 68) -> str:
    """Join items with ', ' and wrap the result to ``width`` characters."""
    lines, line = [], ""
    for item in items:
        if line and len(line) + len(item) + 2 > width:
            lines.append(line + ",")
            line = item
        else:
            line = f"{line}, {item}" if line else item
    return "\n".join(lines + [line] if line else lines)


def _empty_document_ids(docs: pd.DataFrame) -> list[int]:
    return docs.loc[(docs[list(DOCUMENT_FIELDS)] == "").all(axis=1), "doc_id"].astype(int).tolist()


def query_mapping_evidence(queries_proc: pd.DataFrame, documents_proc: pd.DataFrame, relevance: pd.DataFrame) -> dict:
    """Content check of the positional query-ID mapping.

    For a candidate mapping (qrel query id -> query text), each query is scored
    by the fraction of its distinct stemmed terms that occur in the title+body
    of the documents judged relevant for it. The correct mapping should score
    far above a random assignment. The relevance labels are used only to check
    ID alignment; nothing here feeds a model or the split.
    """
    query_terms = {int(r.query_id): set(json.loads(r.query_stemmed_tokens)) for r in queries_proc.itertuples()}
    doc_terms = {
        int(r.doc_id): set(json.loads(r.title_stemmed_tokens)) | set(json.loads(r.body_stemmed_tokens))
        for r in documents_proc.itertuples()
    }
    relevant_docs = relevance[relevance["binary_relevance"] == 1].groupby("query_id")["doc_id"].apply(list).to_dict()

    def mean_coverage(mapping: dict[int, int]) -> dict:
        scores = []
        for qrel_qid, text_qid in mapping.items():
            terms, docs = query_terms.get(text_qid), relevant_docs.get(qrel_qid)
            if terms and docs:
                vocab = set().union(*(doc_terms[d] for d in docs))
                scores.append(len(terms & vocab) / len(terms))
        return {
            "num_queries_scored": len(scores),
            "mean_query_term_coverage": round(sum(scores) / len(scores), 4) if scores else None,
        }

    qids = sorted(query_terms)
    shuffled = qids[:]
    random.Random(0).shuffle(shuffled)
    by_original_id = {int(r.original_query_id): int(r.query_id) for r in queries_proc.itertuples()}
    return {
        "positional": mean_coverage({k: k for k in qids}),
        "identity_original_id": mean_coverage({k: by_original_id[k] for k in qids if k in by_original_id}),
        "random_permutation": mean_coverage(dict(zip(qids, shuffled))),
    }


# ============================================================================
# Section validators
# ============================================================================
def _validate_documents(report: ValidationReport, processed_dir: Path, raw_dir: Path, parse_warnings) -> pd.DataFrame:
    s = "Documents"
    docs = load_output("documents", processed_dir)
    report.check(s, "documents.csv has exactly 1400 rows", len(docs) == 1400, f"rows = {len(docs)}")
    report.check(s, "columns are doc_id, title, author, bibliography, body", list(docs.columns) == ["doc_id", *DOCUMENT_FIELDS])
    report.check(s, "document IDs are unique integers", docs["doc_id"].is_unique and pd.api.types.is_integer_dtype(docs["doc_id"]))
    reparsed, _ = parse_documents(raw_path("documents", raw_dir))
    report.check(s, "documents.csv equals a fresh parse of cran.all.1400 (no lost or altered text)", reparsed.equals(docs))

    report.note(s, "Empty field counts: " + ", ".join(f"{f}={int((docs[f] == '').sum())}" for f in DOCUMENT_FIELDS))
    empty = _empty_document_ids(docs)
    report.warn_if(s, "no document is completely empty", bool(empty), f"completely empty documents (kept): {empty}")
    irregular = _warnings_for(parse_warnings, "documents", raw_dir)
    report.warn_if(s, "no structural irregularities in cran.all.1400", bool(irregular), irregular)
    return docs


def _validate_queries(report: ValidationReport, processed_dir: Path, raw_dir: Path, parse_warnings) -> pd.DataFrame:
    s = "Queries"
    queries = load_output("queries", processed_dir)
    original_ids = queries["original_query_id"]
    report.check(s, "queries.csv has exactly 225 rows", len(queries) == 225, f"rows = {len(queries)}")
    report.check(s, "columns are query_id, original_query_id, query_text", list(queries.columns) == ["query_id", "original_query_id", "query_text"])
    report.check(s, "canonical query IDs are exactly 1..N in file order", queries["query_id"].tolist() == list(range(1, len(queries) + 1)))
    report.check(
        s,
        "original query IDs are unique and keep their zero padding (e.g. '001')",
        original_ids.is_unique and original_ids.str.len().eq(3).all() and original_ids.iloc[0] == "001",
        f"first IDs: {original_ids.head(4).tolist()}",
    )
    reparsed, _ = parse_queries(raw_path("queries", raw_dir))
    report.check(s, "queries.csv equals a fresh parse of cran.qry (same IDs, mapping and text)", reparsed.equals(queries))

    embedded = [
        f"query_id {a.query_id} (original {a.original_query_id}) contains the full text of "
        f"query_id {b.query_id} (original {b.original_query_id})"
        for a in queries.itertuples()
        for b in queries.itertuples()
        if a.query_id != b.query_id and len(b.query_text) >= 40 and b.query_text in a.query_text
    ]
    report.warn_if(
        s,
        "no query text contains the full text of another query",
        bool(embedded),
        "\n".join(embedded) + "\n(kept unchanged; appears to be a copy error in cran.qry)",
    )
    irregular = _warnings_for(parse_warnings, "queries", raw_dir)
    report.warn_if(s, "no structural irregularities in cran.qry", bool(irregular), irregular)
    return queries


def _validate_relevance(report, processed_dir, raw_dir, docs, queries, policy: RelevancePolicy, parse_warnings) -> pd.DataFrame:
    s = "Relevance judgments"
    rel = load_output("relevance", processed_dir)
    special = load_output("relevance_special", processed_dir)
    try:
        raw, _ = parse_qrels(raw_path("qrels", raw_dir))
    except DataIntegrityError as exc:
        report.check(s, "cranqrel has 1837 well-formed rows", False, str(exc))
        return rel
    report.check(s, "cranqrel has 1837 well-formed rows (3 integer fields, known code)", True, f"rows = {len(raw)}")

    key = ["query_id", "doc_id", "relevance_grade"]
    combined = pd.concat([rel[key], special.rename(columns={"original_relevance_grade": "relevance_grade"})[key]])
    report.check(
        s,
        "relevance.csv + relevance_special.csv contain exactly the raw rows",
        combined.sort_values(key).reset_index(drop=True).equals(raw[key].sort_values(key).reset_index(drop=True)),
        f"{len(rel)} ordinary + {len(special)} special = {len(combined)} (raw {len(raw)})",
    )
    report.check(s, "no duplicate query-document pairs", not combined.duplicated(["query_id", "doc_id"]).any())
    report.check(
        s,
        "all referenced doc_ids and query_ids exist",
        combined["doc_id"].isin(docs["doc_id"]).all() and combined["query_id"].isin(queries["query_id"]).all(),
    )
    report.check(
        s,
        "relevance.csv holds only grades 1-5; relevance_special.csv holds only code -1",
        rel["relevance_grade"].between(1, 5).all() and special["original_relevance_grade"].isin(list(SPECIAL_GRADES)).all(),
    )
    report.check(
        s,
        f"binary_relevance follows the policy (relevant grades = {sorted(policy.relevant_grades)})",
        rel["relevance_grade"].map(policy.binary_label).equals(rel["binary_relevance"]),
    )

    grade_counts = combined["relevance_grade"].value_counts().sort_index()
    report.note(s, "Grade counts (all rows): " + ", ".join(f"{g}: {n}" for g, n in grade_counts.items()))
    report.note(
        s,
        f"Special -1 rows: {len(special)} across {special['query_id'].nunique()} queries; kept separately, "
        f"NOT treated as negatives. (query_id:doc_id)\n"
        + _wrap([f"{r.query_id}:{r.doc_id}" for r in special.itertuples()]),
    )
    judged_empty = combined[combined["doc_id"].isin(_empty_document_ids(docs))]
    report.warn_if(
        s,
        "no judgment refers to a completely empty document",
        not judged_empty.empty,
        "; ".join(f"query {r.query_id} -> doc {r.doc_id} (grade {r.relevance_grade})" for r in judged_empty.itertuples()),
    )
    irregular = _warnings_for(parse_warnings, "qrels", raw_dir)
    report.warn_if(s, "no formatting irregularities in cranqrel", bool(irregular), irregular + "\n(parsed as whitespace-separated; values unaffected)")
    report.note(s, "Documents not listed for a query are UNJUDGED, not known to be non-relevant.")
    return rel


def _validate_preprocessing(report, processed_dir, docs, queries, config: PreprocessConfig):
    s = "Preprocessing"
    dproc = load_output("documents_processed", processed_dir)
    qproc = load_output("queries_processed", processed_dir)
    report.check(
        s,
        "processed files have the required columns",
        list(dproc.columns) == document_processed_columns() and list(qproc.columns) == QUERY_PROCESSED_COLUMNS,
    )
    report.check(s, "no documents lost (same doc_ids, same order)", dproc["doc_id"].tolist() == docs["doc_id"].tolist(), f"rows = {len(dproc)}")
    report.check(
        s,
        "no queries lost (same query_id and original_query_id, same order)",
        qproc[["query_id", "original_query_id"]].equals(queries[["query_id", "original_query_id"]]),
        f"rows = {len(qproc)}",
    )

    ddec, dfail = _decode_token_columns(dproc)
    qdec, qfail = _decode_token_columns(qproc)
    report.check(s, "every token cell is a valid JSON array of strings", not (dfail or qfail), "\n".join((dfail + qfail)[:10]))

    # Re-run the shared preprocessing on the original text and compare every representation.
    mismatches = []
    for i, doc in enumerate(docs.itertuples()):
        for f in DOCUMENT_FIELDS:
            rep = preprocess_text(getattr(doc, f), config)
            stored = (dproc[f"{f}_clean"].iloc[i], ddec[f"{f}_tokens"][i], ddec[f"{f}_tokens_no_stop"][i], ddec[f"{f}_stemmed_tokens"][i])
            if stored != (rep["clean"], rep["tokens"], rep["tokens_no_stop"], rep["stemmed_tokens"]):
                mismatches.append(f"doc {doc.doc_id} field {f}")
    for i, q in enumerate(queries.itertuples()):
        rep = preprocess_text(q.query_text, config)
        stored = (qproc["query_clean"].iloc[i], qdec["query_tokens"][i], qdec["query_tokens_no_stop"][i], qdec["query_stemmed_tokens"][i])
        if stored != (rep["clean"], rep["tokens"], rep["tokens_no_stop"], rep["stemmed_tokens"]):
            mismatches.append(f"query {q.query_id}")
    report.check(
        s,
        "stored tokens equal a fresh run of the same preprocessing on the original text",
        not mismatches,
        ", ".join(mismatches[:10]),
    )

    docs_with_repeats = sum(len(t) > len(set(t)) for t in ddec["body_tokens"])
    report.check(
        s,
        "repeated term occurrences are preserved (term frequency intact)",
        docs_with_repeats > 0,
        f"{docs_with_repeats} documents have repeated body tokens",
    )
    report.note(s, "Documents with empty token lists: " + ", ".join(f"{f}={sum(not t for t in ddec[f + '_tokens'])}" for f in DOCUMENT_FIELDS))
    empty_queries = [int(qid) for qid, t in zip(qproc["query_id"], qdec["query_tokens_no_stop"]) if not t]
    report.warn_if(s, "no query has an empty (no-stopword) token list", bool(empty_queries), f"empty: {empty_queries}")
    return dproc, qproc


def _validate_splits(report, processed_dir, queries, rel, seed: int) -> None:
    s = "Train/validation/test split"
    splits = {name: load_output(name, processed_dir) for name in SPLIT_NAMES}
    report.check(s, f"split files have columns {SPLIT_COLUMNS}", all(list(df.columns) == SPLIT_COLUMNS for df in splits.values()))

    ids = {name: set(df["query_id"]) for name, df in splits.items()}
    overlap = (ids["train"] & ids["val"]) | (ids["train"] & ids["test"]) | (ids["val"] & ids["test"])
    report.check(s, "no query appears in more than one split", not overlap, f"overlapping: {sorted(overlap)}" if overlap else "")
    merged = pd.concat(splits.values()).sort_values("query_id").reset_index(drop=True)
    report.check(s, "every query is in exactly one split, with the same text as queries.csv", merged.equals(queries[SPLIT_COLUMNS]))

    sizes = {name: len(df) for name, df in splits.items()}
    report.check(s, "split sizes follow 70/15/15", sizes == compute_split_sizes(len(queries)), f"sizes = {sizes}")
    rerun = split_queries(queries, rel, seed=seed)
    report.check(s, f"re-running the split with seed {seed} reproduces the saved files", all(rerun[n].equals(splits[n]) for n in SPLIT_NAMES))
    report.note(s, "Documents are NOT split: every split is searched against the full 1,400-document corpus.")


def _validate_query_mapping(report, queries, rel, dproc, qproc) -> dict:
    s = "Query ID mapping"
    qrel_ids = set(rel["query_id"])
    report.check(
        s,
        "qrels use exactly the IDs 1..225 (same count as cran.qry records)",
        qrel_ids == set(range(1, len(queries) + 1)),
        f"distinct qrel query IDs = {len(qrel_ids)}, range {min(qrel_ids)}..{max(qrel_ids)}",
    )
    evidence = query_mapping_evidence(qproc, dproc, rel)
    coverage = {name: result["mean_query_term_coverage"] for name, result in evidence.items()}
    clearly_best = all(v is not None and coverage["positional"] - v > 0.1 for k, v in coverage.items() if k != "positional")
    report.warn_if(
        s,
        "positional mapping matches relevant documents far better than the alternatives",
        not clearly_best,
        "mean coverage of query stems by relevant documents: " + str(coverage),
    )
    report.note(s, "Mean coverage of query stems by their relevant documents' title+body stems:\n" + "\n".join(f"{k} = {v}" for k, v in coverage.items()))
    report.note(s, "The positional correspondence is not stated in the distribution files;\nit is an assumption supported by this evidence.")
    return evidence


def validate_outputs(
    processed_dir: Path,
    raw_dir: Path,
    policy: RelevancePolicy,
    preprocess_config: PreprocessConfig,
    seed: int,
    parse_warnings: list[ParseWarning],
) -> tuple[ValidationReport, dict]:
    """Run every validation section on the files in ``processed_dir``.

    Returns:
        The report and the query-mapping evidence (stored in the statistics file).
    """
    report = ValidationReport()
    docs = _validate_documents(report, processed_dir, raw_dir, parse_warnings)
    queries = _validate_queries(report, processed_dir, raw_dir, parse_warnings)
    rel = _validate_relevance(report, processed_dir, raw_dir, docs, queries, policy, parse_warnings)
    dproc, qproc = _validate_preprocessing(report, processed_dir, docs, queries, preprocess_config)
    _validate_splits(report, processed_dir, queries, rel, seed)
    evidence = _validate_query_mapping(report, queries, rel, dproc, qproc)
    return report, {"query_mapping_evidence": evidence}


# ============================================================================
# Dataset statistics (dataset_statistics.json)
# ============================================================================
def _length_summary(lengths: list[int]) -> dict:
    if not lengths:
        return {"mean": 0.0, "median": 0, "min": 0, "max": 0}
    return {
        "mean": round(statistics.fmean(lengths), 3),
        "median": statistics.median(lengths),
        "min": min(lengths),
        "max": max(lengths),
    }


def _decode(series: pd.Series) -> list[list[str]]:
    return [json.loads(v) for v in series]


def compute_statistics(
    documents: pd.DataFrame,
    queries: pd.DataFrame,
    qrels_raw: pd.DataFrame,
    relevance: pd.DataFrame,
    relevance_special: pd.DataFrame,
    documents_processed: pd.DataFrame,
    queries_processed: pd.DataFrame,
    split_summary: dict,
    parse_warnings: list,
    extra: dict,
    top_k: int = 25,
) -> dict:
    """Assemble the statistics dictionary (all values JSON-serializable)."""
    tokens = {f: _decode(documents_processed[f"{f}_tokens"]) for f in DOCUMENT_FIELDS}
    no_stop = {f: _decode(documents_processed[f"{f}_tokens_no_stop"]) for f in DOCUMENT_FIELDS}
    stemmed = {f: _decode(documents_processed[f"{f}_stemmed_tokens"]) for f in DOCUMENT_FIELDS}
    q_tokens = _decode(queries_processed["query_tokens"])
    q_no_stop = _decode(queries_processed["query_tokens_no_stop"])
    q_stemmed = _decode(queries_processed["query_stemmed_tokens"])

    def counts(per_field: dict[str, list[list[str]]]) -> Counter:
        return Counter(t for lists in per_field.values() for toks in lists for t in toks)

    all_counts, no_stop_counts = counts(tokens), counts(no_stop)
    full_doc_lengths = [sum(len(tokens[f][i]) for f in DOCUMENT_FIELDS) for i in range(len(documents_processed))]
    fields = list(DOCUMENT_FIELDS)
    positives = relevance[relevance["binary_relevance"] == 1]

    return {
        "definitions": {
            "token_representation_for_lengths": "*_tokens (normalized tokens, stopwords retained, no stemming)",
            "document_length": "number of tokens in body_tokens; 'full_document_length' sums all four fields",
            "vocabulary": "distinct tokens over all four document fields of all documents, per representation",
            "token_frequency": "corpus counts over all four document fields",
        },
        "documents": {
            "num_documents": int(len(documents)),
            "num_unique_doc_ids": int(documents["doc_id"].nunique()),
            "empty_field_counts": {f: int((documents[f] == "").sum()) for f in fields},
            "num_documents_with_any_missing_field": int((documents[fields] == "").any(axis=1).sum()),
            "completely_empty_doc_ids": _empty_document_ids(documents),
            **{f"{f}_length_tokens": _length_summary([len(t) for t in tokens[f]]) for f in fields},
            "full_document_length_tokens": _length_summary(full_doc_lengths),
        },
        "queries": {
            "num_queries": int(len(queries)),
            "num_unique_canonical_query_ids": int(queries["query_id"].nunique()),
            "num_unique_original_query_ids": int(queries["original_query_id"].nunique()),
            "original_query_id_range": [queries["original_query_id"].iloc[0], queries["original_query_id"].iloc[-1]],
            "query_length_tokens": _length_summary([len(t) for t in q_tokens]),
            "query_length_tokens_no_stop": _length_summary([len(t) for t in q_no_stop]),
            "query_vocabulary_size": len({t for toks in q_tokens for t in toks}),
            "query_vocabulary_size_stemmed": len({t for toks in q_stemmed for t in toks}),
        },
        "relevance": {
            "num_original_qrel_rows": int(len(qrels_raw)),
            "num_ordinary_relevance_rows": int(len(relevance)),
            "num_special_rows": int(len(relevance_special)),
            "grade_counts": {str(g): int(n) for g, n in qrels_raw["relevance_grade"].value_counts().sort_index().items()},
            "num_binary_relevant": int(len(positives)),
            "num_binary_non_relevant": int(len(relevance) - len(positives)),
            "num_queries_with_judgments": int(qrels_raw["query_id"].nunique()),
            "num_distinct_judged_documents": int(relevance["doc_id"].nunique()),
            "positive_judgments_per_query": _length_summary(positives.groupby("query_id").size().tolist()),
        },
        "splits": split_summary,
        "vocabulary": {
            "tokens": len(all_counts),
            "tokens_no_stop": len(no_stop_counts),
            "stemmed_tokens": len(counts(stemmed)),
            "per_field_tokens": {f: len({t for toks in tokens[f] for t in toks}) for f in fields},
        },
        "token_frequency": {
            "total_tokens": int(sum(all_counts.values())),
            "total_tokens_no_stop": int(sum(no_stop_counts.values())),
            "hapax_legomena_tokens": int(sum(1 for c in all_counts.values() if c == 1)),
            "numeric_token_types": int(sum(1 for t in all_counts if any(ch.isdigit() for ch in t))),
            "hyphenated_token_types": int(sum(1 for t in all_counts if "-" in t)),
            f"top_{top_k}_tokens_no_stop": [[t, int(c)] for t, c in no_stop_counts.most_common(top_k)],
        },
        "parse_warnings": [str(w) for w in parse_warnings],
        **extra,
    }
