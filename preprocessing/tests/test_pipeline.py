"""Tests for the Task 1 pipeline.

Unit tests use small synthetic files written to pytest's ``tmp_path``.
End-to-end tests run the full pipeline on the real Cranfield files in
``data/raw`` and write into a temporary directory (the committed outputs in
``data/processed`` are not touched).

Run from the project root:  python -m pytest -v
"""

from __future__ import annotations

import json
import textwrap

import pandas as pd
import pytest

from main import run_pipeline
from src.config import (
    DataIntegrityError,
    RAW_DIR,
    load_output,
    output_path,
    raw_path,
    read_csv,
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
from src.preprocess import (
    PreprocessConfig,
    deserialize_tokens,
    normalize_text,
    preprocess_documents,
    preprocess_queries,
    preprocess_text,
    remove_stopwords,
    serialize_tokens,
    stem_tokens,
    tokenize,
)
from src.split_dataset import compute_split_sizes, split_queries


def write(tmp_path, name: str, content: str):
    path = tmp_path / name
    path.write_text(textwrap.dedent(content).lstrip("\n"), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def docs_file(tmp_path):
    return write(
        tmp_path,
        "docs",
        """
        .I 1
        .T
        experimental investigation of the aerodynamics of a
        wing in a slipstream .
        .A
        brenckman,m.
        .B
        j. ae. scs. 25, 1958, 324.
        .W
        experimental investigation of the aerodynamics of a
        wing in a slipstream .
          an experimental study of a wing .
        .I 2
        .T
        shear flow .
        .A
        .B
        .W
        .5 per cent of the flow ... .t is lowercase and not a marker .
        """,
    )


@pytest.fixture
def queries_file(tmp_path):
    return write(
        tmp_path,
        "qry",
        """
        .I 001
        .W
        what similarity laws must be obeyed when constructing aeroelastic models
        of heated high speed aircraft .
        .I 002
        .W
        what are the structural and aeroelastic problems .
        .I 004
        .W
        what problems of heat conduction in composite slabs have been solved so
        far .
        """,
    )


@pytest.fixture
def small_queries():
    return pd.DataFrame(
        {
            "query_id": list(range(1, 21)),
            "original_query_id": [f"{i * 2:03d}" for i in range(1, 21)],
            "query_text": [f"query number {i}" for i in range(1, 21)],
        }
    )


@pytest.fixture
def small_relevance(small_queries):
    rows = [
        {"query_id": q, "doc_id": d, "relevance_grade": 2, "binary_relevance": 1}
        for q in small_queries["query_id"]
        for d in range(1, q % 5 + 2)
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1-3. Document parsing
# ---------------------------------------------------------------------------
def test_document_markers_parsed_into_separate_fields(docs_file):
    docs, warnings = parse_documents(docs_file, expected_count=2)
    assert docs.columns.tolist() == ["doc_id", "title", "author", "bibliography", "body"]
    assert docs["doc_id"].tolist() == [1, 2]
    first = docs.iloc[0]
    assert first["author"] == "brenckman,m."
    assert first["bibliography"] == "j. ae. scs. 25, 1958, 324."
    assert first["title"].startswith("experimental investigation")
    assert warnings == []


def test_multiline_fields_are_joined_with_single_spaces(docs_file):
    docs, _ = parse_documents(docs_file, expected_count=2)
    assert docs.iloc[0]["title"] == "experimental investigation of the aerodynamics of a wing in a slipstream ."
    assert docs.iloc[0]["body"].endswith("wing in a slipstream . an experimental study of a wing .")
    assert "\n" not in docs.iloc[0]["body"]


def test_empty_fields_and_dot_lines_inside_text(docs_file):
    docs, _ = parse_documents(docs_file, expected_count=2)
    second = docs.iloc[1]
    assert second["author"] == "" and second["bibliography"] == ""
    # ".5 per cent" and ".t" are text, not markers.
    assert second["body"].startswith(".5 per cent") and ".t is lowercase" in second["body"]


def test_missing_marker_gives_empty_field_and_warning(tmp_path):
    path = write(tmp_path, "d", ".I 7\n.T\ntitle only .\n.W\nbody text .\n")
    docs, warnings = parse_documents(path, expected_count=1)
    assert docs.iloc[0]["author"] == "" and docs.iloc[0]["bibliography"] == ""
    assert any("missing field marker" in w.message for w in warnings)


def test_repeated_marker_stays_in_current_field_and_is_reported(tmp_path):
    path = write(tmp_path, "d", ".I 3\n.T\nt .\n.A\na\n.B\nb\n.W\nfirst part\n.A\nsecond part\n.W\nthird part\n")
    docs, warnings = parse_documents(path, expected_count=1)
    assert docs.iloc[0]["author"] == "a"
    assert docs.iloc[0]["body"] == "first part second part third part"
    assert len(warnings) == 2 and all("repeated" in w.message for w in warnings)


def test_document_integrity_errors(tmp_path):
    with pytest.raises(DataIntegrityError, match="duplicate"):
        parse_documents(write(tmp_path, "a", ".I 1\n.W\nx\n.I 1\n.W\ny\n"), expected_count=None)
    with pytest.raises(DataIntegrityError, match="before the first"):
        parse_documents(write(tmp_path, "b", "stray text\n.I 1\n.W\nx\n"), expected_count=None)
    with pytest.raises(DataIntegrityError, match="expected 3 records"):
        parse_documents(write(tmp_path, "c", ".I 1\n.W\nx\n"), expected_count=3)
    with pytest.raises(DataIntegrityError, match="non-integer"):
        parse_documents(write(tmp_path, "e", ".I abc\n.W\nx\n"), expected_count=None)


# ---------------------------------------------------------------------------
# 4-6. Query parsing and ID mapping
# ---------------------------------------------------------------------------
def test_query_parsing_multiline_text(queries_file):
    queries, warnings = parse_queries(queries_file, expected_count=3)
    assert queries.columns.tolist() == ["query_id", "original_query_id", "query_text"]
    assert queries.iloc[0]["query_text"] == (
        "what similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft ."
    )
    assert warnings == []


def test_original_query_ids_preserved_zero_padded(queries_file, tmp_path):
    queries, _ = parse_queries(queries_file, expected_count=3)
    assert queries["original_query_id"].tolist() == ["001", "002", "004"]
    # And they survive a CSV round trip as strings.
    write_csv(queries, tmp_path / "q.csv")
    assert read_csv(tmp_path / "q.csv")["original_query_id"].tolist() == ["001", "002", "004"]


def test_canonical_query_id_is_position_not_original_id(queries_file):
    queries, _ = parse_queries(queries_file, expected_count=3)
    assert queries["query_id"].tolist() == [1, 2, 3]
    row = queries.loc[queries["original_query_id"] == "004"].iloc[0]
    assert row["query_id"] == 3  # the third record, NOT 4
    info = verify_query_id_mapping(queries, [1, 2, 3])
    assert info["qrel_ids_contiguous_1_to_n"] is True
    with pytest.raises(DataIntegrityError):
        verify_query_id_mapping(queries, [1, 2, 4])  # qrel ID 4 has no query


def test_duplicate_original_query_ids_rejected(tmp_path):
    with pytest.raises(DataIntegrityError, match="duplicate"):
        parse_queries(write(tmp_path, "q", ".I 001\n.W\na\n.I 001\n.W\nb\n"), expected_count=None)


# ---------------------------------------------------------------------------
# 7-8. Relevance judgments
# ---------------------------------------------------------------------------
def test_relevance_parsing_and_irregular_whitespace(tmp_path):
    path = write(tmp_path, "qrel", "1 184 2 \n1 29 3 \n2 85  4 \n2 12 -1")
    qrels, warnings = parse_qrels(path, expected_rows=4)
    assert qrels[["query_id", "doc_id", "relevance_grade"]].values.tolist() == [
        [1, 184, 2], [1, 29, 3], [2, 85, 4], [2, 12, -1]
    ]
    assert len(warnings) == 1 and "whitespace" in warnings[0].message


@pytest.mark.parametrize("content", ["1 184\n", "1 184 2 9\n", "1 x 2\n", "1 184 7\n", "1 5 2\n1 5 3\n"])
def test_malformed_or_duplicate_qrels_raise(tmp_path, content):
    with pytest.raises(DataIntegrityError):
        parse_qrels(write(tmp_path, "qrel", content), expected_rows=None)


def test_special_grade_kept_separate_and_never_binarized(tmp_path):
    qrels, _ = parse_qrels(write(tmp_path, "qrel", "1 10 1\n1 11 4\n1 12 -1\n2 13 3\n2 14 5\n"), expected_rows=None)
    rel, special = split_relevance(qrels)
    assert -1 not in rel["relevance_grade"].tolist()
    assert special.columns.tolist() == ["query_id", "doc_id", "original_relevance_grade", "reason"]
    assert special[["query_id", "doc_id", "original_relevance_grade"]].values.tolist() == [[1, 12, -1]]
    assert "undocumented" in special.iloc[0]["reason"]
    assert len(rel) + len(special) == len(qrels)
    # Default policy: 1-4 relevant, 5 not relevant.
    assert dict(zip(rel["relevance_grade"], rel["binary_relevance"])) == {1: 1, 4: 1, 3: 1, 5: 0}


def test_configurable_relevance_threshold(tmp_path):
    qrels, _ = parse_qrels(write(tmp_path, "qrel", "1 10 1\n1 11 3\n1 12 4\n"), expected_rows=None)
    rel, _ = split_relevance(qrels, RelevancePolicy.from_max_grade(3))
    assert rel["binary_relevance"].tolist() == [1, 1, 0]
    assert rel["relevance_grade"].tolist() == [1, 3, 4]  # graded data untouched
    with pytest.raises(ValueError):
        RelevancePolicy(frozenset({-1, 1}))


def test_relevance_references_checked(tmp_path):
    qrels, _ = parse_qrels(write(tmp_path, "qrel", "1 10 1\n3 99 2\n"), expected_rows=None)
    with pytest.raises(DataIntegrityError, match="doc_ids.*99"):
        check_references(qrels, doc_ids=[10], query_ids=[1, 2, 3])
    with pytest.raises(DataIntegrityError, match="query_ids"):
        check_references(qrels, doc_ids=[10, 99], query_ids=[1, 2])


# ---------------------------------------------------------------------------
# 9-11. Text preprocessing
# ---------------------------------------------------------------------------
def test_normalization():
    assert normalize_text("  Wing\tIN a\n\nSlipstream .  ") == "wing in a slipstream ."
    assert normalize_text("ﬁnite​ flow") == "finite flow"  # NFKC ligature, zero-width space removed
    assert normalize_text("") == ""


def test_tokenization_keeps_scientific_terms_and_numbers():
    text = "Boundary-layer flow at Mach 2.5, x-15 at 15,000 fps (chapman-enskog) ; kuchemann's /slip/ ."
    assert tokenize(text) == [
        "boundary-layer", "flow", "at", "mach", "2.5", "x-15", "at", "15,000", "fps",
        "chapman-enskog", "kuchemann", "slip",
    ]
    assert tokenize("j. ae. scs. 25, 1958, 324.") == ["j", "ae", "scs", "25", "1958", "324"]
    assert tokenize("a  b   a") == ["a", "b", "a"]  # duplicates and order kept
    assert tokenize("boundary-layer", PreprocessConfig(split_hyphens=True)) == ["boundary", "layer"]
    assert tokenize("") == []


def test_stopword_removal_is_optional_and_non_destructive():
    rep = preprocess_text("what is the effect of the shock on the boundary layer")
    assert rep["tokens"][:3] == ["what", "is", "the"]  # retained version intact
    assert rep["tokens_no_stop"] == ["effect", "shock", "boundary", "layer"]
    assert remove_stopwords(["the", "wing"]) == ["wing"]


def test_porter_stemming():
    assert stem_tokens(["flows", "pressures", "relaxation", "theory"]) == ["flow", "pressur", "relax", "theori"]
    rep = preprocess_text("the flows of heated gases")
    assert rep["stemmed_tokens"] == ["flow", "heat", "gase"]
    assert len(rep["stemmed_tokens"]) == len(rep["tokens_no_stop"])


def test_same_preprocessing_for_documents_and_queries():
    text = "Heat transfer at hypersonic speeds ."
    docs = pd.DataFrame([{"doc_id": 1, "title": text, "author": "", "bibliography": "", "body": text}])
    queries = pd.DataFrame([{"query_id": 1, "original_query_id": "001", "query_text": text}])
    d = preprocess_documents(docs).iloc[0]
    q = preprocess_queries(queries).iloc[0]
    assert d["title_tokens"] == q["query_tokens"] == d["body_tokens"]
    assert d["title_stemmed_tokens"] == q["query_stemmed_tokens"]
    assert d["author_tokens"] == "[]" and d["author_clean"] == ""


# ---------------------------------------------------------------------------
# 12-13. Splitting
# ---------------------------------------------------------------------------
def test_split_is_deterministic_and_seed_dependent(small_queries, small_relevance):
    a = split_queries(small_queries, small_relevance, seed=42)
    b = split_queries(small_queries, small_relevance, seed=42)
    c = split_queries(small_queries, small_relevance, seed=7)
    for name in ("train", "val", "test"):
        assert a[name].equals(b[name])
    assert any(not a[n].equals(c[n]) for n in ("train", "val", "test"))


def test_split_has_no_overlap_and_covers_all(small_queries, small_relevance):
    splits = split_queries(small_queries, small_relevance, seed=42)
    ids = [set(df["query_id"]) for df in splits.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert set().union(*ids) == set(small_queries["query_id"])
    assert {k: len(v) for k, v in splits.items()} == compute_split_sizes(20)
    assert splits["train"].columns.tolist() == ["query_id", "original_query_id", "query_text"]


def test_split_does_not_depend_on_binary_threshold(small_queries, small_relevance):
    strict = small_relevance.assign(binary_relevance=(small_relevance["doc_id"] % 2).astype(int))
    a = split_queries(small_queries, small_relevance, seed=42)
    b = split_queries(small_queries, strict, seed=42)
    assert all(a[n].equals(b[n]) for n in ("train", "val", "test"))


def test_split_sizes():
    assert compute_split_sizes(225) == {"train": 157, "val": 34, "test": 34}
    assert sum(compute_split_sizes(101).values()) == 101


# ---------------------------------------------------------------------------
# 14. JSON serialization of token arrays
# ---------------------------------------------------------------------------
def test_token_json_serialization_round_trip(tmp_path):
    tokens = ["boundary-layer", "can't", "15,000", 'quote"d', "flow", "flow"]
    encoded = serialize_tokens(tokens)
    assert encoded.startswith("[") and json.loads(encoded) == tokens
    assert deserialize_tokens(encoded) == tokens
    assert serialize_tokens([]) == "[]"
    df = pd.DataFrame({"doc_id": [1], "body_tokens": [encoded]})
    write_csv(df, tmp_path / "t.csv")
    assert read_csv(tmp_path / "t.csv", decode_tokens=True)["body_tokens"][0] == tokens
    with pytest.raises(ValueError):
        deserialize_tokens('"not a list"')


# ---------------------------------------------------------------------------
# 15. End-to-end on the supplied Cranfield files
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pipeline_output(tmp_path_factory):
    if not all(raw_path(k).is_file() for k in ("documents", "queries", "qrels", "qrels_readme")):
        pytest.fail(f"Cranfield raw files are missing from {RAW_DIR}")
    out = tmp_path_factory.mktemp("processed")
    summary = run_pipeline(raw_dir=RAW_DIR, processed_dir=out)
    return out, summary


def test_end_to_end_counts(pipeline_output):
    out, summary = pipeline_output
    assert summary["documents"] == 1400
    assert summary["queries"] == 225
    assert summary["qrel_rows"] == 1837
    assert summary["relevance_rows"] == 1612 and summary["special_rows"] == 225
    assert len(load_output("documents_processed", out)) == 1400
    assert len(load_output("queries_processed", out)) == 225
    assert {k: v["num_queries"] for k, v in summary["splits"].items()} == {"train": 157, "val": 34, "test": 34}


def test_end_to_end_known_values(pipeline_output):
    out, _ = pipeline_output
    docs = load_output("documents", out).set_index("doc_id")
    assert docs.loc[1, "author"] == "brenckman,m."
    assert docs.loc[471, "title"] == "" and docs.loc[471, "body"] == ""  # empty record kept
    queries = load_output("queries", out).set_index("query_id")
    assert queries.loc[1, "original_query_id"] == "001"
    assert queries.loc[3, "original_query_id"] == "004"
    assert queries.loc[225, "original_query_id"] == "365"
    special = load_output("relevance_special", out)
    assert special["original_relevance_grade"].eq(-1).all() and special["query_id"].nunique() == 225
    rel = load_output("relevance", out)
    assert rel["relevance_grade"].value_counts().to_dict() == {3: 734, 2: 387, 4: 363, 1: 128}


def test_end_to_end_reports_and_tokens(pipeline_output):
    out, _ = pipeline_output
    report = output_path("report", out).read_text()
    assert "OVERALL RESULT: PASS" in report
    assert not [line for line in report.splitlines() if line.startswith("[FAIL]")]
    assert "critical failures: 0" in report
    stats = json.loads(output_path("statistics", out).read_text())
    assert stats["documents"]["num_documents"] == 1400
    assert stats["relevance"]["num_special_rows"] == 225
    ev = stats["query_mapping_evidence"]
    assert ev["positional"]["mean_query_term_coverage"] > ev["identity_original_id"]["mean_query_term_coverage"]
    dproc = load_output("documents_processed", out, decode_tokens=True)
    assert all(isinstance(t, list) for t in dproc["body_tokens"])
    assert dproc.loc[dproc["doc_id"] == 1, "title_tokens"].iloc[0][:2] == ["experimental", "investigation"]


def test_end_to_end_is_rerunnable_without_duplicates(pipeline_output):
    out, _ = pipeline_output
    before = {k: load_output(k, out) for k in ("documents", "relevance", "train", "test")}
    run_pipeline(raw_dir=RAW_DIR, processed_dir=out)
    for key, df in before.items():
        assert load_output(key, out).equals(df)
