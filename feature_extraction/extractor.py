"""Corpus-backed query/document features. Uses only the Python standard library."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Optional

FIELDS = ("title", "author", "bibliography", "body")
DEFAULT_PROCESSED_DIR = Path(__file__).resolve().parents[1] / "preprocessing/data/processed"
FEATURE_NAMES = (
    "query_term_number", "query_term_ratio", "stream_length", "idf",
    "tf_sum", "tf_min", "tf_max", "tf_mean", "tf_variance",
    "normalized_tf_sum", "normalized_tf_min", "normalized_tf_max",
    "normalized_tf_mean", "normalized_tf_variance",
    "tf_idf_sum", "tf_idf_min", "tf_idf_max", "tf_idf_mean", "tf_idf_variance",
    "boolean_model", "vector_space_model", "bm25",
)


def _tokens(values: Iterable[str]) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise TypeError("Pass token strings as a sequence, not a text string")
    result = list(values)
    if any(not isinstance(t, str) or not t for t in result):
        raise ValueError("Tokens must be non-empty strings")
    return result


def _decode(value: str) -> list[str]:
    result = json.loads(value)
    if not isinstance(result, list):
        raise ValueError("A token column must contain a JSON list")
    return _tokens(result)


def _stats(values: list[float]) -> tuple[float, ...]:
    if not values:
        return (0.0,) * 5
    total = math.fsum(values)
    mean = total / len(values)
    variance = math.fsum((v - mean) ** 2 for v in values) / len(values)
    return total, min(values), max(values), mean, variance


def _log_tf(count: int) -> float:
    return 1.0 + math.log10(count) if count else 0.0


def load_query_tokens(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
) -> dict[int, list[str]]:
    """Load Part 1's canonical query IDs and Porter-stemmed, stopword-free tokens."""
    queries = {}
    with (Path(processed_dir) / "queries_processed.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            qid = int(row["query_id"])
            if qid in queries:
                raise ValueError(f"Duplicate query ID: {qid}")
            queries[qid] = _decode(row["query_stemmed_tokens"])
    return queries


class FeatureExtractor:
    """Fit document statistics once, then extract a named 22-feature vector.

    ``documents`` maps document IDs to already preprocessed token sequences.
    The CSV loader concatenates the four fields without deduplicating tokens.
    ``extract`` accepts tokens, so unknown test queries need no relevance data.
    An extractor is a snapshot: rebuild it if the indexed collection changes.
    """

    def __init__(
        self,
        documents: Mapping[int, Iterable[str]],
        *,
        k1: float = 1.2,
        b: float = 0.75,
        k3: Optional[float] = None,
        boolean_mode: str = "or",
    ) -> None:
        if not documents:
            raise ValueError("The document collection must not be empty")
        if not math.isfinite(k1) or k1 <= 0:
            raise ValueError("k1 must be finite and positive")
        if not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("b must be in [0, 1]")
        if k3 is not None and (not math.isfinite(k3) or k3 <= 0):
            raise ValueError("k3 must be None or finite and positive")
        if boolean_mode not in ("or", "and"):
            raise ValueError("boolean_mode must be 'or' or 'and'")
        self.k1, self.b, self.k3 = k1, b, k3
        self.boolean_mode = boolean_mode
        self._counts = {did: Counter(_tokens(tokens)) for did, tokens in documents.items()}
        self.document_ids = tuple(sorted(self._counts))
        self.num_documents = len(self._counts)
        self._lengths = {did: sum(counts.values()) for did, counts in self._counts.items()}
        self.average_length = math.fsum(self._lengths.values()) / self.num_documents
        df: Counter[str] = Counter()
        for counts in self._counts.values():
            df.update(counts.keys())
        self._idf = {t: math.log10(self.num_documents / freq) for t, freq in df.items()}
        self._norms = {
            did: math.sqrt(math.fsum((_log_tf(n) * self._idf[t]) ** 2 for t, n in counts.items()))
            for did, counts in self._counts.items()
        }

    @classmethod
    def from_processed(
        cls, processed_dir: Path = DEFAULT_PROCESSED_DIR, **kwargs,
    ) -> "FeatureExtractor":
        """Load all four stemmed-token fields from existing Part 1 CSV output."""
        documents = {}
        with (Path(processed_dir) / "documents_processed.csv").open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                did = int(row["doc_id"])
                if did in documents:
                    raise ValueError(f"Duplicate document ID: {did}")
                documents[did] = [
                    token for field in FIELDS
                    for token in _decode(row[f"{field}_stemmed_tokens"])
                ]
        return cls(documents, **kwargs)

    def extract(self, query_tokens: Iterable[str], doc_id: int) -> dict[str, float]:
        """Return exactly 22 unscaled finite numbers in ``FEATURE_NAMES`` order.

        Unknown document IDs raise KeyError. Out-of-vocabulary query terms
        remain in coverage/statistics denominators but have IDF weight zero.
        Empty queries have zero query-dependent features; length is preserved.
        """
        counts = self._counts[doc_id]
        query = Counter(_tokens(query_tokens))
        terms = sorted(query)
        length = self._lengths[doc_id]
        tf = [float(counts[t]) for t in terms]
        covered = sum(n > 0 for n in tf)
        idfs = [self._idf.get(t, 0.0) for t in terms]
        normalized = [n / length if length else 0.0 for n in tf]
        tfidf = [_log_tf(counts[t]) * weight for t, weight in zip(terms, idfs)]
        query_weights = [_log_tf(query[t]) * weight for t, weight in zip(terms, idfs)]
        query_norm = math.sqrt(math.fsum(w * w for w in query_weights))
        denominator = query_norm * self._norms[doc_id]
        cosine = math.fsum(q * d for q, d in zip(query_weights, tfidf)) / denominator if denominator else 0.0
        # Guard roundoff just outside the mathematical cosine range.
        cosine = min(1.0, max(0.0, cosine))
        boolean = covered > 0 if self.boolean_mode == "or" else bool(terms) and covered == len(terms)
        bm25_terms = []
        if length and self.average_length:
            normalization = self.k1 * (1 - self.b + self.b * length / self.average_length)
            for t, freq, weight in zip(terms, tf, idfs):
                query_factor = ((self.k3 + 1) * query[t] / (self.k3 + query[t])) if self.k3 is not None else 1.0
                bm25_terms.append(weight * (self.k1 + 1) * freq / (normalization + freq) * query_factor)
        values = (
            float(covered), covered / len(terms) if terms else 0.0,
            float(length), math.fsum(idfs),
            *_stats(tf), *_stats(normalized), *_stats(tfidf),
            float(boolean), cosine, math.fsum(bm25_terms),
        )
        return dict(zip(FEATURE_NAMES, values))

    def extract_vector(self, query_tokens: Iterable[str], doc_id: int) -> tuple[float, ...]:
        """Return the same features as a tuple in the documented fixed order."""
        return tuple(self.extract(query_tokens, doc_id).values())
