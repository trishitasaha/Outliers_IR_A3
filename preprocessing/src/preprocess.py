"""Text normalization, tokenization, stopword removal and stemming.

The SAME functions are applied to every document field and to queries, so a
query term and a document term are always produced by identical rules.

Representations produced for each text:

* ``clean``          normalized text (NFKC, lower-cased, whitespace collapsed);
                     punctuation is kept so the text is still readable.
* ``tokens``         all tokens, stopwords retained, original order, duplicates kept.
* ``tokens_no_stop`` ``tokens`` with English stopwords removed.
* ``stemmed_tokens`` Porter stems of ``tokens_no_stop``.

Tokenization rules (see :data:`TOKEN_RE`):

* A token is a run of letters/digits.
* Hyphenated compounds stay whole: ``boundary-layer``, ``x-15``, ``re-entry``.
* Decimal points / thousands separators between digits stay inside numbers:
  ``0.1``, ``15.4``, ``15,000``.
* Internal apostrophes are kept (``can't``); a possessive ``'s`` is dropped
  (``kuchemann's`` -> ``kuchemann``).
* All other punctuation separates tokens and is discarded, e.g. the trailing
  `` .`` that ends every Cranfield sentence, ``/slip flow/`` -> ``slip``, ``flow``.
* Numbers and single-character tokens are NOT removed.

Cranfield writes an em dash as the word ``-dash`` (e.g. ``exact -dash or``);
this yields the token ``dash``, which is kept as-is and documented.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd
from nltk.stem.porter import PorterStemmer

from src.config import DOCUMENT_FIELDS

# Fixed snapshot of the NLTK English stopword list (179 words, nltk 3.9.1).
# It is embedded rather than downloaded so the pipeline runs offline and every
# group member gets exactly the same list.
ENGLISH_STOPWORDS: frozenset[str] = frozenset(
    """
    i me my myself we our ours ourselves you you're you've you'll you'd your yours
    yourself yourselves he him his himself she she's her hers herself it it's its
    itself they them their theirs themselves what which who whom this that that'll
    these those am is are was were be been being have has had having do does did
    doing a an the and but if or because as until while of at by for with about
    against between into through during before after above below to from up down
    in out on off over under again further then once here there when where why how
    all any both each few more most other some such no nor not only own same so
    than too very s t can will just don don't should should've now d ll m o re ve
    y ain aren aren't couldn couldn't didn didn't doesn doesn't hadn hadn't hasn
    hasn't haven haven't isn isn't ma mightn mightn't mustn mustn't needn needn't
    shan shan't shouldn shouldn't wasn wasn't weren weren't won won't wouldn wouldn't
    """.split()
)

_ALNUM = r"[^\W_]"  # any Unicode letter or digit
TOKEN_RE = re.compile(
    rf"{_ALNUM}+"
    rf"(?:-{_ALNUM}+"            # hyphenated compound: boundary-layer, x-15
    rf"|(?<=\d)[.,]\d+"          # decimal / thousands separator: 0.1, 15,000
    rf"|'{_ALNUM}+"              # internal apostrophe: can't, kuchemann's
    rf")*"
)
_WHITESPACE_RE = re.compile(r"\s+")
_POSSESSIVE_RE = re.compile(r"'s$")

_STEMMER = PorterStemmer()  # NLTK default mode (NLTK_EXTENSIONS)


@dataclass(frozen=True)
class PreprocessConfig:
    """Options for the shared preprocessing policy.

    Attributes:
        split_hyphens: If True, hyphenated compounds are split into their parts
            (``boundary-layer`` -> ``boundary``, ``layer``). Default False keeps
            scientific compounds such as ``x-15`` intact.
    """

    split_hyphens: bool = False

    def describe(self) -> dict:
        return {
            "unicode_normalization": "NFKC",
            "lowercase": True,
            "control_characters": "format chars removed, other control chars -> space",
            "whitespace": "tabs/newlines -> space, runs collapsed, ends stripped",
            "tokenizer": "regex " + TOKEN_RE.pattern,
            "possessive_s_removed": True,
            "split_hyphens": self.split_hyphens,
            "stopwords": f"embedded NLTK English list ({len(ENGLISH_STOPWORDS)} words)",
            "stemmer": "nltk PorterStemmer (NLTK_EXTENSIONS mode), applied to tokens_no_stop",
        }


DEFAULT_CONFIG = PreprocessConfig()


def normalize_text(text: str) -> str:
    """Normalize text without removing any content words or punctuation.

    Steps: NFKC normalization, lower-casing, removal of invisible format
    characters, control characters -> space, whitespace collapsing.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).lower()
    chars = []
    for ch in text:
        category = unicodedata.category(ch)
        if category == "Cf":  # zero-width and other invisible format characters
            continue
        chars.append(" " if category == "Cc" else ch)  # tabs, newlines, other controls
    return _WHITESPACE_RE.sub(" ", "".join(chars)).strip()


def tokenize(text: str, config: PreprocessConfig = DEFAULT_CONFIG) -> list[str]:
    """Normalize ``text`` and split it into tokens (order and duplicates preserved)."""
    tokens = [_POSSESSIVE_RE.sub("", t) for t in TOKEN_RE.findall(normalize_text(text))]
    if config.split_hyphens:
        tokens = [part for t in tokens for part in t.split("-") if part]
    return tokens


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Return ``tokens`` without English stopwords (a new list)."""
    return [t for t in tokens if t not in ENGLISH_STOPWORDS]


@lru_cache(maxsize=None)
def stem(token: str) -> str:
    """Porter-stem a single token (cached; the stemmer is deterministic)."""
    return _STEMMER.stem(token)


def stem_tokens(tokens: list[str]) -> list[str]:
    """Return the Porter stems of ``tokens`` (a new list)."""
    return [stem(t) for t in tokens]


def preprocess_text(text: str, config: PreprocessConfig = DEFAULT_CONFIG) -> dict[str, object]:
    """Produce every representation of one piece of text."""
    tokens = tokenize(text, config)
    no_stop = remove_stopwords(tokens)
    return {
        "clean": normalize_text(text),
        "tokens": tokens,
        "tokens_no_stop": no_stop,
        "stemmed_tokens": stem_tokens(no_stop),
    }


def serialize_tokens(tokens: list[str]) -> str:
    """Encode a token list as a JSON array string for storage in a CSV cell."""
    return json.dumps(list(tokens), ensure_ascii=False)


def deserialize_tokens(value: str) -> list[str]:
    """Decode a JSON array string produced by :func:`serialize_tokens`."""
    tokens = json.loads(value)
    if not isinstance(tokens, list):
        raise ValueError(f"Expected a JSON array, got {type(tokens).__name__}")
    return tokens


def document_processed_columns() -> list[str]:
    """Column order of ``documents_processed.csv``."""
    cols = ["doc_id"]
    for suffix in ("clean", "tokens", "tokens_no_stop", "stemmed_tokens"):
        for f in DOCUMENT_FIELDS:
            cols.append(f"{f}_{suffix}")
    return cols


QUERY_PROCESSED_COLUMNS = [
    "query_id",
    "original_query_id",
    "query_clean",
    "query_tokens",
    "query_tokens_no_stop",
    "query_stemmed_tokens",
]


def _representations_to_columns(prefix: str, rep: dict[str, object]) -> dict[str, str]:
    return {
        f"{prefix}_clean": rep["clean"],
        f"{prefix}_tokens": serialize_tokens(rep["tokens"]),
        f"{prefix}_tokens_no_stop": serialize_tokens(rep["tokens_no_stop"]),
        f"{prefix}_stemmed_tokens": serialize_tokens(rep["stemmed_tokens"]),
    }


def preprocess_documents(documents: pd.DataFrame, config: PreprocessConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Preprocess each document field independently (fields are never merged).

    Returns a DataFrame with the columns of :func:`document_processed_columns`;
    token columns hold JSON array strings. Every input document is kept, even if
    all of its fields are empty.
    """
    rows = []
    for record in documents.itertuples(index=False):
        row: dict[str, object] = {"doc_id": int(record.doc_id)}
        for f in DOCUMENT_FIELDS:
            row.update(_representations_to_columns(f, preprocess_text(getattr(record, f), config)))
        rows.append(row)
    processed = pd.DataFrame(rows, columns=document_processed_columns())
    if len(processed) != len(documents) or set(processed["doc_id"]) != set(documents["doc_id"]):
        raise AssertionError("Document IDs changed during preprocessing")
    return processed


def preprocess_queries(queries: pd.DataFrame, config: PreprocessConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Preprocess queries with exactly the same functions used for documents.

    Queries that yield no tokens are kept (and flagged by validation).
    """
    rows = []
    for record in queries.itertuples(index=False):
        row: dict[str, object] = {
            "query_id": int(record.query_id),
            "original_query_id": record.original_query_id,
        }
        row.update(_representations_to_columns("query", preprocess_text(record.query_text, config)))
        rows.append(row)
    processed = pd.DataFrame(rows, columns=QUERY_PROCESSED_COLUMNS)
    if len(processed) != len(queries) or set(processed["query_id"]) != set(queries["query_id"]):
        raise AssertionError("Query IDs changed during preprocessing")
    return processed
