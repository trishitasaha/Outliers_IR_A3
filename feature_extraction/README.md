# Part 3: the 22 query–document features

This package extracts features only. It does not construct a training matrix,
assign relevance labels, sample negatives, split queries, scale features, or train
an SVM. It uses the Python standard library; tests use the existing pytest dependency.

From the repository root:

```python
from feature_extraction import FEATURE_NAMES, FeatureExtractor, load_query_tokens

extractor = FeatureExtractor.from_processed()
queries = load_query_tokens()
features = extractor.extract(queries[1], doc_id=184)  # ordered name -> float
vector = extractor.extract_vector(queries[1], doc_id=184)  # tuple of 22 floats
assert tuple(features) == FEATURE_NAMES
```

Pass an alternative processed-data directory to both loaders if necessary. IDs
are Part 1's canonical `query_id` (1–225), not `original_query_id`. The loader
uses JSON token arrays, not the human-readable clean-text columns.

For a new query, pass its tokens to `extract` after applying **the same Part 1
normalization, tokenization, stopword removal and Porter stemming**. No qrels or
saved query ID are needed. For a different collection, construct
`FeatureExtractor({doc_id: token_sequence, ...})`; rebuild the extractor when
the document collection changes. It snapshots the corpus at construction.

## Document representation

Concatenate `title_stemmed_tokens`, `author_stemmed_tokens`,
`bibliography_stemmed_tokens`, and `body_stemmed_tokens`, in that order. Keep
all token occurrences, including title text repeated in the body. There are no
field weights and exactly 22 features, not 22 per field. Empty documents are
retained and count toward N and average document length. Query splits remain
those supplied by Part 1; the entire fixed collection supplies corpus statistics.

## Definitions and feature order

Let Q be the **set of distinct preprocessed query terms**, m = |Q|, f(t,d) the
raw document occurrence count, L the document token count, N the collection
size, and df(t) the number of documents containing t. Use:

- `I(t) = log10(N / df(t))` for indexed terms.
- `w(f) = 1 + log10(f)` if f > 0, otherwise 0.
- `x(t) = f(t,d)`; `n(t) = x(t)/L` (zero when L = 0).
- `z(t) = w(f(t,d)) * I(t)`.

Summary statistics always include **all terms in Q**, including zeros for absent
terms. Variance is population variance, `sum((v - mean)^2)/m`. For an empty
query, all summaries and query-dependent features are zero. Stream length still
describes the document.

| Position | Feature name | Definition |
|---|---|---|
| 1 | query_term_number | Number of terms in Q with positive document TF |
| 2 | query_term_ratio | Feature 1 / m, or zero for m = 0 |
| 3 | stream_length | L |
| 4 | idf | Sum of I(t) over Q |
| 5–9 | tf_sum, tf_min, tf_max, tf_mean, tf_variance | Statistics of x(t) |
| 10–14 | normalized_tf_sum, normalized_tf_min, normalized_tf_max, normalized_tf_mean, normalized_tf_variance | Statistics of n(t) |
| 15–19 | tf_idf_sum, tf_idf_min, tf_idf_max, tf_idf_mean, tf_idf_variance | Statistics of z(t) |
| 20 | boolean_model | 1 if at least one term matches, else 0 (default OR) |
| 21 | vector_space_model | Cosine of log-TF × IDF query/document vectors |
| 22 | bm25 | Formula below |

Cosine uses query weights `w(f(t,q))*I(t)` and document weights
`w(f(t,d))*I(t)`. Its document norm includes **all document terms**, not just
matching terms. A zero norm gives cosine zero. Repeated query terms change query
vector weights, but do not repeat terms in the summary statistics.

BM25 follows the lecture's equation:

```text
sum over t in Q:
    I(t) * (k1 + 1) * f(t,d)
    / (f(t,d) + k1 * (1 - b + b * L / average_length))
```

Defaults are `k1=1.2`, `b=0.75`; no tuning uses query labels. If `k3` is supplied,
multiply each summand by `(k3+1)*f(t,q)/(k3+f(t,q))`, as in the lecture's optional
long-query variant. Empty documents and an entirely empty corpus produce BM25
zero. An empty document mapping is rejected.

## Source decisions and ambiguities

- `PA3.pdf`, Method B, supplies the required 22 feature types.
- `lecture-wise/LE12_Learning_to_rank.pptx`, slide 23, explicitly cites the
  Microsoft feature list: <https://www.microsoft.com/en-us/research/project/mslr/>.
  Its first 22 feature types match the assignment, with the first two named
  **covered query term number/ratio**. The original Microsoft list repeats each
  feature type over five web streams; this implementation uses the agreed single
  concatenated Cranfield stream.
- `lecture-wise/LE4-.pptx`, slide 22, defines unsmoothed log10 IDF; slide 18
  defines `1+log10(TF)`; slides 41 and 54 describe cosine scoring. Slide 26's
  embedded formula instead shows `log(1+TF)`. We explicitly choose slide 18's
  piecewise log-TF convention for TF-IDF and cosine. The plain TF summary
  features still use raw counts. No scikit-learn smoothing is applied.
- `lecture-wise/LE9_RankingFormat.pptx`, physical slides 38–39 (footer 37–38),
  gives the BM25 formula, optional query-TF factor, and parameter guidance. Use
  the displayed `log(N/df)` IDF, not Lucene's different smoothed variant.
- The sources do not fully specify IDF aggregation, duplicate-query semantics,
  variance denominator, or a Boolean operator. The conventions above are explicit
  implementation choices, not a claim to reproduce Microsoft's private extractor.
  Boolean mode can be changed using `boolean_mode="and"`; an empty query scores
  zero in either mode.
- The single IDF feature sums **all query terms**, so it is constant across
  documents for a given query. It is retained as requested, although it cannot
  distinguish documents under a linear ranking function.
- Unseen query terms have no finite unsmoothed corpus IDF. They receive weight
  zero, while remaining in coverage and summary denominators. This avoids
  inventing smoothing and keeps unknown test queries finite. No corpus documents
  can match such a term. This is an explicit edge-case convention.

## Handoff and verification

Use `FEATURE_NAMES` as the authoritative order. IDs, labels and split membership
are metadata, not additional features. Keep original grades and special `-1`
judgments separate when your matrix/labeling code joins Part 1 outputs. This
package never reads relevance judgments or split assignments.

Run from the repository root:

```bash
python3 -m pytest feature_extraction/tests -q
```

Tests independently calculate all 22 expected values on a tiny corpus, including
nonmatching query terms and a document term outside the query, then check empty
inputs, optional scoring conventions and compatibility with Part 1's files.
