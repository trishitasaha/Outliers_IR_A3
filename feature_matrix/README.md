# Part 3 (b): Feature Matrix and Labels

This module takes the 22 query–document features produced by
[`feature_extraction/`](../feature_extraction/README.md) and the relevance
judgments prepared in [`preprocessing/`](../preprocessing/README.md), and turns
them into the **feature matrix `X` and label vectors `y`** used to train and
evaluate the linear SVM in Part 4 (Method B, Learning to Rank).

It does **not** extract features and does **not** train any model.

---

## 1. Where it fits in the pipeline

```
preprocessing/            feature_extraction/         feature_matrix/  (this module)       SVM (Part 4)
 cleaned tokens   ->   22 features per    ->   X, y for train / val / test  -> linear SVM
 qrels, splits             (query, document)           + scaler                            ranking
```

| Input (from other parts) | Used for |
|---|---|
| `FeatureExtractor`, `load_query_tokens`, `FEATURE_NAMES` (`feature_extraction/`) | The 22 feature values |
| `preprocessing/data/processed/relevance.csv` | Relevance grades (1–4) ->abels |
| `preprocessing/data/processed/relevance_special.csv` | Undocumented `-1` pairs ->xcluded |
| `preprocessing/data/processed/{train,val,test}_queries.csv` | The shared 157 / 34 / 34 query split |

---

## 2. How to run

From the **repository root**:

```bash
pip install numpy pytest
python3 -m feature_matrix.build_matrix          
python3 -m pytest feature_matrix feature_extraction -q   
```

Optional arguments:

| Argument | Default | Meaning |
|---|---|---|
| `--relevant-max-grade N` | `4` | Grades 1..N count as relevant (Part 1 uses 1–4) |
| `--neg-ratio K` | `20` | Negatives kept per positive in `train_sampled.npz` |
| `--seed S` | `42` | Seed for negative sampling |
| `--processed-dir PATH` | Part 1 output folder | Location of the processed CSVs |
| `--output-dir PATH` | `feature_matrix/output` | Where the matrices are written |

Requires Python 3.9+ and `numpy`.

---

## 3. What is built

One row per **(query, document) pair**. Every query in a split is paired with
**all 1,400 documents**, so the full ranked list can be evaluated later. Rows are
ordered by `query_id`, then `doc_id`. Columns follow `FEATURE_NAMES`:

`query_term_number, query_term_ratio, stream_length, idf, tf_sum, tf_min, tf_max,
tf_mean, tf_variance, normalized_tf_{sum,min,max,mean,variance},
tf_idf_{sum,min,max,mean,variance}, boolean_model, vector_space_model, bm25`

### Result

| Split | Queries | Rows (pairs) | Positive pairs | Positive rate |
|---|---|---|---|---|
| train | 157 | 219,643 | 1,144 | 0.52 % |
| val | 34 | 47,566 | 230 | 0.48 % |
| test | 34 | 47,566 | 238 | 0.50 % |
| train_sampled | 157 | 24,024 | 1,144 | 4.8 % |

Row count check: 157 × 1400 − 157 dropped `-1` pairs = 219,643.
Positive check: 1,144 + 230 + 238 = 1,612 = number of rows in `relevance.csv`.

---

## 4. Labeling decisions

1. **Positive label.** `label = 1` if the pair appears in `relevance.csv` with
   grade 1–4 (Cranfield: 1 = complete answer … 4 = minimum interest). The
   threshold is configurable with `--relevant-max-grade`.
2. **Negative label.** `label = 0` for every other document. Cranfield lists only
   relevant pairs, so these are *unjudged* documents treated as non-relevant. This
   is the standard assumption for learning to rank on Cranfield, and it is an
   assumption, not ground truth.
3. **Undocumented `-1` code.** The 225 pairs with code `-1` have an unknown
   meaning, so they are **dropped** (neither positive nor negative).
4. **Original grades kept.** The `grade` array stores 1–4 for judged pairs and 0
   for unjudged ones, in case graded metrics are wanted.
5. **Class imbalance.** About 0.5 % of rows are positive. Part 4 can either use
   `class_weight="balanced"` on the full train set, or train on
   `train_sampled.npz` (all positives plus up to 20 random negatives per
   positive, drawn separately for each query). Validation and test always stay
   **complete** so that ranking evaluation is unbiased.
6. **No leakage.** The standardization statistics (`scaler.json`) are computed on
   the **train split only** and are then applied to val and test. Query splits
   are exactly those from Part 1; no query appears in more than one split.

---

## 5. Output files (`feature_matrix/output/`)

| File | Content |
|---|---|
| `train.npz`, `val.npz`, `test.npz` | `X` (n × 22), `y` (n), `grade` (n), `query_ids` (n), `doc_ids` (n) |
| `train_sampled.npz` | Train rows with all positives and sampled negatives |
| `scaler.json` | Per-feature `mean` and `std` from the train split |
| `matrix_summary.json` | Row counts, positive counts, settings used |
| `{train,val,test}_features.csv` | Human-readable copy (`query_id, doc_id, 22 features, relevance_grade, label`). Git-ignored because of size (about 72 MB); regenerate with the command above |

---

## 6. Using the matrices in Part 4

```python
from feature_matrix.loader import load_split, load_scaler
from sklearn.svm import LinearSVC
import numpy as np

Xtr, ytr, _        = load_split("train_sampled")   # or "train"
Xva, yva, info_va  = load_split("val")             # use "test" only for the final evaluation
sc = load_scaler()

model = LinearSVC(C=1.0, class_weight="balanced", max_iter=20000)
model.fit(sc.transform(Xtr), ytr)

scores = model.decision_function(sc.transform(Xva))
for qid in np.unique(info_va["query_ids"]):
    mask = info_va["query_ids"] == qid
    ranked_docs = info_va["doc_ids"][mask][np.argsort(-scores[mask])]
```

`model.coef_` gives the learned real-valued weights in `FEATURE_NAMES` order.

---

## 7. Sanity checks performed

- All 219,643 + 47,566 + 47,566 feature rows are finite (no NaN or infinity), and
  no feature column is constant.
- Ranking train queries by a single feature gives mean precision@10 of about
  0.23 for `bm25`, 0.22 for `vector_space_model`, 0.21 for `tf_idf_sum`, and
  only 0.015 for `boolean_model`. This confirms that labels and features are
  aligned (the better-matching features rank relevant documents higher).
- A quick linear SVM trained on `train_sampled` and applied to `val` gave
  precision@10 of about 0.24. This only checks that the matrix is usable; it is
  not the final Part 4 result.
- Tests (`feature_matrix/tests/`) check the matrix shape, that `-1` pairs are
  dropped, the effect of the relevance threshold, that rows equal the extractor's
  output, that negative sampling keeps every positive, and that the three query
  splits do not overlap.

---

## 8. Known limitations

- **Unjudged documents as negatives.** Some unjudged documents may actually be
  relevant. This slightly understates precision for every method alike.
- **The `idf` feature is constant for a given query** (it sums IDF over all query
  terms), so it cannot distinguish documents of the same query. It is kept
  because the assignment requires it; expect the SVM to give it little weight.
- **Boolean model feature** uses OR matching (the extractor default), so it is
  1 for most documents and is a weak signal.
- Features in the files are **unscaled**; apply `scaler.json` (or your own
  scaler fitted on train) before training.

---

## 9. Files in this folder

| File | Purpose |
|---|---|
| `build_matrix.py` | Builds all matrices (run as a module) |
| `loader.py` | `load_split()` and `load_scaler()` helpers for Part 4 |
| `tests/test_build_matrix.py` | Unit tests |
| `output/` | Generated matrices, scaler and summary |
| `.gitignore` | Excludes the large CSV copies |
| `__init__.py` | Makes the folder an importable package |

*Author: Part 3 (matrix and labels) — see the main repository README for the group.*
