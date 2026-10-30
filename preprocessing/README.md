# Cranfield IR Project — Task 1: Dataset Preparation

## A. Project Overview

Our group compares two ranking methods on the Cranfield collection:

1. **Boolean Weighted Zone Scoring**: query terms are matched in the Title, Author, Bibliography and Body zones, and the zone weights are tuned by grid search (Task 2).
2. **Linear SVM ranking**: 22 query–document features (Task 3) feed a linear SVM that ranks documents by its decision score (Task 4).

Task 1 (this project) prepares the data for both methods. It parses the raw files, cleans and tokenizes the text, organizes the relevance judgments, creates one shared train/validation/test split of the queries, and validates everything. It does no scoring, feature extraction or model training.

## B. Dataset Description

The four original files are in `data/raw/` and are never modified:

| File | Content |
|---|---|
| `cran.all.1400` | 1,400 aeronautics abstracts |
| `cran.qry` | 225 queries (IDs `001`…`365`, not consecutive) |
| `cranqrel` | 1,837 relevance judgments: `query_id  doc_id  relevance_code` |
| `cranqrel.readme` | Meaning of the relevance codes 1–5 |

Each document has an ID (`.I`) and four fields, which are kept separate throughout:

| Marker | Field | Example |
|---|---|---|
| `.T` | Title | *experimental investigation of the aerodynamics of a wing in a slipstream* |
| `.A` | Author | *brenckman,m.* |
| `.B` | Bibliography | *j. ae. scs. 25, 1958, 324.* |
| `.W` | Body (abstract) | the abstract text. It usually starts by repeating the title |

## C. Installation and Execution

Run these from inside the `cranfield_ir_project` folder:

```bash
pip install -r requirements.txt
python main.py            # on macOS you can also use: python3 main.py
```

That one command runs everything: parsing, preprocessing, splitting and validation. It takes a few seconds. All outputs are written to `data/processed/`, which is created automatically, and running it again simply overwrites them. A summary is printed at the end, and the run should say **Validation: PASS**.

**Optional: use a virtual environment** (recommended, so the packages don't affect your other projects):

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

**Optional settings.** You do not need these for a normal run.

```bash
python main.py --relevant-max-grade 3   # only grades 1-3 count as relevant (default: 1-4)
python main.py --split-hyphens          # split "boundary-layer" into "boundary", "layer"
python main.py --seed 42                # random seed for the split (default 42)
```

**Run the tests:** `python -m pytest`

Requires Python 3.9 or newer, pandas and nltk. Nothing else needs to be downloaded.

## D. Preprocessing Performed

- **Document and query parsing.** Records are split at `.I`, and each field's text runs until the next marker. Multi-line text is joined into one line. Empty fields are kept as empty strings, and no document or query is ever dropped.
- **Query ID mapping.** `cran.qry` uses IDs like `001, 002, 004, 008…`, but `cranqrel` numbers the queries `1…225`. The k-th query in `cran.qry` is query k in `cranqrel`. Every file therefore has `query_id` (1–225, **use this for all joins**) and `original_query_id` (the file's label, e.g. `"004"`).
- **Normalization.** Unicode NFKC normalization, lower-casing, control characters removed, and repeated whitespace collapsed.
- **Tokenization.** Text is split into words and numbers; other punctuation is removed. Scientific terms are kept whole: `boundary-layer`, `x-15`, `0.1`, `15,000`. Numbers and duplicate words are kept, so term frequencies stay correct.
- **Stopword removal.** A fixed English stopword list (179 words, the NLTK list) is applied, but stored as a separate representation.
- **Porter stemming.** NLTK Porter stemmer, applied to the stopword-free tokens (e.g. `flows` → `flow`).
- **Relevance judgments.** The original grade is kept, and a `binary_relevance` label is added (default: grades 1–4 = relevant). Rows with the undocumented code `-1` go into a separate file.
- **Splitting.** Queries are split 70/15/15 into **157 train / 34 validation / 34 test**, with seed 42. Each query is in exactly one split, and the documents are not split, so every split searches all 1,400 documents.

**What is preserved:** the original text (`documents.csv`, `queries.csv`), and for **each field separately** four forms: clean text, tokens, tokens without stopwords, and stemmed tokens. Tasks 2 and 3 can therefore pick the representation they need without re-processing anything.

## E. Generated Output Files

All files are in `data/processed/`. Token columns contain JSON lists, for example `["heat", "transfer"]`.

| File | Main columns | Purpose | Needed by |
|---|---|---|---|
| `documents_processed.csv` | `doc_id`; for each of `title`, `author`, `bibliography`, `body`: `_clean`, `_tokens`, `_tokens_no_stop`, `_stemmed_tokens` | Preprocessed text, one set of columns per field | Tasks 2, 3 |
| `queries_processed.csv` | `query_id`, `original_query_id`, `query_clean`, `query_tokens`, `query_tokens_no_stop`, `query_stemmed_tokens` | Preprocessed queries | Tasks 2, 3 |
| `relevance.csv` | `query_id`, `doc_id`, `relevance_grade`, `binary_relevance` | Relevance labels (grades 1–5) | Tasks 2, 3, 4 |
| `train_queries.csv` | `query_id`, `original_query_id`, `query_text` | Training queries (157) | Tasks 2, 3, 4 |
| `val_queries.csv` | same as above | Validation queries (34): tuning and model selection | Tasks 2, 3, 4 |
| `test_queries.csv` | same as above | Test queries (34): final evaluation only | Tasks 2, 3, 4 |
| `documents.csv` | `doc_id`, `title`, `author`, `bibliography`, `body` | Original document text | Reference |
| `queries.csv` | `query_id`, `original_query_id`, `query_text` | Original queries and the ID mapping | Reference |
| `relevance_special.csv` | `query_id`, `doc_id`, `original_relevance_grade`, `reason` | The 225 rows with code `-1` | Reference |
| `dataset_statistics.json` | — | Counts, lengths, vocabulary size, split summary | Reference |
| `validation_report.txt` | — | PASS/WARN/FAIL result of every data check | Everyone |

Load these files with `load_output()` from `src/config.py` rather than plain `pd.read_csv`. It keeps empty fields as `""`, keeps leading zeros in `original_query_id`, and can turn token columns into Python lists.

## G. Important Assumptions

- **Query ID mapping.** The files never state that the k-th query in `cran.qry` is query k in `cranqrel`. We infer it from record order. The data supports it strongly: queries share far more words with their relevant documents under this mapping than under any alternative (details in `validation_report.txt`).
- **Grade -1.** `cranqrel` contains 225 rows (one per query) with code `-1`, which `cranqrel.readme` does not define. Their meaning is unknown, so they are kept in `relevance_special.csv` and are **not** used as relevant or non-relevant examples.
- **Relevance threshold.** By default grades 1–4 are relevant. Lower grades mean *more* relevant (1 = complete answer, 4 = minimum interest). Use `--relevant-max-grade 3` if the group decides only grades 1–3 count. The original grades are always kept.
- **No negative examples.** Cranfield contains no grade-5 ("no interest") rows, so `relevance.csv` holds only relevant pairs. Documents not listed for a query are **unjudged**, not known to be irrelevant. Treating them as negatives for the SVM is a modelling decision for Task 3 to make and document.
- **Raw-data irregularities are kept, not repaired.** Two documents are empty, and a few records have formatting errors. All of them are listed as warnings in `validation_report.txt`.
- **Fixed random seed and reproducibility.** The split uses seed 42 and is stratified by the number of judgments per query. It does not change with the relevance threshold. Running `python main.py` again on any computer produces identical files, and validation checks this.
