from __future__ import annotations
import argparse
import csv
import json
import random
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature_extraction import FEATURE_NAMES, FeatureExtractor, load_query_tokens  

PROCESSED = ROOT / "preprocessing" / "data" / "processed"
OUTPUT = Path(__file__).resolve().parent / "output"
SPLITS = ("train", "val", "test")

def read_split_ids(split: str, processed: Path = PROCESSED) -> list[int]:
    with (processed / f"{split}_queries.csv").open(newline="", encoding="utf-8") as f:
        return sorted(int(r["query_id"]) for r in csv.DictReader(f))

def read_grades(processed: Path = PROCESSED) -> dict[tuple[int, int], int]:
    grades = {}
    with (processed / "relevance.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            grades[(int(r["query_id"]), int(r["doc_id"]))] = int(r["relevance_grade"])
    return grades

def read_special(processed: Path = PROCESSED) -> set[tuple[int, int]]:
    with (processed / "relevance_special.csv").open(newline="", encoding="utf-8") as f:
        return {(int(r["query_id"]), int(r["doc_id"])) for r in csv.DictReader(f)}

def build_split(extractor, queries, query_ids, grades, special, max_grade, doc_ids):
    X, y, g, qs, ds = [], [], [], [], []
    for qid in query_ids:
        tokens = queries[qid]
        for did in doc_ids:
            if (qid, did) in special:
                continue
            grade = grades.get((qid, did), 0)
            X.append(extractor.extract_vector(tokens, did))
            y.append(1 if 0 < grade <= max_grade else 0)
            g.append(grade)
            qs.append(qid)
            ds.append(did)
    return (
        np.asarray(X, dtype=np.float64),
        np.asarray(y, dtype=np.int8),
        np.asarray(g, dtype=np.int8),
        np.asarray(qs, dtype=np.int32),
        np.asarray(ds, dtype=np.int32),
    )


def sample_negatives(y, query_ids, neg_ratio, seed):
    rng = random.Random(seed)
    keep = np.zeros(len(y), dtype=bool)
    for qid in np.unique(query_ids):
        rows = np.flatnonzero(query_ids == qid)
        pos = rows[y[rows] == 1]
        neg = rows[y[rows] == 0]
        keep[pos] = True
        n = min(len(neg), max(1, neg_ratio * max(1, len(pos))))
        keep[rng.sample(list(neg), n)] = True
    return keep

def write_csv(path, X, y, g, qs, ds):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "doc_id", *FEATURE_NAMES, "relevance_grade", "label"])
        for i in range(len(y)):
            w.writerow([qs[i], ds[i], *(repr(float(v)) for v in X[i]), g[i], y[i]])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed-dir", type=Path, default=PROCESSED)
    p.add_argument("--output-dir", type=Path, default=OUTPUT)
    p.add_argument("--relevant-max-grade", type=int, default=4,
                   help="grades 1..N count as relevant (default 4, same as Part 1)")
    p.add_argument("--neg-ratio", type=int, default=20,
                   help="negatives kept per positive in train_sampled.npz (default 20)")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    print("Loading corpus and fitting extractor ...")
    extractor = FeatureExtractor.from_processed(a.processed_dir)
    queries = load_query_tokens(a.processed_dir)
    grades = read_grades(a.processed_dir)
    special = read_special(a.processed_dir)
    doc_ids = list(extractor.document_ids)

    summary = {"relevant_max_grade": a.relevant_max_grade, "neg_ratio": a.neg_ratio,
               "seed": a.seed, "feature_names": list(FEATURE_NAMES), "splits": {}}
    data = {}
    for split in SPLITS:
        ids = read_split_ids(split, a.processed_dir)
        print(f"Building {split}: {len(ids)} queries x {len(doc_ids)} documents ...")
        X, y, g, qs, ds = build_split(extractor, queries, ids, grades, special,
                                      a.relevant_max_grade, doc_ids)
        data[split] = (X, y, g, qs, ds)
        np.savez_compressed(a.output_dir / f"{split}.npz", X=X, y=y, grade=g, query_ids=qs, doc_ids=ds)
        write_csv(a.output_dir / f"{split}_features.csv", X, y, g, qs, ds)
        summary["splits"][split] = {
            "queries": len(ids), "rows": int(len(y)), "positives": int(y.sum()),
            "positive_rate": float(y.mean()),
            "queries_without_positive": int(sum(1 for q in ids if y[qs == q].sum() == 0)),
        }

    Xtr = data["train"][0]
    mean, std = Xtr.mean(axis=0), Xtr.std(axis=0)
    std_safe = np.where(std > 0, std, 1.0)  # constant columns stay zero after scaling
    (a.output_dir / "scaler.json").write_text(json.dumps(
        {"feature_names": list(FEATURE_NAMES), "mean": mean.tolist(), "std": std_safe.tolist()}, indent=2))

    X, y, g, qs, ds = data["train"]
    keep = sample_negatives(y, qs, a.neg_ratio, a.seed)
    np.savez_compressed(a.output_dir / "train_sampled.npz",
                        X=X[keep], y=y[keep], grade=g[keep], query_ids=qs[keep], doc_ids=ds[keep])
    summary["train_sampled"] = {"rows": int(keep.sum()), "positives": int(y[keep].sum())}

    (a.output_dir / "matrix_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["splits"], indent=2))
    print("train_sampled:", summary["train_sampled"])
    print(f"Written to {a.output_dir}")


if __name__ == "__main__":
    main()
