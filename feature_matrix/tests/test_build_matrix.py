import numpy as np
import pytest

from feature_extraction import FEATURE_NAMES, FeatureExtractor
from feature_matrix.build_matrix import build_split, sample_negatives, read_grades, read_special, read_split_ids, PROCESSED


@pytest.fixture
def tiny():
    docs = {1: ["heat", "flow"], 2: ["wing", "flow", "flow"], 3: ["shock"]}
    queries = {10: ["flow"], 11: ["heat"]}
    grades = {(10, 2): 1, (10, 1): 4, (11, 1): 2}
    special = {(11, 3)}
    return FeatureExtractor(docs), queries, grades, special, [1, 2, 3]


def test_shape_labels_and_special_dropped(tiny):
    ext, queries, grades, special, docs = tiny
    X, y, g, qs, ds = build_split(ext, queries, [10, 11], grades, special, 4, docs)
    assert X.shape == (5, 22)  # 2*3 pairs minus the (11,3) special pair
    assert (11, 3) not in set(zip(qs.tolist(), ds.tolist()))
    assert dict(zip(zip(qs.tolist(), ds.tolist()), y.tolist()))[(10, 2)] == 1
    assert y.sum() == 3 and g.max() == 4


def test_threshold_changes_labels_not_grades(tiny):
    ext, queries, grades, special, docs = tiny
    _, y4, g4, *_ = build_split(ext, queries, [10, 11], grades, special, 4, docs)
    _, y3, g3, *_ = build_split(ext, queries, [10, 11], grades, special, 3, docs)
    assert y3.sum() == y4.sum() - 1 and (g3 == g4).all()


def test_row_matches_extractor(tiny):
    ext, queries, grades, special, docs = tiny
    X, _, _, qs, ds = build_split(ext, queries, [10], grades, special, 4, docs)
    assert tuple(X[1]) == ext.extract_vector(queries[10], 2)


def test_sampling_keeps_all_positives(tiny):
    y = np.array([1, 0, 0, 0, 0, 1, 0, 0])
    q = np.array([1, 1, 1, 1, 1, 2, 2, 2])
    keep = sample_negatives(y, q, 1, seed=0)
    assert keep[y == 1].all()
    assert keep[(q == 1) & (y == 0)].sum() == 1 and keep[(q == 2) & (y == 0)].sum() == 1


def test_real_data_consistency():
    grades, special = read_grades(), read_special()
    assert not (set(grades) & special)  # -1 pairs never appear as relevant
    ids = [set(read_split_ids(s)) for s in ("train", "val", "test")]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert len(FEATURE_NAMES) == 22
