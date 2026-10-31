import math

import pytest

from feature_extraction import FEATURE_NAMES, FeatureExtractor, load_query_tokens


def test_all_22_features_against_small_corpus():
    extractor = FeatureExtractor({
        1: ["apple", "apple", "banana", "extra"],
        2: ["banana"], 3: ["banana"], 4: [],
    })
    # Distinct query terms: apple, banana, unseen. Repetition affects cosine,
    # but not the coverage denominator or the TF summary statistics.
    result = extractor.extract(["apple", "apple", "banana", "unseen"], 1)
    a, b = math.log10(4), math.log10(4 / 3)
    x, y = (1 + math.log10(2)) * a, b
    mean = (x + y) / 3
    expected = [
        2, 2 / 3, 4, a + b,
        3, 0, 2, 1, 2 / 3,
        .75, 0, .5, .25, 1 / 24,
        x + y, 0, x, mean, ((x - mean)**2 + (y - mean)**2 + mean**2) / 3,
        1,
        math.sqrt(x*x + y*y) / math.sqrt(x*x + y*y + a*a),
        a * 2.2 * 2 / (1.2 * (.25 + .75 * 4 / 1.5) + 2)
        + b * 2.2 / (1.2 * (.25 + .75 * 4 / 1.5) + 1),
    ]
    assert tuple(result) == FEATURE_NAMES
    assert len(result) == 22
    assert tuple(result.values()) == pytest.approx(expected)
    assert extractor.extract_vector(["apple", "apple", "banana", "unseen"], 1) == tuple(result.values())


def test_empty_document_query_and_unknown_vocabulary():
    extractor = FeatureExtractor({1: ["a"], 2: []})
    empty_document = extractor.extract(["a"], 2)
    assert empty_document["idf"] == pytest.approx(math.log10(2))
    assert all(v == 0 for k, v in empty_document.items() if k != "idf")
    empty_query = extractor.extract([], 1)
    assert empty_query["stream_length"] == 1
    assert all(v == 0 for k, v in empty_query.items() if k != "stream_length")
    unseen = extractor.extract(["unseen"], 1)
    assert all(v == 0 for k, v in unseen.items() if k != "stream_length")
    all_empty = FeatureExtractor({1: [], 2: []}).extract(["a"], 1)
    assert all(v == 0 for v in all_empty.values())


def test_boolean_options_and_optional_query_weighting():
    docs = {1: ["a", "b"], 2: ["b"]}
    assert FeatureExtractor(docs).extract(["a", "missing"], 1)["boolean_model"] == 1
    strict = FeatureExtractor(docs, boolean_mode="and")
    assert strict.extract(["a", "missing"], 1)["boolean_model"] == 0
    assert strict.extract(["a", "b"], 1)["boolean_model"] == 1
    assert strict.extract([], 1)["boolean_model"] == 0
    plain = FeatureExtractor(docs).extract(["a", "a"], 1)["bm25"]
    weighted = FeatureExtractor(docs, k3=1.2).extract(["a", "a"], 1)["bm25"]
    assert weighted == pytest.approx(plain * 2.2 * 2 / 3.2)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        FeatureExtractor({})
    for kwargs in ({"k1": 0}, {"b": 2}, {"k3": -1}, {"boolean_mode": "xor"}):
        with pytest.raises(ValueError):
            FeatureExtractor({1: ["a"]}, **kwargs)
    with pytest.raises(TypeError):
        FeatureExtractor({1: "raw text"})
    extractor = FeatureExtractor({1: ["a"]})
    with pytest.raises(KeyError):
        extractor.extract(["a"], 999)


def test_part1_loading_and_representative_pairs():
    extractor = FeatureExtractor.from_processed()
    queries = load_query_tokens()
    assert extractor.num_documents == 1400
    assert len(queries) == 225
    for qid in (1, 125, 225):
        for did in (1, 471, 995, 1400):
            result = extractor.extract(queries[qid], did)
            assert tuple(result) == FEATURE_NAMES
            assert all(math.isfinite(v) for v in result.values())
    assert extractor.extract(queries[1], 471)["stream_length"] == 0
