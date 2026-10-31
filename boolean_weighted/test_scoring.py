
from scoring import calculate_score, score_document


boolean_scores = {
    "title": 1,
    "author": 0,
    "bibliography": 1,
    "body": 1
}

weights = {
    "title": 4,
    "author": 1,
    "bibliography": 2,
    "body": 1
}


score = calculate_score(
    boolean_scores,
    weights
)

print("Weighted score:", score)


query_tokens = [
    "aerodynam",
    "wing",
    "aircraft"
]

document = {
    "title_stemmed_tokens":
        '["aerodynam", "wing", "slipstream"]',

    "author_stemmed_tokens":
        '["brenckman"]',

    "bibliography_stemmed_tokens":
        '["aero", "aircraft", "1958"]',

    "body_stemmed_tokens":
        '["wing", "slipstream", "velocity", "aircraft"]'
}


score = score_document(
    query_tokens,
    document,
    weights
)

print("Document score:", score)
