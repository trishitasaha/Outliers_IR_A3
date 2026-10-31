
import json

FIELDS = [
    "title",
    "author",
    "bibliography",
    "body"
]


def parse_tokens(tokens):

    if isinstance(tokens, str):
        tokens = json.loads(tokens)

    return {
        str(token).lower()
        for token in tokens
    }


def boolean_field_match(
    query_tokens,
    field_tokens,
    min_matches=1
):

    query_set = parse_tokens(query_tokens)
    field_set = parse_tokens(field_tokens)

    matched_terms = query_set & field_set

    return int(
        len(matched_terms) >= min_matches
    )


def get_boolean_scores(
    query_tokens,
    document,
    min_matches=1
):

    query_set = parse_tokens(query_tokens)

    scores = {}

    for field in FIELDS:

        field_tokens = parse_tokens(
            document[field + "_stemmed_tokens"]
        )

        scores[field] = boolean_field_match(
            query_set,
            field_tokens,
            min_matches
        )

    return scores
