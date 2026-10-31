
import json


FIELDS = [
    "title",
    "author",
    "bibliography",
    "body"
]


def parse_tokens(tokens):
    """
    Convert tokens stored as a JSON string or list
    into a Python list of lowercase tokens.
    """

    if isinstance(tokens, str):
        tokens = json.loads(tokens)

    return [token.lower() for token in tokens]


def boolean_field_match(query_tokens, field_tokens):
    """
    OR-based Boolean matching.

    Returns 1 if at least one query term
    appears in the document field.
    Otherwise, returns 0.
    """

    query_set = set(query_tokens)
    field_set = set(field_tokens)

    return int(bool(query_set & field_set))


def get_boolean_scores(query_tokens, document):
    """
    Calculate Boolean matching scores
    for all four document fields.

    document is a dictionary containing
    the four stemmed token fields.
    """

    query_tokens = parse_tokens(query_tokens)

    scores = {}

    for field in FIELDS:

        field_tokens = parse_tokens(
            document[field + "_stemmed_tokens"]
        )

        scores[field] = boolean_field_match(
            query_tokens,
            field_tokens
        )

    return scores
