import json

FIELDS = ["title", "author", "bibliography", "body"]


def parse_tokens(tokens):
    """Accept tokens stored either as a JSON string or a Python list."""
    if isinstance(tokens, str):
        tokens = json.loads(tokens)
    return [str(token).lower() for token in tokens]


def soft_field_match(query_tokens, field_tokens):
    """Return the fraction of unique query terms found in one field."""
    query_set = set(query_tokens)
    if not query_set:
        return 0.0

    field_set = set(field_tokens)
    return len(query_set & field_set) / len(query_set)


def get_soft_scores(query_tokens, document):
    """Return query-term coverage for each of the four document fields."""
    query_tokens = parse_tokens(query_tokens)
    scores = {}

    for field in FIELDS:
        field_tokens = parse_tokens(document[field + "_stemmed_tokens"])
        scores[field] = soft_field_match(query_tokens, field_tokens)

    return scores