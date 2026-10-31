
from boolean_matching import boolean_field_match

query = ["aeroelastic", "aircraft", "heat", "high", "speed"]

field_1 = ["heat"]
field_2 = ["aircraft", "heat"]
field_3 = ["aeroelastic", "aircraft", "heat", "high", "speed"]

print("OR rule:")
print(boolean_field_match(query, field_1, min_matches=1))
print(boolean_field_match(query, field_2, min_matches=1))
print(boolean_field_match(query, field_3, min_matches=1))

print("\nMinimum 2 matches:")
print(boolean_field_match(query, field_1, min_matches=2))
print(boolean_field_match(query, field_2, min_matches=2))
print(boolean_field_match(query, field_3, min_matches=2))

print("\nMinimum 3 matches:")
print(boolean_field_match(query, field_1, min_matches=3))
print(boolean_field_match(query, field_2, min_matches=3))
print(boolean_field_match(query, field_3, min_matches=3))
