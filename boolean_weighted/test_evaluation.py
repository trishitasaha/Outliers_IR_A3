
from evaluation import calculate_precision_recall_f1


retrieved_docs = [1, 2, 3, 4, 5]

relevant_docs = {2, 4, 6, 8}

precision, recall, f1 = calculate_precision_recall_f1(
    retrieved_docs,
    relevant_docs
)

print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)
