from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)


def evaluate(y_true, predictions, probabilities):
    return {
        "metrics": {
            "ROC-AUC": float(roc_auc_score(y_true, probabilities)),
            "Accuracy": float(accuracy_score(y_true, predictions)),
            "Precision": float(precision_score(y_true, predictions, zero_division=0)),
            "Recall": float(recall_score(y_true, predictions, zero_division=0)),
            "F1": float(f1_score(y_true, predictions, zero_division=0)),
        },
        "report": classification_report(y_true, predictions, digits=4, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
    }
