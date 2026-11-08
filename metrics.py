"""Metric helpers shared by train.py and app.py (binary classification: phishing vs legitimate)."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

POSITIVE_LABEL = "phishing"
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def compute_metrics(y_true, y_pred, phishing_proba):
    """Return the six assignment metrics; 'phishing' is treated as the positive class.

    AUC is NaN when y_true contains only one class (it is undefined in that case).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    binary = {"pos_label": POSITIVE_LABEL, "zero_division": 0}

    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true == POSITIVE_LABEL, phishing_proba)
    else:
        auc = float("nan")

    scores = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, **binary),
        "Recall": recall_score(y_true, y_pred, **binary),
        "F1": f1_score(y_true, y_pred, **binary),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    return {name: round(float(value), 4) for name, value in scores.items()}
