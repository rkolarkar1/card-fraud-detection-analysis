"""
Shared evaluation helper. Used by train_model.py and ablation.py so the
best-F1-threshold calculation stays identical wherever a trained model is
evaluated.
"""
from sklearn.metrics import precision_recall_curve


def best_f1_threshold(y_true, y_proba) -> dict:
    """Finds the classification threshold that maximizes F1 on the given
    precision-recall curve. Returns the threshold, its F1/precision/recall,
    and the raw curve arrays (precisions, recalls, thresholds, f1_scores,
    best_idx) for callers that also want to plot the curve."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best_idx = f1_scores[:-1].argmax()

    return {
        "threshold": float(thresholds[best_idx]),
        "f1": float(f1_scores[best_idx]),
        "precision": float(precisions[best_idx]),
        "recall": float(recalls[best_idx]),
        "precisions": precisions,
        "recalls": recalls,
        "thresholds": thresholds,
        "f1_scores": f1_scores,
        "best_idx": best_idx,
    }
