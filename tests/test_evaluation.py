"""
Tests for evaluation metrics: accuracy, precision, recall, F1, and confusion matrix.
"""

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def test_metric_calculations_consistency():
    y_true = ["yes", "yes", "no", "stop", "go", "yes", "stop", "no", "up", "down"]
    y_pred = ["yes", "no",  "no", "stop", "go", "yes", "go",   "no", "up", "down"]
    classes = ["down", "go", "no", "stop", "up", "yes"]

    acc = accuracy_score(y_true, y_pred)
    assert 0.0 <= acc <= 1.0
    assert np.isclose(acc, 0.8)

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    assert 0.0 <= f1 <= 1.0

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    assert cm.shape == (len(classes), len(classes))
    assert np.sum(cm) == len(y_true)
