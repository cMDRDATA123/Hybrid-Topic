"""Coverage-explicit reporting for new experiments; legacy metrics stay intact.

This module never selects a model. Call it only after predictions are frozen.
HMP/ARI/NMI on all documents treat -1 as a shared partition, as legacy evaluate
does. That partition is not a discovered topic or a correct rejection label.
Assigned-only scores are conditional diagnostics with differing sample sets.
"""

from __future__ import annotations

import numpy as np

from hybrid_topic.evaluation import evaluate


REPORT_VERSION = "coverage-explicit-external-v1"


def build_evaluation_report(true_ids, pred_ids) -> dict:
    """Return JSON-safe counts and unchanged metrics under two named scopes.

    IDs must be integers; callers must encode categorical gold labels explicitly.
    Predicted IDs are nonnegative topics or -1. An empty assigned subset has null
    metrics; sklearn's singleton conventions are retained and flagged.
    """
    true, pred = np.asarray(true_ids), np.asarray(pred_ids)
    if true.ndim != 1 or pred.ndim != 1 or len(true) != len(pred) or not len(true):
        raise ValueError("evaluation requires nonempty aligned one-dimensional IDs")
    if true.dtype.kind not in "iu" or pred.dtype.kind not in "iu":
        raise ValueError("IDs must be integers, not floats, strings, or booleans")
    if any(isinstance(value, (bool, np.bool_))
           for values in (true_ids, pred_ids) for value in values):
        raise ValueError("boolean IDs must not be coerced to integer labels")
    if pred.dtype.kind == "i" and np.any(pred < -1):
        raise ValueError("predicted IDs must be nonnegative or -1")
    assigned = pred != -1
    n_assigned = int(assigned.sum())

    def metrics(gold, predicted):
        return {key: float(value) for key, value in evaluate(gold, predicted).items()
                if key != "n_predicted_topics"}

    warnings = []
    if n_assigned < len(pred):
        warnings.append("rejection_is_a_partition_in_all_document_metrics")
    if not n_assigned:
        warnings.append("no_assigned_documents")
    if n_assigned == 1:
        warnings.append("single_assigned_document")
    if n_assigned and len(np.unique(true[assigned])) == 1:
        warnings.append("single_assigned_gold_class")
    if len(np.unique(true)) == 1:
        warnings.append("single_gold_class")
    return {
        "report_version": REPORT_VERSION,
        "rejection_label": -1,
        "n_documents": len(pred),
        "n_assigned": n_assigned,
        "n_rejected": len(pred) - n_assigned,
        "coverage": n_assigned / len(pred),
        "n_assigned_topics": len(np.unique(pred[assigned])),
        "n_partitions_including_rejection": len(np.unique(pred)),
        "all_documents": {
            "n_documents": len(pred),
            "rejection_policy": "shared_partition_including_minus_one",
            "metrics": metrics(true, pred),
        },
        "assigned_documents": {
            "n_documents": n_assigned,
            "rejection_policy": "exclude_minus_one_conditional_diagnostic",
            "metrics": metrics(true[assigned], pred[assigned]) if n_assigned else None,
        },
        "mapped_accuracy_semantics": "Hungarian mapping fitted on evaluation gold; not deployment accuracy",
        "warnings": warnings,
    }
