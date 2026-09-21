"""Text normalization and held-out clustering evaluation utilities."""

from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def clean_text(text: str, max_chars: int | None = None) -> str:
    """Normalize whitespace and remove common quoted-message headers."""

    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(">"):
            continue
        if re.match(r"^(from|subject|organization|lines|nntp-posting-host):", stripped, flags=re.I):
            continue
        lines.append(stripped)
    cleaned = re.sub(r"\s+", " ", " ".join(lines)).strip()
    if max_chars is not None and len(cleaned) > max_chars:
        shortened = cleaned[:max_chars]
        return shortened.rsplit(" ", 1)[0] if " " in shortened else shortened
    return cleaned


def _validated_ids(true_ids: np.ndarray, pred_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(true_ids)
    pred = np.asarray(pred_ids)
    if true.ndim != 1 or pred.ndim != 1:
        raise ValueError("true_ids and pred_ids must be one-dimensional")
    if len(true) != len(pred):
        raise ValueError("true_ids and pred_ids must have the same length")
    if len(true) == 0:
        raise ValueError("evaluation requires at least one document")
    return true, pred


def harmonic_purity(true_ids: np.ndarray, pred_ids: np.ndarray) -> tuple[float, float, float]:
    """Return purity, inverse purity, and their harmonic mean."""

    true, pred = _validated_ids(true_ids, pred_ids)
    by_pred: dict[int, Counter[int]] = defaultdict(Counter)
    by_true: dict[int, Counter[int]] = defaultdict(Counter)
    for true_id, pred_id in zip(true, pred, strict=True):
        by_pred[int(pred_id)][int(true_id)] += 1
        by_true[int(true_id)][int(pred_id)] += 1

    purity = sum(counter.most_common(1)[0][1] for counter in by_pred.values()) / len(true)
    inverse = sum(counter.most_common(1)[0][1] for counter in by_true.values()) / len(true)
    harmonic = 0.0 if purity + inverse == 0 else 2 * purity * inverse / (purity + inverse)
    return purity, inverse, harmonic


def mapped_accuracy(true_ids: np.ndarray, pred_ids: np.ndarray) -> tuple[float, dict[int, int]]:
    """Map predicted clusters to gold classes with Hungarian matching."""

    true, pred = _validated_ids(true_ids, pred_ids)
    true_values = sorted(set(int(value) for value in true))
    pred_values = sorted(set(int(value) for value in pred))
    true_index = {value: i for i, value in enumerate(true_values)}
    pred_index = {value: i for i, value in enumerate(pred_values)}
    matrix = np.zeros((len(true_values), len(pred_values)), dtype=int)
    for true_id, pred_id in zip(true, pred, strict=True):
        matrix[true_index[int(true_id)], pred_index[int(pred_id)]] += 1
    rows, cols = linear_sum_assignment(-matrix)
    mapping = {pred_values[col]: true_values[row] for row, col in zip(rows, cols, strict=True)}
    matched = sum(
        mapping.get(int(pred_id)) == int(true_id)
        for true_id, pred_id in zip(true, pred, strict=True)
    )
    return matched / len(true), mapping


def evaluate(true_ids: np.ndarray, pred_ids: np.ndarray) -> dict[str, float]:
    """Compute external metrics after model selection has been frozen."""

    true, pred = _validated_ids(true_ids, pred_ids)
    purity, inverse, harmonic = harmonic_purity(true, pred)
    accuracy, _ = mapped_accuracy(true, pred)
    return {
        "mapped_accuracy": accuracy,
        "purity": purity,
        "inverse_purity": inverse,
        "harmonic_purity": harmonic,
        "ari": adjusted_rand_score(true, pred),
        "nmi": normalized_mutual_info_score(true, pred),
        "n_predicted_topics": float(len(set(int(value) for value in pred))),
    }
