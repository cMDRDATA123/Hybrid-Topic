"""Graph diffusion assignment utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np



@dataclass(frozen=True)
class DiffusionResult:
    scores: np.ndarray
    assignments: np.ndarray
    confidence: np.ndarray
    reached: np.ndarray
    natural_coverage: float
    iterations_run: int
    converged: bool
    threshold: float
    adjacency: np.ndarray


def _as_2d_float_array(name: str, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    return arr


def quantile_threshold(similarity: np.ndarray, p: float) -> float:
    """Return the off-diagonal similarity threshold at quantile p."""

    sim = _as_2d_float_array("similarity", similarity)
    if sim.shape[0] != sim.shape[1]:
        raise ValueError("similarity must be square")
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    mask = ~np.eye(sim.shape[0], dtype=bool)
    values = sim[mask]
    if values.size == 0:
        return float("inf")
    return float(np.quantile(values, p))


def build_adjacency(similarity: np.ndarray, p: float) -> tuple[np.ndarray, float]:
    """Build a row-normalized graph from similarities above the P quantile."""

    sim = _as_2d_float_array("similarity", similarity)
    threshold = quantile_threshold(sim, p)
    adjacency = np.where(sim >= threshold, sim, 0.0)
    np.fill_diagonal(adjacency, 0.0)
    row_sums = adjacency.sum(axis=1, keepdims=True)
    adjacency = np.divide(adjacency, row_sums, out=np.zeros_like(adjacency), where=row_sums > 0)
    return adjacency, threshold


def diffuse_scores(
    seed_scores: np.ndarray,
    similarity: np.ndarray,
    *,
    p: float,
    rounds: int = 10,
    diffusion_rate: float = 0.70,
    seed_per_topic: int = 5,
    convergence_tolerance: float = 1e-6,
) -> DiffusionResult:
    """Diffuse topic evidence across a document graph.

    ``seed_scores`` are raw document-topic evidence. The configured strongest
    documents per topic are used as seeds; a document is naturally covered only
    when a seed or graph propagation reaches it. Unreached texts remain
    unassigned for residual discovery rather than receiving a fallback.
    """

    seeds = _as_2d_float_array("seed_scores", seed_scores)
    sim = _as_2d_float_array("similarity", similarity)
    if sim.shape[0] != sim.shape[1]:
        raise ValueError("similarity must be square")
    if sim.shape[0] != seeds.shape[0]:
        raise ValueError("seed_scores and similarity must have the same number of documents")
    if rounds < 1:
        raise ValueError("rounds must be positive")
    if not 0.0 < diffusion_rate < 1.0:
        raise ValueError("diffusion_rate must be in (0, 1)")
    if seed_per_topic < 1:
        raise ValueError("seed_per_topic must be positive")
    if convergence_tolerance < 0.0:
        raise ValueError("convergence_tolerance must be non-negative")

    adjacency, threshold = build_adjacency(sim, p)
    seeded_scores = np.zeros_like(seeds)
    for topic_index in range(seeds.shape[1]):
        ranked = np.argsort(seeds[:, topic_index])[::-1][:seed_per_topic]
        seeded_scores[ranked, topic_index] = np.maximum(seeds[ranked, topic_index], 0.0)

    scores = seeded_scores.copy()
    converged = False
    iterations_run = 0
    for iteration in range(rounds):
        propagated = np.einsum("ij,jk->ik", adjacency, scores, optimize=True)
        updated_scores = (1.0 - diffusion_rate) * seeded_scores + diffusion_rate * propagated
        iterations_run = iteration + 1
        if float(np.max(np.abs(updated_scores - scores))) <= convergence_tolerance:
            converged = True
            scores = updated_scores
            break
        scores = updated_scores

    assignments = scores.argmax(axis=1)
    confidence = scores.max(axis=1)
    reached = confidence > 0.0
    assignments = assignments.astype(int, copy=False)
    assignments[~reached] = -1
    return DiffusionResult(
        scores=scores,
        assignments=assignments,
        confidence=confidence,
        reached=reached,
        natural_coverage=float(reached.mean()),
        iterations_run=iterations_run,
        converged=converged,
        threshold=threshold,
        adjacency=adjacency,
    )
