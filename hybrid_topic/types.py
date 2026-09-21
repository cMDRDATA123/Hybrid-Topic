"""Shared topic records and backend contracts for the current public model."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol
import numpy as np

@dataclass(frozen=True)
class Topic:
    """Minimal normalized topic record shared by first-pass and residual generation."""

    topic_id: str
    name: str
    definition: str
    core_keywords: tuple[str, ...]

    def prototype_text(self) -> str:
        keywords = ", ".join(self.core_keywords)
        return f"[Topic] {self.name}\n[Definition] {self.definition}\n[Keywords] {keywords}"



@dataclass(frozen=True)
class GenerationResult:
    """Auditable output from an initial or residual LLM generation call."""

    topics: tuple[Topic, ...]
    prompt: str
    raw_response: str
    metadata: dict[str, object] = field(default_factory=dict)



class TopicGenerator(Protocol):
    """LLM adapter contract; implementations must not receive labels."""

    def generate_initial(self, documents: list[str]) -> GenerationResult: ...

    def generate_residual(
        self, existing_codebook: list[Topic], residual_documents: list[str]
    ) -> GenerationResult: ...



class TextEmbedder(Protocol):
    """Local embedding adapter contract."""

    def encode(self, texts: list[str]) -> np.ndarray: ...



def _unit_rows(vectors: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty 2D array")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0.0):
        raise ValueError(f"{name} must not contain zero vectors")
    return matrix / norms[:, None]

