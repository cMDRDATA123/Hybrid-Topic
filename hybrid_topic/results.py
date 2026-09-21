"""Inspect and export topic assignments without running or selecting a model."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import json

import numpy as np
import pandas as pd

from hybrid_topic.types import Topic


def validate_documents(documents: Sequence[str]) -> list[str]:
    """Preserve text and order; reject invalid rows instead of silently dropping them."""
    if isinstance(documents, (str, bytes)):
        raise ValueError("documents must be a collection of strings, not one string")
    try:
        values = list(documents)
    except TypeError as error:
        raise ValueError("documents must be a collection of strings") from error
    if not values:
        raise ValueError("documents must not be empty")
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"document at index {index} must be a non-empty string")
    return values


def read_documents(path: str | Path, *, text_column: str = "text") -> list[str]:
    """Read one CSV text column, keeping literal NA/numeric text and row order."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if text_column not in frame.columns:
        raise ValueError(f"text_column {text_column!r} is missing from the CSV")
    return validate_documents(frame[text_column].tolist())


class TopicResult:
    """A result snapshot, not a fitted model or a parameter selector.

    Assignment IDs are zero-based codebook positions; -1 means unassigned.
    Source topic IDs are retained separately because historical generators can
    repeat them across residual rounds. Scores are optional, uncalibrated values.
    Topic IDs are local to this snapshot, not cross-run topic identities.
    """

    def __init__(
        self,
        documents: Sequence[str],
        codebook: Sequence[Topic],
        assignments: Sequence[int],
        *,
        scores: Sequence[float] | None = None,
    ) -> None:
        texts = validate_documents(documents)
        topics = tuple(codebook)
        if not topics or any(not isinstance(topic, Topic) for topic in topics):
            raise ValueError("codebook must contain Topic records")
        ids = np.asarray(assignments)
        if ids.ndim != 1 or len(ids) != len(texts) or ids.dtype.kind not in "iu":
            raise ValueError("assignments must contain one integer topic index per document")
        if np.any(ids < -1) or np.any(ids >= len(topics)):
            raise ValueError("assignment topic indices must be -1 or refer to the codebook")
        values = None
        if scores is not None:
            values = np.asarray(scores, dtype=float)
            if values.ndim != 1 or len(values) != len(texts) or not np.all(np.isfinite(values)):
                raise ValueError("scores must contain one finite value per document")
        self._documents = tuple(texts)
        self._topics = tuple(
            Topic(t.topic_id, t.name, t.definition, tuple(t.core_keywords)) for t in topics
        )
        self._assignments = tuple(int(value) for value in ids)
        self._scores = tuple(float(value) for value in values) if values is not None else None

    @property
    def coverage(self) -> float:
        return sum(value >= 0 for value in self._assignments) / len(self._documents)

    def get_topic(self, topic_id: int) -> dict[str, object]:
        """Return topic details, including topics assigned zero documents."""
        if not isinstance(topic_id, (int, np.integer)) or not 0 <= topic_id < len(self._topics):
            raise KeyError(f"unknown topic_id: {topic_id}")
        topic = self._topics[topic_id]
        return {
            "topic_id": int(topic_id),
            "source_topic_id": topic.topic_id,
            "name": topic.name,
            "definition": topic.definition,
            "keywords": list(topic.core_keywords),
            "count": self._assignments.count(topic_id),
        }

    def get_topic_info(self) -> pd.DataFrame:
        return pd.DataFrame([self.get_topic(i) for i in range(len(self._topics))])

    def get_document_info(self, *, topic_id: int | None = None) -> pd.DataFrame:
        """Return original rows, optionally filtering a topic or -1 (unassigned)."""
        if topic_id is not None and topic_id != -1:
            self.get_topic(topic_id)
        rows = []
        for i, (text, assigned_id) in enumerate(zip(self._documents, self._assignments, strict=True)):
            rows.append({
                "document_id": i,
                "text": text,
                "topic_id": assigned_id,
                "topic_name": self._topics[assigned_id].name if assigned_id >= 0 else None,
                "assigned": assigned_id >= 0,
                "score": self._scores[i] if self._scores is not None else None,
            })
        frame = pd.DataFrame(rows)
        # pandas 3 infers a string column and converts None to NaN. Keep the
        # public missing-topic value as None so JSON exports contain null.
        frame["topic_name"] = pd.Series([row["topic_name"] for row in rows], dtype=object)
        if topic_id is not None:
            frame = frame.loc[frame["topic_id"] == topic_id].reset_index(drop=True)
        return frame

    def export(self, directory: str | Path) -> Path:
        """Export UTF-8 CSV tables and JSON; refuse to overwrite existing outputs.

        This exports results only, not an executable model for new documents.
        Original document text is included in these local files.
        """
        directory = Path(directory)
        paths = [directory / name for name in ("topics.csv", "documents.csv", "result.json")]
        for path in paths:
            if path.exists():
                raise FileExistsError(f"refusing to overwrite result file: {path}")
        topics = self.get_topic_info()
        documents = self.get_document_info()
        payload = {
            "schema": "hybrid_topic_result_v1",
            "coverage": self.coverage,
            "score_semantics": "uncalibrated_assignment_score" if self._scores is not None else "unavailable",
            "topics": topics.to_dict(orient="records"),
            "documents": documents.to_dict(orient="records"),
        }
        serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        topics["keywords"] = topics["keywords"].map(lambda value: json.dumps(value, ensure_ascii=False))
        directory.mkdir(parents=True, exist_ok=True)
        topics.to_csv(paths[0], index=False, encoding="utf-8", mode="x")
        documents.to_csv(paths[1], index=False, encoding="utf-8", mode="x")
        with paths[2].open("x", encoding="utf-8") as stream:
            stream.write(serialized)
        return directory
