"""Public model lifecycle with bounded residual discovery and reference inference.

This development workflow does not run the research Pareto/residual coordinator.
The fixed defaults are engineering starting points, with explicit user overrides.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Sequence
import json

import numpy as np
import pandas as pd

from hybrid_topic.diffusion_assignment import diffuse_scores
from hybrid_topic.residual_policy import ResidualConfig, residual_stop_reason
from hybrid_topic.types import Topic, TextEmbedder, TopicGenerator, _unit_rows
from hybrid_topic.results import TopicResult, validate_documents


WORKFLOW = "bounded-residual-reference-diffusion-v2"
LEGACY_WORKFLOW = "initial-codebook-reference-diffusion-v1"
PARAMETER_DEFAULTS_VERSION = "empirical-defaults-v1"


class HybridTopic:
    """Discover topics with at most two residual rounds and existing diffusion.

    ``p=0.95`` is a fixed engineering starting point, not a per-dataset search
    or a universally optimal setting. A configured generator can carry an
    optional analysis direction. Providing ``codebook`` to fit skips all generation.
    ``max_residual_rounds=0`` disables expansion; the first release caps it at two.
    """

    def __init__(
        self,
        *,
        embedding_model: TextEmbedder,
        p: float = 0.95,
        generator: TopicGenerator | None = None,
        embedding_model_id: str | None = None,
        seed_per_topic: int = 5,
        diffusion_rate: float = 0.70,
        rounds: int = 10,
        max_residual_rounds: int = 2,
    ) -> None:
        if not 0 < p < 1 or not 0 < diffusion_rate < 1:
            raise ValueError("p and diffusion_rate must be in (0, 1)")
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (seed_per_topic, rounds)):
            raise ValueError("seed_per_topic and rounds must be positive integers")
        if type(max_residual_rounds) is not int or not 0 <= max_residual_rounds <= 2:
            raise ValueError("max_residual_rounds must be an integer from 0 to 2")
        self._max_residual_rounds = max_residual_rounds
        self.embedding_model = embedding_model
        self.generator = generator
        self.embedding_model_id = embedding_model_id or getattr(embedding_model, "model_name", None)
        self._config = {
            "p": float(p), "seed_per_topic": seed_per_topic,
            "diffusion_rate": float(diffusion_rate), "rounds": rounds,
        }
        self._codebook: tuple[Topic, ...] | None = None
        self._topic_vectors: np.ndarray | None = None
        self._reference_vectors: np.ndarray | None = None
        self._reference_scores: np.ndarray | None = None
        self._reference_threshold: float | None = None
        self._counts: list[int] = []
        self.result_: TopicResult | None = None
        self.metadata_: dict[str, object] = {}

    def _require_fit(self) -> None:
        if self._codebook is None:
            raise RuntimeError("fit the model or load a saved model first")

    @staticmethod
    def _validated_codebook(codebook: Sequence[Topic]) -> tuple[Topic, ...]:
        topics = tuple(codebook)
        if not topics or any(not isinstance(t, Topic) for t in topics):
            raise ValueError("codebook must contain Topic records")
        for topic in topics:
            if any(not isinstance(s, str) or not s.strip() for s in (topic.topic_id, topic.name, topic.definition)):
                raise ValueError("topic IDs, names and definitions must be non-empty strings")
            if isinstance(topic.core_keywords, str) or not topic.core_keywords or any(
                not isinstance(s, str) or not s.strip() for s in topic.core_keywords
            ):
                raise ValueError("each topic must contain non-empty keywords")
        return tuple(Topic(t.topic_id, t.name, t.definition, tuple(t.core_keywords)) for t in topics)

    def _embed(self, texts: list[str]) -> np.ndarray:
        vectors = _unit_rows(self.embedding_model.encode(texts), name="embeddings")
        if len(vectors) != len(texts):
            raise ValueError("embedding rows must match the input documents")
        return vectors

    def _extend_codebook(
        self, topics: tuple[Topic, ...], topic_vectors: np.ndarray,
        proposed: Sequence[Topic],
    ) -> tuple[tuple[Topic, ...], np.ndarray]:
        """Keep new names and prototypes distinct, using the existing 0.95 limit.

        Only proposals are checked against the current codebook; existing topics
        are retained. Each accepted prototype is embedded once and then reused.
        """
        proposed = self._validated_codebook(proposed) if proposed else ()
        seen = {" ".join(t.name.casefold().split()) for t in topics}
        for topic in proposed:
            name = " ".join(topic.name.casefold().split())
            if name in seen:
                continue
            vector = self._embed([topic.prototype_text()])[0]
            if len(vector) != topic_vectors.shape[1]:
                raise ValueError("residual topic embedding dimensions differ")
            similarities = np.einsum("id,d->i", topic_vectors, vector, optimize=False)
            if float(similarities.max()) > 0.95:
                continue
            topics = (*topics, topic)
            topic_vectors = np.vstack((topic_vectors, vector))
            seen.add(name)
        return topics, topic_vectors

    def fit(self, documents: Sequence[str], *, codebook: Sequence[Topic] | None = None) -> HybridTopic:
        """Discover and expand topics; failed refits preserve the previous model."""
        texts = validate_documents(documents)
        if len(texts) < 2:
            raise ValueError("fit requires at least two documents to establish a graph threshold")
        if codebook is None and self.generator is None:
            raise ValueError("provide a generator or a codebook")
        topics = self._validated_codebook(codebook) if codebook is not None else None
        vectors = self._embed(texts)
        generation_history = []
        if topics is None:
            initial_generation = self.generator.generate_initial(list(texts))
            topics = self._validated_codebook(initial_generation.topics)
            if initial_generation.metadata:
                generation_history.append({**initial_generation.metadata, "stage": "initial"})
        topic_vectors = self._embed([topic.prototype_text() for topic in topics])
        if vectors.shape[1] != topic_vectors.shape[1]:
            raise ValueError("document and topic embedding dimensions differ")
        # The local NumPy/BLAS matmul path raises spurious arithmetic warnings
        # even for finite unit vectors. Direct contractions avoid that path;
        # differential checks agree within 3e-15 for the inspected inputs.
        topic_similarity = np.einsum("id,jd->ij", vectors, topic_vectors, optimize=False)
        document_similarity = np.einsum("id,jd->ij", vectors, vectors, optimize=False)
        diffusion = diffuse_scores(topic_similarity, document_similarity, **self._config)
        initial_topic_count = len(topics)
        history = []
        no_progress = 0
        stop_config = ResidualConfig(max_rounds=self._max_residual_rounds)
        stop_reason = "provided_codebook" if codebook is not None else None
        while stop_reason is None:
            residuals = [text for text, assigned in zip(texts, diffusion.assignments) if assigned < 0]
            stop_reason = residual_stop_reason(
                completed_rounds=len(history), residual_doc_count=len(residuals),
                generated_topic_count=None, valid_new_topic_count=None,
                consecutive_no_progress=no_progress, config=stop_config,
            )
            if stop_reason is not None:
                break
            generation = self.generator.generate_residual(list(topics), residuals)
            if generation.metadata:
                generation_history.append({**generation.metadata, "stage": "residual",
                                           "round": len(history) + 1,
                                           "source_document_indices": np.flatnonzero(diffusion.assignments < 0).tolist()})
            previous_count = len(topics)
            topics, topic_vectors = self._extend_codebook(topics, topic_vectors, generation.topics)
            accepted = len(topics) - previous_count
            step = {
                "round": len(history) + 1, "input_documents": len(residuals),
                "proposed_topics": len(generation.topics), "accepted_topics": accepted,
                "total_topics": len(topics),
            }
            if accepted:
                topic_similarity = np.einsum("id,jd->ij", vectors, topic_vectors, optimize=False)
                # Reuse document embeddings/similarities; retain only this stage.
                diffusion = diffuse_scores(topic_similarity, document_similarity, **self._config)
            remaining = int((diffusion.assignments < 0).sum())
            step["unassigned_after"] = remaining
            history.append(step)
            no_progress = no_progress + 1 if remaining >= len(residuals) else 0
            if not accepted:
                stop_reason = residual_stop_reason(
                    completed_rounds=len(history) - 1, residual_doc_count=len(residuals),
                    generated_topic_count=len(generation.topics), valid_new_topic_count=0,
                    consecutive_no_progress=no_progress, config=stop_config,
                )
        result = TopicResult(texts, topics, diffusion.assignments, scores=diffusion.confidence)
        metadata = {
            "workflow": WORKFLOW,
            "inference": "frozen_reference_one_step_v1",
            "query_encoding": "one_document_per_call",
            "configuration": {**self._config, "max_residual_rounds": self._max_residual_rounds},
            "parameter_defaults_version": PARAMETER_DEFAULTS_VERSION,
            "codebook_source": "provided" if codebook is not None else "generator_initial",
            "generation_format": getattr(self.generator, "response_format_version", "custom_generator") if codebook is None else "provided_codebook",
            "topic_instruction": getattr(self.generator, "topic_instruction", None) if codebook is None else None,
            "training_documents": len(texts),
            "initial_topic_count": initial_topic_count,
            "residual_history": history,
            "residual_stop_reason": stop_reason,
            "residual_generation_calls": len(history),
            "generation_calls": (1 + len(history)) if codebook is None else 0,
            "generation_history": generation_history,
            "residual_max_topic_similarity": 0.95,
            "training_coverage": result.coverage,
            "graph_threshold": float(diffusion.threshold) if np.isfinite(diffusion.threshold) else None,
            "iterations_run": diffusion.iterations_run,
            "score_semantics": "uncalibrated_assignment_score",
        }
        # Commit the new model only after all validation and assignment succeeds.
        self._codebook = topics
        self._topic_vectors = topic_vectors.copy()
        self._reference_vectors = vectors.copy()
        self._reference_scores = diffusion.scores.copy()
        self._reference_threshold = float(diffusion.threshold)
        self._counts = result.get_topic_info()["count"].tolist()
        self.result_ = result
        self.metadata_ = metadata
        return self

    def fit_transform(self, documents: Sequence[str], *, codebook: Sequence[Topic] | None = None) -> tuple[np.ndarray, np.ndarray]:
        self.fit(documents, codebook=codebook)
        frame = self.result_.get_document_info()
        return frame["topic_id"].to_numpy(copy=True), frame["score"].to_numpy(copy=True)

    def transform_result(self, documents: Sequence[str]) -> TopicResult:
        """Assign each new document against frozen training evidence.

        New documents are encoded one at a time and never connect to or seed one
        another. With a deterministic embedding backend, each prediction is
        independent of batch size/order/composition. Text with
        no positive evidence above the frozen graph threshold remains unassigned.
        This inductive extension differs from fit's transductive diffusion and
        requires its own quality evaluation. It does not modify the trained model.
        """
        self._require_fit()
        texts = validate_documents(documents)
        assignments = []
        confidence = []
        # Singleton encoding prevents batch/padding roundoff from changing an
        # edge at the threshold. It also bounds query embedding memory.
        for text in texts:
            vector = self._embed([text])[0]
            if vector.shape[0] != self._reference_vectors.shape[1]:
                raise ValueError("new document embedding dimensions do not match the saved model")
            similarities = np.einsum("id,d->i", self._reference_vectors, vector, optimize=False)
            weights = np.where(
                (similarities >= self._reference_threshold) & (similarities > 0),
                similarities, 0.0,
            )
            mass = float(weights.sum())
            scores = np.einsum("i,ik->k", weights, self._reference_scores, optimize=False) / mass if mass > 0 else np.zeros(len(self._codebook))
            scores = self._config["diffusion_rate"] * scores
            best = int(scores.argmax())
            score = float(scores[best])
            assignments.append(best if score > 0 else -1)
            confidence.append(max(score, 0.0))
        return TopicResult(texts, self._codebook, assignments, scores=confidence)

    def transform(self, documents: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        result = self.transform_result(documents).get_document_info()
        return result["topic_id"].to_numpy(copy=True), result["score"].to_numpy(copy=True)

    def get_topic(self, topic_id: int) -> dict[str, object]:
        self._require_fit()
        if not isinstance(topic_id, (int, np.integer)) or not 0 <= topic_id < len(self._codebook):
            raise KeyError(f"unknown topic_id: {topic_id}")
        topic = self._codebook[topic_id]
        return {
            "topic_id": int(topic_id), "source_topic_id": topic.topic_id,
            "name": topic.name, "definition": topic.definition,
            "keywords": list(topic.core_keywords), "count": self._counts[topic_id],
        }

    def get_topic_info(self) -> pd.DataFrame:
        self._require_fit()
        return pd.DataFrame([self.get_topic(i) for i in range(len(self._codebook))])

    def get_document_info(self, *, topic_id: int | None = None) -> pd.DataFrame:
        self._require_fit()
        if self.result_ is None:
            raise RuntimeError("saved models do not contain training documents; export results before saving")
        return self.result_.get_document_info(topic_id=topic_id)

    def export(self, directory: str | Path) -> Path:
        self.get_document_info()  # Validate that the training result is available.
        return self.result_.export(directory)

    def save(self, directory: str | Path) -> Path:
        """Save JSON model state, not API clients, credentials or training text.

        The same embedding backend must be supplied when loading. This format
        stores topic/reference embeddings and scores, not embedding model weights.
        """
        self._require_fit()
        if not isinstance(self.embedding_model_id, str) or not self.embedding_model_id.strip():
            raise ValueError("set embedding_model_id before saving a custom embedding backend")
        payload = {
            "schema": "hybrid_topic_model_v1", "workflow": self.metadata_["workflow"],
            "embedding_model_id": self.embedding_model_id,
            "configuration": {**self._config, "max_residual_rounds": self._max_residual_rounds},
            "codebook": [asdict(topic) for topic in self._codebook],
            "topic_embeddings": self._topic_vectors.tolist(),
            "reference_embeddings": self._reference_vectors.tolist(),
            "reference_scores": self._reference_scores.tolist(),
            "reference_threshold": self._reference_threshold if np.isfinite(self._reference_threshold) else None,
            "topic_counts": self._counts, "metadata": self.metadata_,
        }
        serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "model.json"
        with path.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
        return directory

    @classmethod
    def load(
        cls, directory: str | Path, *, embedding_model: TextEmbedder,
        embedding_model_id: str | None = None,
    ) -> HybridTopic:
        """Load state without instantiating a generator or making model calls."""
        payload = json.loads((Path(directory) / "model.json").read_text(encoding="utf-8"))
        if payload.get("schema") != "hybrid_topic_model_v1" or payload.get("workflow") not in {WORKFLOW, LEGACY_WORKFLOW}:
            raise ValueError("unsupported model schema or workflow")
        identity = embedding_model_id or getattr(embedding_model, "model_name", None)
        if not isinstance(identity, str) or not identity.strip() or identity != payload.get("embedding_model_id"):
            raise ValueError("embedding_model_id must match the saved embedding backend")
        configuration = dict(payload["configuration"])
        required = {"p", "seed_per_topic", "diffusion_rate", "rounds"}
        if not required.issubset(configuration):
            raise ValueError("saved configuration is missing required assignment parameters")
        configuration.setdefault("max_residual_rounds", 0)
        model = cls(embedding_model=embedding_model, embedding_model_id=identity, **configuration)
        topics = model._validated_codebook([Topic(**item) for item in payload["codebook"]])
        vectors = np.asarray(payload["topic_embeddings"], dtype=float)
        _unit_rows(vectors, name="saved topic embeddings")
        if len(vectors) != len(topics) or not np.allclose(np.linalg.norm(vectors, axis=1), 1.0):
            raise ValueError("saved topic embeddings must match the codebook and be normalized")
        references = np.asarray(payload["reference_embeddings"], dtype=float)
        _unit_rows(references, name="saved reference embeddings")
        if references.shape[1] != vectors.shape[1] or not np.allclose(np.linalg.norm(references, axis=1), 1.0):
            raise ValueError("saved reference embeddings must use the topic embedding space")
        evidence = np.asarray(payload["reference_scores"], dtype=float)
        if evidence.shape != (len(references), len(topics)) or not np.all(np.isfinite(evidence)):
            raise ValueError("saved reference scores must align with reference documents and topics")
        threshold = payload["reference_threshold"]
        if threshold is None:
            if len(references) != 1:
                raise ValueError("missing reference threshold for a multi-document model")
            threshold = float("inf")
        elif not isinstance(threshold, (float, int)) or not np.isfinite(threshold) or not -1.000001 <= threshold <= 1.000001:
            raise ValueError("invalid reference threshold")
        counts = payload["topic_counts"]
        if len(counts) != len(topics) or any(type(count) is not int or count < 0 for count in counts):
            raise ValueError("invalid saved topic counts")
        model._codebook = topics
        model._topic_vectors = vectors.copy()
        model._reference_vectors = references.copy()
        model._reference_scores = evidence.copy()
        model._reference_threshold = float(threshold)
        model._counts = list(counts)
        model.metadata_ = dict(payload["metadata"])
        return model
