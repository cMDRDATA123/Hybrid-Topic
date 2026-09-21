"""Offline lifecycle demo with fixed topics and a tiny deterministic embedder.

Run from the repository root: python -m examples.basic_workflow --output tmp/demo
This exercises software behavior with the fixed defaults, not discovery quality.
"""

import argparse
from pathlib import Path

import numpy as np

from hybrid_topic import HybridTopic
from hybrid_topic.types import Topic


class DemoEmbedder:
    """Toy word features, used only to keep the example fully offline."""

    model_name = "hybrid-topic-offline-demo-v1"

    def encode(self, texts):
        groups = [("battery", "charge"), ("delivery", "shipping"), ("unrelated", "finance")]
        return np.asarray([
            [float(any(term in text.casefold() for term in group)) for group in groups]
            for text in texts
        ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    documents = [
        "The battery is short lived", "The battery needs a charge",
        "Delivery was delayed", "Shipping and delivery were fast",
        "Unrelated finance comment",
    ]
    codebook = [
        Topic("T1", "Battery", "Battery and charging experience", ("battery", "charge")),
        Topic("T2", "Delivery", "Delivery and shipping experience", ("delivery", "shipping")),
    ]
    model = HybridTopic(embedding_model=DemoEmbedder())
    model.fit(documents, codebook=codebook)
    model.export(args.output / "training_results")
    model.save(args.output / "model")
    restored = HybridTopic.load(args.output / "model", embedding_model=DemoEmbedder())
    new_documents = ["Battery charge issue", "Delivery arrived", "Unrelated finance text"]
    result = restored.transform_result(new_documents)
    result.export(args.output / "new_document_results")
    individual = [restored.transform([text])[0][0] for text in new_documents]
    assert individual == result.get_document_info()["topic_id"].tolist()
    print(model.get_topic_info()[["topic_id", "name", "count"]].to_string(index=False))
    print(result.get_document_info().to_string(index=False))
    print("Save/load and individual-versus-batch predictions agree.")
    print(f"Artifacts: {args.output.resolve()}")


if __name__ == "__main__":
    main()
