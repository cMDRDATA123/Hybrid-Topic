"""Model limits and reproducible, whole-document prompt packing."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Callable

PACKING_VERSION = "context-budget-whole-documents-v1"
DEFAULT_MODEL_CATALOG_PATH = Path(__file__).with_name("model_catalog.json")


def _catalog_path(path: str | Path | None) -> Path:
    """Resolve the replaceable capability file used by generation budgeting."""
    if path is not None:
        return Path(path).expanduser()
    configured = os.environ.get("HYBRID_TOPIC_MODEL_CATALOG")
    return Path(configured).expanduser() if configured else DEFAULT_MODEL_CATALOG_PATH


def load_model_catalog(path: str | Path | None = None) -> dict:
    """Load and validate the external provider/model catalog.

    The file is intentionally read at call time. A user can refresh or replace
    it between runs without changing Python code. The resolver still records
    the entry's source in ``ModelLimits`` and refuses entries without verified
    context/tokenizer data.
    """
    catalog_path = _catalog_path(path)
    try:
        value = json.loads(catalog_path.read_text())
    except OSError as exc:
        raise ValueError(f"cannot read model catalog: {catalog_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid model catalog JSON: {catalog_path}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("providers"), dict):
        raise ValueError("model catalog must contain a providers object")
    return value


def freeze_model_catalog(destination: str | Path, catalog_path: str | Path | None = None) -> dict:
    """Copy the selected catalog once; never overwrite a prior frozen file."""
    source = _catalog_path(catalog_path).resolve()
    raw = source.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or not isinstance(value.get("providers"), dict):
        raise ValueError("model catalog must contain a providers object")
    path = Path(destination).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
    return {"path": str(path), "source_path": str(source),
            "sha256": hashlib.sha256(raw).hexdigest()}


def verify_frozen_model_catalog(record: dict) -> None:
    if hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest() != record["sha256"]:
        raise RuntimeError("frozen model catalog changed")


def _catalog_models(catalog: dict, provider: str) -> dict[str, dict]:
    provider_entry = catalog.get("providers", {}).get(provider, {})
    if not isinstance(provider_entry, dict):
        return {}
    models = provider_entry.get("models", {})
    return models if isinstance(models, dict) else {}


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class ModelLimits:
    context_window: int
    max_output_tokens: int | None
    source: str
    tokenizer_encoding: str | None = None
    reasoning_efforts: tuple[str, ...] | None = None
    provider: str = "custom"
    model: str = ""
    catalog_path: str | None = None
    catalog_snapshot_date: str | None = None
    catalog_sha256: str | None = None
    catalog_entry: dict = field(default_factory=dict)
    capability_sources: dict = field(default_factory=dict)


def resolve_model_limits(
    model: str,
    *,
    provider: str = "openai",
    context_window: int | None = None,
    max_output_tokens: int | None = None,
    tokenizer_encoding: str | None = None,
    catalog_path: str | Path | None = None,
    require_tokenizer: bool = True,
) -> ModelLimits:
    """Resolve capabilities from the replaceable catalog.

    A catalog may list a provider model before its limits have been verified.
    Such entries are recognized, but still need explicit context and tokenizer
    values. An explicit value may tighten a verified limit but may not exceed
    it. Unknown models need all values required for safe packing; this prevents
    silently treating a provider's context or tokenizer as a guess.
    """
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("provider must be a non-empty string")
    provider = provider.strip().lower()
    model = model.strip()
    resolved_catalog_path = _catalog_path(catalog_path)
    catalog_bytes = resolved_catalog_path.read_bytes()
    catalog = json.loads(catalog_bytes)
    if not isinstance(catalog, dict) or not isinstance(catalog.get("providers"), dict):
        raise ValueError("model catalog must contain a providers object")
    entry = _catalog_models(catalog, provider).get(model)
    if not isinstance(entry, dict):
        entry = None
    documented_context = entry.get("context_window") if entry else None
    documented_output = entry.get("max_output_tokens") if entry else None
    catalog_tokenizer = entry.get("tokenizer_encoding") if entry else None
    efforts = entry.get("reasoning_efforts") if entry else None
    if efforts is not None and (not isinstance(efforts, (list, tuple)) or
            any(not isinstance(e, str) or not e.strip() for e in efforts)):
        raise ValueError(f"invalid reasoning_efforts in model catalog for {provider}/{model}")
    efforts = tuple(efforts) if efforts is not None else None
    if documented_context is not None:
        documented_context = _positive_int(documented_context, "catalog context_window")
    if documented_output is not None:
        documented_output = _positive_int(documented_output, "catalog max_output_tokens")
    if catalog_tokenizer is not None and (not isinstance(catalog_tokenizer, str) or not catalog_tokenizer.strip()):
        raise ValueError(f"invalid tokenizer_encoding in model catalog for {provider}/{model}")

    has_verified_context = documented_context is not None
    has_verified_tokenizer = catalog_tokenizer is not None or not require_tokenizer
    if not has_verified_context or not has_verified_tokenizer:
        if not has_verified_context and context_window is None:
            if entry is not None:
                raise ValueError("verified context_window is required for this catalog model")
            raise ValueError("context_window is required for an unknown model")
        if not has_verified_tokenizer and tokenizer_encoding is None:
            if entry is not None:
                raise ValueError("verified tokenizer_encoding is required for this catalog model")
            raise ValueError("tokenizer_encoding is required for an unknown model")
        context = _positive_int(context_window, "context_window") if context_window is not None else documented_context
        output_limit = documented_output
        source = "explicit_configuration" if context_window is not None or tokenizer_encoding is not None else "provider_model_catalog"
        if documented_context is not None and context > documented_context:
            raise ValueError("context_window exceeds the documented model context limit")
        catalog_tokenizer = tokenizer_encoding or catalog_tokenizer
    else:
        if context_window is None:
            context = documented_context
            source = str(entry.get("source") or "provider_model_catalog") if entry else "provider_model_catalog"
        else:
            context = _positive_int(context_window, "context_window")
            if context > documented_context:
                raise ValueError("context_window exceeds the documented model context limit")
            source = "explicit_configuration"
        output_limit = documented_output
        catalog_tokenizer = tokenizer_encoding or catalog_tokenizer
    if tokenizer_encoding is not None and (not isinstance(tokenizer_encoding, str) or not tokenizer_encoding.strip()):
        raise ValueError("tokenizer_encoding must be a non-empty string")
    if max_output_tokens is not None:
        _positive_int(max_output_tokens, "max_output_tokens")
    return ModelLimits(
        context, output_limit, source, catalog_tokenizer, efforts, provider, model,
        str(resolved_catalog_path),
        str(catalog.get("snapshot_date")) if catalog.get("snapshot_date") is not None else None,
        hashlib.sha256(catalog_bytes).hexdigest(), dict(entry or {}),
        {"context_window": "explicit_configuration" if context_window is not None else "catalog",
         "tokenizer_encoding": "explicit_configuration" if tokenizer_encoding is not None else
             ("catalog" if catalog_tokenizer else "external_counter"),
         "max_output_tokens": "catalog" if documented_output is not None else "unknown",
         "reasoning_efforts": "catalog" if efforts is not None else "unknown"},
    )


def resolve_local_model_limits(model_path: str | Path, *, context_window: int | None = None) -> ModelLimits:
    """Read a local model's declared context from its config when available."""
    path = Path(model_path)
    config_path = path / "config.json"
    config: dict[str, object] = {}
    if config_path.is_file():
        try:
            loaded = json.loads(config_path.read_text())
            if isinstance(loaded, dict):
                config = loaded
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot read local model config: {config_path}") from exc
    declared = context_window
    if declared is None:
        for key in ("max_position_embeddings", "max_seq_len", "max_sequence_length", "seq_length"):
            value = config.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                declared = value
                break
    if declared is None:
        raise ValueError("context_window is required when the local model config declares no context")
    return ModelLimits(
        _positive_int(declared, "context_window"), None,
        "local_model_config" if config_path.is_file() and context_window is None else "explicit_configuration",
        None, (), "local", str(path),
    )


@dataclass(frozen=True)
class GenerationBudget:
    context_window: int
    input_context_ratio: float = 0.75
    max_output_tokens: int = 16384
    safety_margin_tokens: int = 1024

    def __post_init__(self):
        _positive_int(self.context_window, "context_window")
        _positive_int(self.max_output_tokens, "max_output_tokens")
        _positive_int(self.safety_margin_tokens, "safety_margin_tokens")
        if (isinstance(self.input_context_ratio, bool)
                or not isinstance(self.input_context_ratio, (int, float))
                or not 0 < self.input_context_ratio <= 0.75):
            raise ValueError("input_context_ratio must be in (0, 0.75]")
        if self.input_limit <= 0:
            raise ValueError("No input budget remains; reduce max_output_tokens or increase context_window")

    @property
    def input_limit(self) -> int:
        return min(math.floor(self.context_window * self.input_context_ratio),
                   self.context_window - self.max_output_tokens - self.safety_margin_tokens)


def pack_documents(documents: list[str], *, render: Callable[[str], str],
                   count_tokens: Callable[[str], int], budget: GenerationBudget,
                   sampling_seed: int = 42) -> tuple[str, dict]:
    """Count the complete rendered request, then sample whole documents if needed.

    Random-order budget packing may favor shorter documents; it is not
    label-stratified or a claim of exact proportional representation.
    ``count_tokens`` must include transport/template and any repair overhead.
    """
    if isinstance(sampling_seed, bool) or not isinstance(sampling_seed, int):
        raise ValueError("sampling_seed must be an integer")
    if not documents or any(not isinstance(d, str) or not d.strip() for d in documents):
        raise ValueError("generation requires nonempty text documents")
    blocks = [f"D{i}: {text}" for i, text in enumerate(documents)]
    fixed_tokens = count_tokens(render(""))
    if fixed_tokens >= budget.input_limit:
        raise ValueError("Fixed prompt/codebook exhausts input budget; increase context or reduce output limit")
    indices = list(range(len(documents)))
    prompt = render("\n\n".join(blocks))
    candidate_tokens = count_tokens(prompt)
    if candidate_tokens > budget.input_limit:
        order = indices.copy()
        random.Random(sampling_seed).shuffle(order)
        remaining = budget.input_limit - fixed_tokens
        indices = []
        # Whole requests are checked again below because token boundaries can
        # differ at joins. No text or codebook is silently truncated.
        for i in order:
            cost = max(1, count_tokens(render(blocks[i] + "\n\n")) - fixed_tokens)
            if cost <= remaining:
                indices.append(i)
                remaining -= cost
        indices.sort()
        prompt = render("\n\n".join(blocks[i] for i in indices))
        while indices and count_tokens(prompt) > budget.input_limit:
            indices.pop()
            prompt = render("\n\n".join(blocks[i] for i in indices))
        if not indices:
            raise ValueError("No complete document fits the input budget; split long documents or increase budget")
    return prompt, {
        "version": PACKING_VERSION,
        "candidate_document_count": len(documents),
        "selected_document_count": len(indices),
        "selected_document_indices": indices,
        "sampling_fraction": len(indices) / len(documents),
        "sampling_seed": sampling_seed,
        "sampling_method": "all" if len(indices) == len(documents) else "random_order_token_budget",
        "candidate_input_tokens": candidate_tokens,
        "reserved_input_tokens": count_tokens(prompt),
        "input_limit": budget.input_limit,
        "truncated_documents": 0,
    }
