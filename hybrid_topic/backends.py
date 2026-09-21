"""Generation and embedding adapters shared by the tool and research runner.

Imports do not load models or create API clients. Configure model access explicitly.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import inspect
import json
from dataclasses import asdict
import time
import warnings
from typing import Any
import numpy as np

from hybrid_topic.generation_budget import (
    GenerationBudget, pack_documents, resolve_local_model_limits, resolve_model_limits,
)
from hybrid_topic.types import GenerationResult, TextEmbedder, Topic, TopicGenerator


def _topic_payload(topics: list[Topic]) -> list[dict[str, object]]:
    return [
        {
            "topic_id": topic.topic_id,
            "name": topic.name,
            "definition": topic.definition,
            "core_keywords": list(topic.core_keywords),
        }
        for topic in topics
    ]


GENERATION_FORMAT_VERSION = "strict-codebook-schema-v1"
TOPIC_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "topic_codebook", "strict": True,
        "schema": {
            "type": "object", "additionalProperties": False, "required": ["topics"],
            "properties": {"topics": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["topic_id", "name", "definition", "core_keywords"],
                "properties": {
                    "topic_id": {"type": "string"}, "name": {"type": "string"},
                    "definition": {"type": "string"},
                    "core_keywords": {"type": "array", "items": {"type": "string"}},
                },
            }}},
        },
    },
}


class TopicGenerationError(ValueError):
    """Invalid generation, with the original response retained for diagnosis."""
    def __init__(self, message: str, raw_response: str):
        super().__init__(message)
        self.raw_response = raw_response


def _parse_topics(raw_response: str, *, prefix: str) -> tuple[Topic, ...]:
    text = raw_response.strip()
    if text.startswith("```"):
        text = text.partition("\n")[2].rsplit("```", 1)[0].strip()
    try:
        payload = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise TopicGenerationError("generator did not return valid JSON", raw_response) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("topics"), list):
        raise TopicGenerationError("generator response must contain a topics list", raw_response)
    topics: list[Topic] = []
    for index, item in enumerate(payload["topics"], start=1):
        if not isinstance(item, dict):
            raise TopicGenerationError(f"topic {index} is not an object; no partial codebook accepted", raw_response)
        topic_id = item.get("topic_id", f"{prefix}{index}")
        name, definition = item.get("name"), item.get("definition")
        keywords = item.get("core_keywords", item.get("keywords"))
        valid = all(isinstance(value, str) and value.strip() for value in (topic_id, name, definition))
        valid = valid and isinstance(keywords, list) and bool(keywords) and all(isinstance(k, str) and k.strip() for k in keywords)
        if not valid:
            raise TopicGenerationError(f"topic {index} has invalid or missing fields; no partial codebook accepted", raw_response)
        topics.append(Topic(topic_id.strip(), name.strip(), definition.strip(), tuple(k.strip() for k in keywords)))
    return tuple(topics)


def _direction_block(topic_instruction: str | None) -> str:
    if topic_instruction is None:
        return ""
    if not isinstance(topic_instruction, str):
        raise ValueError("topic_instruction must be a string or None")
    direction = topic_instruction.strip()
    return f"Analysis direction supplied by the user:\n{direction}\n\n" if direction else ""


def _initial_prompt(documents: list[str], *, topic_instruction: str | None = None, document_block: str | None = None) -> str:
    return (
        "Create an Open-K topic codebook from the induction documents below. "
        "Use no gold labels, benchmark category names, target K, or external taxonomy. "
        "Discover a compact set of recurring topics yourself. Each topic must provide a name, "
        "definition, and all useful core_keywords. Return JSON only as "
        '{"topics":[{"topic_id":"T1","name":"...","definition":"...","core_keywords":["..."]}]}.\n\n'
        f"{_direction_block(topic_instruction)}"
        "Induction documents:\n" + ("\n\n".join(f"D{i}: {text}" for i, text in enumerate(documents)) if document_block is None else document_block)
    )


def _residual_prompt(
    existing_codebook: list[Topic], residual_documents: list[str], *, topic_instruction: str | None = None,
    document_block: str | None = None
) -> str:
    return (
        "Extend the current Open-K codebook using only the residual documents. "
        "The full current codebook is shown below. Return an empty topics list when the residual "
        "does not contain a recurring theme genuinely absent from that codebook. Do not rename, "
        "duplicate, or split an existing topic. Return JSON only as "
        '{"topics":[{"topic_id":"R1","name":"...","definition":"...","core_keywords":["..."]}]}.\n\n'
        f"{_direction_block(topic_instruction)}"
        f"Current codebook:\n{json.dumps(_topic_payload(existing_codebook), ensure_ascii=False)}\n\n"
        "Residual documents:\n" + ("\n\n".join(f"D{i}: {text}" for i, text in enumerate(residual_documents)) if document_block is None else document_block)
    )


class SentenceTransformerEmbedder(TextEmbedder):
    def __init__(
        self, model_name: str, *, model: Any | None = None,
        device: str | None = "mps", local_files_only: bool = True,
    ) -> None:
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ModuleNotFoundError as exc:
                if exc.name != "sentence_transformers":
                    raise
                raise ImportError(
                    "SentenceTransformerEmbedder requires the embeddings extra. "
                    "From the repository: python -m pip install '.[embeddings]'"
                ) from exc

            model = SentenceTransformer(
                model_name,
                trust_remote_code=True,
                device=device,
                local_files_only=local_files_only,
            )
        self.model = model
        self.model_name = model_name
        self.model.max_seq_length = 512

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                texts,
                batch_size=16,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float64,
        )


_SYSTEM = "You produce valid compact JSON for open topic discovery."
_REPAIR = ("\n\nThe previous response failed JSON/topic-field validation. Generate the complete "
           "topics object again from the SAME documents above. Include topic_id, name, "
           "definition and nonempty core_keywords for every topic. Return JSON only.")


# Options must not bypass the owned prompt, output budget, JSON contract or
# single-response lifecycle, including through OpenAI SDK extra_body merging.
_OWNED_REQUEST_FIELDS = {"model", "messages", "system", "max_tokens", "max_completion_tokens",
                         "max_output_tokens", "response_format", "output_config", "reasoning_effort",
                         "stream", "n", "tools", "tool_choice", "stop", "stop_sequences"}


def _request_options(options):
    result = deepcopy(options or {})
    if not isinstance(result, dict):
        raise ValueError("request_options must be a dictionary")
    extra = result.get("extra_body", {})
    if not isinstance(extra, dict):
        raise ValueError("extra_body must be a dictionary")
    forbidden = _OWNED_REQUEST_FIELDS.intersection(result) | _OWNED_REQUEST_FIELDS.intersection(extra)
    # Transport/authentication options belong on the configured client, and
    # must never be serialized into saved model metadata.
    allowed = {"temperature", "top_p", "top_k", "seed", "thinking", "service_tier",
               "frequency_penalty", "presence_penalty", "extra_body"}
    if forbidden or set(result) - allowed:
        raise ValueError(f"unsupported or reserved request_options: {sorted(forbidden | (set(result) - allowed))}")
    json.dumps(result, allow_nan=False)
    return result


def _token_count(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("input token counter must return a nonnegative integer")
    return value


class _BudgetedTopicGenerator:
    """Shared packing and format-only retry behavior for public adapters."""
    def _configure(self, model, *, topic_instruction, context_window,
                   input_context_ratio, max_output_tokens, max_format_retries,
                   sampling_seed, safety_margin_tokens, provider="custom",
                   limits=None, reasoning_effort=None, tokenizer_encoding=None,
                   catalog_path=None):
        _direction_block(topic_instruction)
        if isinstance(max_format_retries, bool) or not isinstance(max_format_retries, int) or max_format_retries < 0:
            raise ValueError("max_format_retries must be a nonnegative integer")
        if isinstance(sampling_seed, bool) or not isinstance(sampling_seed, int):
            raise ValueError("sampling_seed must be an integer")
        if isinstance(max_output_tokens, bool) or not isinstance(max_output_tokens, int) or max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive integer")
        limits = limits or resolve_model_limits(
            model, provider=provider, context_window=context_window,
            max_output_tokens=max_output_tokens, tokenizer_encoding=tokenizer_encoding,
        )
        effective_output = min(max_output_tokens, limits.max_output_tokens or max_output_tokens)
        if effective_output != max_output_tokens:
            warnings.warn(f"max_output_tokens limited to model maximum {effective_output}", UserWarning, stacklevel=3)
        self.budget = GenerationBudget(limits.context_window, input_context_ratio, effective_output, safety_margin_tokens)
        self.topic_instruction = topic_instruction
        self.max_format_retries = max_format_retries
        self.sampling_seed = sampling_seed
        if reasoning_effort is not None:
            if not isinstance(reasoning_effort, str) or not reasoning_effort.strip():
                raise ValueError("reasoning_effort must be a non-empty string or None")
            if limits.reasoning_efforts is None:
                raise ValueError("reasoning_effort support is unverified; configure reasoning_efforts in the model catalog")
            if reasoning_effort not in limits.reasoning_efforts:
                raise ValueError(f"reasoning_effort {reasoning_effort!r} is not supported by {provider}/{model}")
        self.generation_config = {**asdict(self.budget), "model": model,
                                  "provider": provider,
                                  "context_source": limits.source,
                                  "catalog_path": limits.catalog_path,
                                  "catalog_snapshot_date": limits.catalog_snapshot_date,
                                  "catalog_sha256": limits.catalog_sha256,
                                  "catalog_entry": deepcopy(limits.catalog_entry),
                                  "capability_sources": dict(limits.capability_sources),
                                  "requested_max_output_tokens": max_output_tokens,
                                  "model_max_output_tokens": limits.max_output_tokens,
                                  "reasoning_effort": reasoning_effort,
                                  "tokenizer_encoding": limits.tokenizer_encoding,
                                  "max_format_retries": max_format_retries,
                                  "sampling_seed": sampling_seed}
        self.last_generation_ = None

    def _messages(self, prompt):
        return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}]

    def _run(self, documents, render, *, prefix):
        self.last_generation_ = None
        repair_suffix = _REPAIR if self.max_format_retries else ""
        # Reserve repair space before selecting documents, so retries never
        # change the sample or exceed the original input budget.
        prompt, packing = pack_documents(
            documents, render=render,
            count_tokens=lambda text: max(self._count_input(text), self._count_input(text + repair_suffix)),
            budget=self.budget, sampling_seed=self.sampling_seed,
        )
        metadata = {"configuration": dict(self.generation_config), "packing": packing, "attempts": []}
        self.last_generation_ = {**metadata, "attempts": []}
        for attempt in range(self.max_format_retries + 1):
            attempt_prompt = prompt + (_REPAIR if attempt else "")
            input_tokens = self._count_input(attempt_prompt)
            if input_tokens > self.budget.input_limit:
                raise ValueError("complete request exceeds input budget")
            record = {"attempt": attempt + 1, "input_tokens": input_tokens}
            started = time.perf_counter()
            raw_response = ""
            try:
                raw_response, details = self._request(attempt_prompt)
                record.update(details)
                if details.get("finish_reason") not in (None, "stop") or details.get("refusal"):
                    # Truncation/refusal is not a JSON syntax error. Regenerating
                    # at the same output cap cannot guarantee completion.
                    raise TopicGenerationError("generation refused or did not finish; no partial codebook accepted", raw_response)
                try:
                    topics = _parse_topics(raw_response, prefix=prefix)
                except TopicGenerationError:
                    record["status"] = "format_error"
                    raise
                record["status"] = "success"
            except Exception as exc:
                record.setdefault("status", "request_error")
                record["error_type"] = type(exc).__name__
                if record["status"] != "format_error" or attempt == self.max_format_retries:
                    raise
            finally:
                record["elapsed_seconds"] = time.perf_counter() - started
                metadata["attempts"].append(dict(record))
                self.last_generation_["attempts"].append({**record, "raw_response": raw_response})
            if record["status"] == "success":
                return GenerationResult(topics, attempt_prompt, raw_response, metadata)
        raise RuntimeError("unreachable retry state")

    def generate_initial(self, documents: list[str]) -> GenerationResult:
        return self._run(documents, lambda block: _initial_prompt(
            [], topic_instruction=self.topic_instruction, document_block=block), prefix="T")

    def generate_residual(self, existing_codebook: list[Topic], residual_documents: list[str]) -> GenerationResult:
        return self._run(residual_documents, lambda block: _residual_prompt(
            existing_codebook, [], topic_instruction=self.topic_instruction, document_block=block), prefix="R")


class OpenAITopicGenerator(_BudgetedTopicGenerator, TopicGenerator):
    response_format_version = GENERATION_FORMAT_VERSION

    def __init__(self, *, model: str, api_key: str, topic_instruction: str | None = None,
                 context_window: int | None = None, input_context_ratio: float = 0.75,
                 max_output_tokens: int = 16384, max_format_retries: int = 1,
                 sampling_seed: int = 42, safety_margin_tokens: int = 1024,
                 provider: str = "openai", base_url: str | None = None,
                 reasoning_effort: str | None = None, tokenizer_encoding: str | None = None,
                 response_format_mode: str | None = None,
                 catalog_path: str | None = None, input_token_counter=None,
                 token_count_source: str | None = None, request_options: dict | None = None) -> None:
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            if exc.name not in ("openai", "tiktoken"):
                raise
            raise ImportError("OpenAITopicGenerator requires the openai extra: pip install '.[openai]'") from exc
        if provider not in OpenAICompatibleTopicGenerator.DEFAULT_BASE_URLS:
            raise ValueError(f"unsupported provider: {provider}")
        self.request_options = _request_options(request_options)
        native_only = {"thinking", "top_k"}.intersection(self.request_options)
        if provider == "openai" and (native_only or self.request_options.get("extra_body")):
            raise ValueError("OpenAI request_options do not support thinking/top_k or uncounted extra_body")
        for name in native_only:
            extra = self.request_options.setdefault("extra_body", {})
            if name in extra:
                raise ValueError(f"duplicate request option: {name}")
            extra[name] = self.request_options.pop(name)
        self._input_token_counter = input_token_counter
        if input_token_counter is not None:
            if not callable(input_token_counter) or not isinstance(token_count_source, str) or not token_count_source.strip():
                raise ValueError("input_token_counter requires a callable and token_count_source")
        elif provider != "openai":
            raise ValueError("input_token_counter and token_count_source are required for non-OpenAI providers; a tiktoken name is insufficient")
        limits = resolve_model_limits(
            model, provider=provider, context_window=context_window,
            max_output_tokens=max_output_tokens, tokenizer_encoding=tokenizer_encoding,
            catalog_path=catalog_path, require_tokenizer=input_token_counter is None,
        )
        self._configure(model, topic_instruction=topic_instruction, context_window=limits.context_window,
                        input_context_ratio=input_context_ratio, max_output_tokens=max_output_tokens,
                        max_format_retries=max_format_retries, sampling_seed=sampling_seed,
                        safety_margin_tokens=safety_margin_tokens, provider=provider, limits=limits,
                        reasoning_effort=reasoning_effort, tokenizer_encoding=tokenizer_encoding,
                        catalog_path=catalog_path)
        response_format_mode = response_format_mode or ("json_object" if provider == "deepseek" else "json_schema")
        if response_format_mode not in ("json_schema", "json_object"):
            raise ValueError("response_format_mode must be json_schema or json_object")
        if provider == "deepseek" and response_format_mode != "json_object":
            raise ValueError("DeepSeek supports json_object, not json_schema")
        self.response_format_mode = response_format_mode
        self.response_format_version = GENERATION_FORMAT_VERSION if response_format_mode == "json_schema" else "validated-json-fields-v1"
        self.generation_config["request_options"] = deepcopy(self.request_options)
        self.generation_config["response_format_mode"] = response_format_mode
        if input_token_counter is None:
            try:
                import tiktoken
            except ModuleNotFoundError as exc:
                raise ImportError("OpenAI token counting requires pip install '.[openai]'") from exc
            encoding_name = limits.tokenizer_encoding
            try:
                self._encoding = tiktoken.get_encoding(encoding_name) if encoding_name else tiktoken.encoding_for_model(model)
            except (KeyError, ValueError) as exc:
                raise ValueError("tokenizer_encoding could not be resolved; configure a verified tokenizer") from exc
            self.generation_config["token_count_method"] = f"tiktoken:{self._encoding.name}+schema+128_estimated_overhead"
        else:
            self.generation_config["token_count_method"] = token_count_source
        client_kwargs = {"api_key": api_key, "timeout": 900.0, "max_retries": 0}
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
        self.model = model

    def _count_input(self, prompt):
        if self._input_token_counter is not None:
            return _token_count(self._input_token_counter(self._request_kwargs(prompt)))
        # API framing/schema serialization is an estimate; the separate safety
        # margin remains available for provider-side differences.
        return sum(len(self._encoding.encode(text, disallowed_special=())) for text in
                   (_SYSTEM, prompt, json.dumps(TOPIC_RESPONSE_FORMAT))) + 128

    def _request_kwargs(self, prompt):
        kwargs = {
            **deepcopy(self.request_options),
            "model": self.model, "messages": self._messages(prompt),
            "response_format": deepcopy(TOPIC_RESPONSE_FORMAT) if self.response_format_mode == "json_schema" else {"type": "json_object"},
            ("max_completion_tokens" if self.generation_config["provider"] == "openai" else "max_tokens"): self.budget.max_output_tokens,
        }
        if self.generation_config["reasoning_effort"] is not None:
            kwargs["reasoning_effort"] = self.generation_config["reasoning_effort"]
        return kwargs

    def _request(self, prompt):
        response = self.client.chat.completions.create(**self._request_kwargs(prompt))
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        return choice.message.content or "", {
            "finish_reason": choice.finish_reason,
            "refusal": bool(getattr(choice.message, "refusal", None)),
            "response_model": getattr(response, "model", None),
            "usage": usage.model_dump() if usage is not None else None,
        }


class OpenAICompatibleTopicGenerator(OpenAITopicGenerator):
    """OpenAI-compatible adapter for OpenAI, xAI, DeepSeek and local servers."""

    DEFAULT_BASE_URLS = {
        "openai": None,
        "xai": "https://api.x.ai/v1",
        "deepseek": "https://api.deepseek.com",
        "ollama": "http://localhost:11434/v1",
        "vllm": "http://localhost:8000/v1",
    }

    def __init__(self, *, provider: str, model: str, api_key: str,
                 base_url: str | None = None, **kwargs) -> None:
        provider = provider.strip().lower()
        if provider not in self.DEFAULT_BASE_URLS:
            raise ValueError(f"unsupported OpenAI-compatible provider: {provider}")
        super().__init__(model=model, api_key=api_key, provider=provider,
                         base_url=base_url if base_url is not None else self.DEFAULT_BASE_URLS[provider],
                         **kwargs)


class AnthropicTopicGenerator(_BudgetedTopicGenerator, TopicGenerator):
    """Claude Messages adapter; counts the actual request using count_tokens."""

    response_format_version = GENERATION_FORMAT_VERSION

    def __init__(self, *, model: str, api_key: str, topic_instruction: str | None = None,
                 context_window: int | None = None, input_context_ratio: float = 0.75,
                 max_output_tokens: int = 16384, max_format_retries: int = 1,
                 sampling_seed: int = 42, safety_margin_tokens: int = 1024,
                 reasoning_effort: str | None = None, tokenizer_encoding: str | None = None,
                 client=None, catalog_path: str | None = None,
                 response_format_mode: str = "json_schema", request_options: dict | None = None) -> None:
        if tokenizer_encoding is not None:
            raise ValueError("Claude uses messages.count_tokens; remove tokenizer_encoding")
        self.request_options = _request_options(request_options)
        if "extra_body" in self.request_options:
            raise ValueError("Claude options use native SDK fields, not extra_body")
        if response_format_mode not in ("json_schema", "json"):
            raise ValueError("Claude response_format_mode must be json_schema or json")
        self.response_format_mode = response_format_mode
        self.response_format_version = GENERATION_FORMAT_VERSION if response_format_mode == "json_schema" else "validated-json-fields-v1"
        limits = resolve_model_limits(model, provider="anthropic", context_window=context_window,
            max_output_tokens=max_output_tokens, catalog_path=catalog_path, require_tokenizer=False)
        self._configure(model, topic_instruction=topic_instruction, context_window=limits.context_window,
                        input_context_ratio=input_context_ratio, max_output_tokens=max_output_tokens,
                        max_format_retries=max_format_retries, sampling_seed=sampling_seed,
                        safety_margin_tokens=safety_margin_tokens, provider="anthropic", limits=limits,
                        reasoning_effort=reasoning_effort, catalog_path=catalog_path)
        if client is None:
            try:
                from anthropic import Anthropic
            except ModuleNotFoundError as exc:
                if exc.name != "anthropic":
                    raise
                raise ImportError("AnthropicTopicGenerator requires pip install '.[anthropic]'") from exc
            client = Anthropic(api_key=api_key, max_retries=0, timeout=900.0)
        self.client = client
        self.model = model
        self.generation_config.update(token_count_method="anthropic.messages.count_tokens",
            request_options=deepcopy(self.request_options), response_format_mode=response_format_mode)
        # Validate against the installed SDK before counting or generation.
        # SDK versions and model families expose different optional fields.
        try:
            inspect.signature(self.client.messages.create).bind(**self._request_kwargs(""))
            inspect.signature(self.client.messages.count_tokens).bind(**self._count_kwargs(""))
        except TypeError as exc:
            raise ValueError(f"Claude SDK does not support this request configuration; install the anthropic extra or remove unsupported options: {exc}") from exc
        # Packing revisits the same prompt; cache only within one generation so
        # source text is not retained on a long-lived fitted model.
        self._count_cached = lru_cache(maxsize=32)(self._count_remote)

    def _run(self, *args, **kwargs):
        self._count_cached.cache_clear()
        try:
            return super()._run(*args, **kwargs)
        finally:
            self._count_cached.cache_clear()

    def _request_kwargs(self, prompt):
        kwargs = {**deepcopy(self.request_options), "model": self.model, "system": _SYSTEM,
                  "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": self.budget.max_output_tokens}
        output = {}
        if self.response_format_mode == "json_schema":
            output["format"] = {"type": "json_schema", "schema": deepcopy(TOPIC_RESPONSE_FORMAT["json_schema"]["schema"])}
        if self.generation_config["reasoning_effort"] is not None:
            output["effort"] = self.generation_config["reasoning_effort"]
        if output:
            kwargs["output_config"] = output
        return kwargs

    def _count_kwargs(self, prompt):
        request = self._request_kwargs(prompt)
        # Count endpoint has a smaller schema: sampling/output caps are absent.
        return {k:v for k,v in request.items() if k in ("model", "system", "messages", "thinking", "output_config")}

    def _count_remote(self, prompt):
        return _token_count(self.client.messages.count_tokens(**self._count_kwargs(prompt)).input_tokens)

    def _count_input(self, prompt):
        return self._count_cached(prompt)

    def _request(self, prompt):
        response = self.client.messages.create(**self._request_kwargs(prompt))
        text = "".join(block.text for block in getattr(response, "content", ()) if getattr(block, "type", None) == "text")
        usage = getattr(response, "usage", None)
        stop_reason = getattr(response, "stop_reason", None)
        return text, {
            "finish_reason": "stop" if stop_reason == "end_turn" else (stop_reason or "unknown"),
            "refusal": stop_reason == "refusal", "response_model": getattr(response, "model", None),
            "usage": usage.model_dump() if hasattr(usage, "model_dump") else getattr(usage, "__dict__", None),
        }


class TransformersTopicGenerator(_BudgetedTopicGenerator, TopicGenerator):
    """Local Transformers causal-LM adapter with config-derived context."""

    response_format_version = "validated-json-fields-v1"

    def __init__(self, model: str, *, topic_instruction: str | None = None,
                 context_window: int | None = None, input_context_ratio: float = 0.75,
                 max_output_tokens: int = 16384, max_format_retries: int = 1,
                 sampling_seed: int = 42, safety_margin_tokens: int = 1024,
                 model_kwargs: dict[str, object] | None = None) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as exc:
            if exc.name != "transformers":
                raise
            raise ImportError("TransformersTopicGenerator requires the local extra: pip install '.[transformers]'") from exc
        limits = resolve_local_model_limits(model, context_window=context_window)
        self._configure(model, topic_instruction=topic_instruction, context_window=limits.context_window,
                        input_context_ratio=input_context_ratio, max_output_tokens=max_output_tokens,
                        max_format_retries=max_format_retries, sampling_seed=sampling_seed,
                        safety_margin_tokens=safety_margin_tokens, provider="transformers", limits=limits)
        kwargs = dict(model_kwargs or {})
        self.tokenizer = AutoTokenizer.from_pretrained(model, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs)
        self.generation_config["token_count_method"] = "transformers_tokenizer_chat_template"

    def _formatted(self, prompt):
        return self.tokenizer.apply_chat_template(self._messages(prompt), tokenize=False, add_generation_prompt=True)

    def _count_input(self, prompt):
        return len(self.tokenizer.encode(self._formatted(prompt), add_special_tokens=False))

    def _request(self, prompt):
        import torch
        encoded = self.tokenizer(self._formatted(prompt), return_tensors="pt", add_special_tokens=False)
        device = getattr(self.model, "device", None)
        if device is not None:
            encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            generated = self.model.generate(**encoded, max_new_tokens=self.budget.max_output_tokens,
                                            do_sample=False)
        prompt_len = encoded["input_ids"].shape[-1]
        new_tokens = generated[0][prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        eos = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        if eos is None:
            eos = self.tokenizer.eos_token_id
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        stopped = len(new_tokens) > 0 and int(new_tokens[-1]) in eos
        finish = "stop" if stopped else ("length" if len(new_tokens) >= self.budget.max_output_tokens else "unknown")
        return text, {"finish_reason": finish, "refusal": False, "response_model": self.generation_config["model"],
                      "usage": {"prompt_tokens": prompt_len, "completion_tokens": int(generated.shape[-1] - prompt_len)}}


class MLXTopicGenerator(_BudgetedTopicGenerator, TopicGenerator):
    response_format_version = "validated-json-fields-v1"

    def __init__(self, model: str, *, topic_instruction: str | None = None,
                 context_window: int | None = None, input_context_ratio: float = 0.75,
                 max_output_tokens: int = 16384, max_format_retries: int = 1,
                 sampling_seed: int = 42, safety_margin_tokens: int = 1024) -> None:
        try:
            from mlx_lm import load
        except ModuleNotFoundError as exc:
            if exc.name != "mlx_lm":
                raise
            raise ImportError("MLXTopicGenerator requires the mlx extra: pip install '.[mlx]'") from exc
        limits = resolve_local_model_limits(model, context_window=context_window)
        self._configure(model, topic_instruction=topic_instruction, context_window=limits.context_window,
                        input_context_ratio=input_context_ratio, max_output_tokens=max_output_tokens,
                        max_format_retries=max_format_retries, sampling_seed=sampling_seed,
                        safety_margin_tokens=safety_margin_tokens, provider="mlx", limits=limits)
        self.model, self.tokenizer = load(model)
        self.generation_config["token_count_method"] = "local_tokenizer_with_chat_template"
        self.generation_config["temperature"] = 0.7

    def _formatted(self, prompt):
        return self.tokenizer.apply_chat_template(self._messages(prompt), tokenize=False, add_generation_prompt=True)

    def _prompt_tokens(self, prompt):
        formatted = self._formatted(prompt)
        bos = getattr(self.tokenizer, "bos_token", None)
        return self.tokenizer.encode(formatted, add_special_tokens=bos is None or not formatted.startswith(bos))

    def _count_input(self, prompt):
        return len(self._prompt_tokens(prompt))

    def _request(self, prompt):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        parts = []
        final = None
        # Pass exactly the token IDs used for budgeting to the generation API.
        for final in stream_generate(self.model, self.tokenizer, prompt=self._prompt_tokens(prompt),
                                     max_tokens=self.budget.max_output_tokens, sampler=make_sampler(temp=0.7)):
            parts.append(final.text)
        return "".join(parts), {
            "finish_reason": final.finish_reason if final is not None else "empty",
            "usage": {"prompt_tokens": final.prompt_tokens, "completion_tokens": final.generation_tokens} if final else None,
            "peak_memory_gb": final.peak_memory if final else None,
        }
