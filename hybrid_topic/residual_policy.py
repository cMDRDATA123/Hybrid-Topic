"""Residual stopping rules retained without the retired research selector."""
from dataclasses import dataclass

@dataclass(frozen=True)
class ResidualConfig:
    """Residual discovery loop settings.

    The public model supplies its explicit 0/1/2 round limit.
    """

    max_rounds: int | None
    min_residual_docs: int = 2
    no_progress_patience: int = 2
    pass_codebook_context: bool = True

    def __post_init__(self) -> None:
        if self.max_rounds is not None and self.max_rounds < 0:
            raise ValueError("max_rounds must be None or non-negative")
        if self.min_residual_docs != 2:
            raise ValueError("the frozen protocol requires at least two residual documents")
        if self.no_progress_patience != 2:
            raise ValueError("the frozen protocol stops after two non-progressing residual rounds")
        if not self.pass_codebook_context:
            raise ValueError("residual discovery must pass the current codebook as context")



def residual_stop_reason(
    *,
    completed_rounds: int,
    residual_doc_count: int,
    generated_topic_count: int | None,
    valid_new_topic_count: int | None,
    consecutive_no_progress: int,
    config: ResidualConfig,
) -> str | None:
    """Return the frozen natural stop reason, or ``None`` when another round may run."""

    if residual_doc_count < config.min_residual_docs:
        return "residual_too_small"
    if config.max_rounds is not None and completed_rounds >= config.max_rounds:
        return "max_residual_rounds_reached"
    if generated_topic_count is not None and valid_new_topic_count is not None and valid_new_topic_count == 0:
        return "no_new_topics" if generated_topic_count == 0 else "only_duplicate_or_unsupported_topics"
    if consecutive_no_progress >= config.no_progress_patience:
        return "no_progress_patience_exhausted"
    return None

