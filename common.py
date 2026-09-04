"""Shared, side-effect-free helpers used by the RelaCaTS-v2 pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


CONFIDENCE_SUFFIXES = {
    "llama": (
        "<|eot_id|><|start_header_id|>user<|end_header_id|>"
        "Is the answer correct? (Yes/No)<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
    "qwen": (
        "<|im_end|>\n<|im_start|>user\n"
        "Is the answer correct? (Yes/No)<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    "deepseek": (
        "<｜end▁of▁sentence｜><｜User｜>Directly answer the question by Yes or No:"
        "Is the answer correct?<｜Assistant｜>"
    ),
}


GENERATION_DEFAULTS = {
    "llama": {"max_new_tokens": 1024, "max_model_len": 8192},
    "qwen": {"max_new_tokens": 1024, "max_model_len": 8192},
    "deepseek": {"max_new_tokens": 2048, "max_model_len": 8192},
}


def model_family(model_name: str) -> str:
    """Return the chat-template family without the original repo's branch bug."""

    name = model_name.lower()
    # DeepSeek distill paths often contain the word ``qwen``; check first.
    if "deepseek" in name or "/ds" in name or name.startswith("ds"):
        return "deepseek"
    if "llama" in name:
        return "llama"
    if "qwen" in name:
        return "qwen"
    raise ValueError(
        f"Unsupported model family for {model_name!r}; expected Llama, Qwen, or DeepSeek"
    )


def confidence_suffix(model_name: str) -> str:
    return CONFIDENCE_SUFFIXES[model_family(model_name)]


def generation_defaults(model_name: str) -> dict[str, int]:
    """Return a copy of the audited generation limits for a model family."""

    return dict(GENERATION_DEFAULTS[model_family(model_name)])


def remove_forced_think_from_chat_template(template: str | None) -> str | None:
    """Remove only DeepSeek's automatic *generation-prompt* ``<think>``.

    This deliberately leaves ordinary message content and model-generated
    ``<think>...</think>`` blocks untouched.  Both escaped and literal newline
    spellings occur in tokenizer files written by different Transformers
    versions.
    """

    if template is None:
        return None
    normalized = template.replace("<｜Assistant｜><think>\\n", "<｜Assistant｜>")
    normalized = normalized.replace("<｜Assistant｜><think>\n", "<｜Assistant｜>")
    return normalized


def remove_forced_think_from_prompt(prompt: str) -> str:
    """Strip an auto-inserted trailing DeepSeek think opener from a prompt."""

    return re.sub(r"(<｜Assistant｜>)<think>[ \t]*(?:\\n|\n)?\s*\Z", r"\1", prompt)


def confidence_from_logprobs(logprobs: Any) -> tuple[float, float, bool, bool]:
    """Extract the original CaTS P(Yes) and diagnostic P(No) from vLLM top-k.

    CaTS uses the unnormalised probability mass of tokens whose decoded form
    contains ``Yes``.  We preserve that definition rather than normalising by
    Yes+No.
    """

    if not logprobs:
        raise RuntimeError("vLLM returned no next-token log probabilities")
    yes_prob = 0.0
    no_prob = 0.0
    yes_found = False
    no_found = False
    values = logprobs.values() if hasattr(logprobs, "values") else logprobs
    for item in values:
        decoded = str(getattr(item, "decoded_token", "") or "")
        probability = math.exp(float(getattr(item, "logprob")))
        if "Yes" in decoded:
            yes_prob += probability
            yes_found = True
        if "No" in decoded:
            no_prob += probability
            no_found = True
    return yes_prob, no_prob, yes_found, no_found


def build_reasoning_prompt(tokenizer: Any, question: str, answer_type: str) -> str:
    """Build the same CoT response format used by the original CaTS repo."""

    demo = "(A)" if answer_type == "option letter" else "1"
    instruction = (
        "For the following question, provide a step-by-step explanation of your "
        "thought process.\n"
        "Use the format demonstrated below for your response.\n"
        "```Example Format:\n"
        "Explanation: <Your detailed explanation here, outlining how you arrived at "
        "your answer.>\n"
        f"Answer: <Insert your concise answer here, which should include a "
        f"{answer_type} (e.g., {demo})>\n"
        "Ensure that your response strictly adheres to this format. Explicitly "
        "include the words 'Explanation:', 'Answer:'."
    )
    chat = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": f"Question: {question}"},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        rendered = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        return remove_forced_think_from_prompt(rendered)
    return f"{instruction}\n\nQuestion: {question}\nAnswer:"


def stable_id(*parts: object, length: int = 16) -> str:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def atomic_write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield value


def validate_or_write_metadata(path: str | Path, requested: dict[str, Any]) -> None:
    path = Path(path)
    if path.exists():
        existing = read_json(path)
        if existing != requested:
            raise RuntimeError(
                f"Existing metadata differs at {path}. Use a new output directory.\n"
                f"existing={existing}\nrequested={requested}"
            )
        return
    atomic_write_json(path, requested)


def batched(sequence: Sequence[Any], batch_size: int) -> Iterator[Sequence[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(sequence), batch_size):
        yield sequence[start : start + batch_size]
