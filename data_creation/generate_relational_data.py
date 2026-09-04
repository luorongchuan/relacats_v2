"""Generate RelaCaTS-v2 teacher data with a fixed 32-response budget.

The output is deliberately question-sharded: each original question is one
atomic JSON file.  Two independent workers can therefore use physical GPUs 6
and 7 without concurrently appending to (and corrupting) one large JSON file.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any, Sequence

from relacats_v2.common import (
    atomic_write_json,
    batched,
    build_reasoning_prompt,
    confidence_from_logprobs,
    confidence_suffix,
    generation_defaults,
    read_json,
    read_jsonl,
    stable_id,
    validate_or_write_metadata,
)
from relacats_v2.core import (
    canonicalize_answer,
    generate_identity_views,
    generate_option_permutation_views,
)
from relacats_v2.data_creation import dataset_adapter as _dataset_adapter
from relacats_v2.evaluation.answer_parsing import extract_explicit_answer

MCQExample = _dataset_adapter.MCQExample
SUPPORTED_DATASETS = tuple(_dataset_adapter.SUPPORTED_DATASETS)
NUMERIC_DATASETS = tuple(
    getattr(_dataset_adapter, "NUMERIC_DATASETS", ("gsm8k", "svamp"))
)
load_mcq_examples = _dataset_adapter.load_mcq_examples
load_dataset_examples = getattr(_dataset_adapter, "load_dataset_examples", None)


LOGGER = logging.getLogger("relacats.generate")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "/home/luorongchuan/workspace_135/models/Qwen2.5-7B-Instruct"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "relacats_v2/outputs/generated_data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-questions", type=int, default=1000)
    parser.add_argument(
        "--candidate-file",
        help=(
            "Optional diagnosis JSONL. When provided, online datasets are not loaded; "
            "records need question_id, dataset_name, stem/original_question and options."
        ),
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--samples-per-view", type=int, default=8)
    parser.add_argument("--total-budget", type=int, default=32)
    parser.add_argument(
        "--relation-mode",
        choices=("auto", "option_permutation", "identity"),
        default="auto",
        help=(
            "Relation profile. auto selects numeric identity-only for GSM8K/SVAMP "
            "and option permutations for MCQ datasets."
        ),
    )
    parser.add_argument(
        "--allow-nonstandard-budget",
        action="store_true",
        help=(
            "Permit a non-32 smoke budget. Formal ordinary MCQ uses 4x8, "
            "WinoGrande uses 2x16, and formal numeric identity uses 1x32."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument(
        "--confidence-temperature",
        type=float,
        default=0.0,
        help="Temperature for the one-token confidence query (original CaTS uses 0).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Defaults to 2048 for DeepSeek and 1024 for Qwen/Llama",
    )
    parser.add_argument("--max-model-len", type=int, help="Defaults to 8192")
    parser.add_argument("--question-batch-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    defaults = generation_defaults(args.model_name)
    if args.max_new_tokens is None:
        args.max_new_tokens = defaults["max_new_tokens"]
    if args.max_model_len is None:
        args.max_model_len = defaults["max_model_len"]
    return args


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _formal_option_profiles(args: argparse.Namespace) -> set[tuple[int, int, int]]:
    """Profiles accepted by a formal option-only invocation.

    Mixed invocations keep the ordinary 4x8 CLI defaults and dispatch
    WinoGrande to its dataset-specific 2x16 profile internally.  A Wino-only
    invocation may also spell out 2x16 explicitly for callers that do not use
    the mixed-dataset wrapper.
    """

    profiles = {(4, 8, 32)}
    datasets = tuple(getattr(args, "datasets", ()) or ())
    if datasets and all(str(name).strip().lower() == "winogrande" for name in datasets):
        profiles.add((2, 16, 32))
    return profiles


def validate_args(args: argparse.Namespace) -> None:
    if not args.candidate_file and not args.datasets:
        raise ValueError("Provide --datasets or --candidate-file")
    if args.candidate_file and args.datasets:
        raise ValueError("--candidate-file and --datasets are mutually exclusive")
    for field in (
        "max_questions",
        "num_views",
        "samples_per_view",
        "total_budget",
        "max_new_tokens",
        "max_model_len",
        "question_batch_size",
        "num_shards",
    ):
        if int(getattr(args, field)) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    actual_budget = args.num_views * args.samples_per_view
    if actual_budget != args.total_budget:
        raise ValueError(
            f"num_views*samples_per_view={actual_budget}, not total_budget={args.total_budget}"
        )
    if not args.allow_nonstandard_budget:
        if args.relation_mode == "identity":
            expected = (1, 32, 32)
            if (args.num_views, args.samples_per_view, args.total_budget) != expected:
                raise ValueError(
                    "Formal numeric identity mode must use 1 view x 32 responses. "
                    f"Got {args.num_views}x{args.samples_per_view}={args.total_budget}."
                )
        elif args.relation_mode == "option_permutation":
            actual_profile = (args.num_views, args.samples_per_view, args.total_budget)
            if actual_profile not in _formal_option_profiles(args):
                raise ValueError(
                    "Formal option-permutation mode must use 4x8=32, or "
                    "WinoGrande-only 2x16=32. "
                    f"Got {args.num_views}x{args.samples_per_view}={args.total_budget}."
                )
        elif (args.num_views, args.samples_per_view, args.total_budget) not in _formal_option_profiles(args):
            # auto is intentionally the default for mixed datasets.  Numeric
            # examples override this profile to 1x32.  Ordinary MCQ examples
            # use 4x8, while a WinoGrande-only invocation may use 2x16.
            raise ValueError(
                "Formal auto mode expects MCQ 4x8=32 (or WinoGrande-only 2x16=32). "
                "Use --relation-mode identity for a numeric-only 1x32 run, or "
                "--allow-nonstandard-budget for smoke."
            )
    if args.relation_mode == "identity" and args.datasets:
        non_numeric = [name for name in args.datasets if name not in NUMERIC_DATASETS]
        if non_numeric:
            raise ValueError(
                "identity relation mode is only valid for numeric datasets; "
                f"got {non_numeric}"
            )
    if args.relation_mode == "option_permutation" and args.datasets:
        numeric = [name for name in args.datasets if name in NUMERIC_DATASETS]
        if numeric:
            raise ValueError(
                "option_permutation mode cannot be used for numeric datasets; "
                f"got {numeric}"
            )
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("--gpu-memory-utilization must be in (0, 1)")
    if args.temperature < 0:
        raise ValueError("--temperature must be non-negative")
    if args.confidence_temperature is not None and args.confidence_temperature < 0:
        raise ValueError("--confidence-temperature must be non-negative")
    if args.max_new_tokens >= args.max_model_len:
        raise ValueError("--max-new-tokens must be smaller than --max-model-len")


def _stem_without_rendered_options(question: str) -> str:
    """Avoid duplicating an option block in diagnosis candidate records."""

    text = str(question).strip()
    match = re.search(r"\n\s*Options\s*:\s*\n", text, flags=re.IGNORECASE)
    return text[: match.start()].rstrip() if match else text


def load_candidate_examples(path: Path, default_split: str) -> list[MCQExample]:
    examples: list[MCQExample] = []
    for index, record in enumerate(read_jsonl(path)):
        options = record.get("options")
        if not isinstance(options, Sequence) or isinstance(options, (str, bytes)):
            raise ValueError(f"Candidate {index} needs an options list")
        options_tuple = tuple(str(value).strip() for value in options)
        stem = record.get("stem", record.get("question_stem"))
        if stem is None:
            stem = _stem_without_rendered_options(record.get("original_question", ""))
        gold = record.get("gold_original_answer", record.get("correct_answer"))
        labels = tuple(chr(65 + offset) for offset in range(len(options_tuple)))
        gold_text = str(gold).strip().upper()
        if gold_text not in labels:
            raise ValueError(
                f"Candidate {index} gold answer {gold!r} is outside {labels!r}"
            )
        dataset_name = str(record.get("dataset_name", "diagnosis")).strip()
        source_index = int(record.get("source_index", index))
        question_id = str(
            record.get(
                "question_id",
                f"{dataset_name}:{default_split}:{source_index}:"
                f"{stable_id(stem, *options_tuple, length=12)}",
            )
        )
        examples.append(
            MCQExample(
                dataset_name=dataset_name,
                split=str(record.get("split", default_split)),
                source_index=source_index,
                question_id=question_id,
                stem=str(stem),
                options=options_tuple,
                correct_index=labels.index(gold_text),
            )
        )
    if not examples:
        raise ValueError(f"No candidate questions found in {path}")
    return examples


def extract_option_answer(response: str, number_of_options: int) -> str | None:
    """Extract an explicit final option without guessing from reasoning text."""

    labels = tuple(chr(65 + index) for index in range(number_of_options))
    answer = extract_explicit_answer(
        "generated_mcq", str(response), answer_type="option letter"
    )
    token = str(answer).upper() if answer is not None else ""
    return token if token in labels else None


def extract_numeric_answer(response: str) -> str | None:
    """Extract one scalar only from an explicit final-answer field."""

    answer = extract_explicit_answer(
        "generated_numeric", str(response), answer_type="number"
    )
    return str(answer) if answer is not None else None


def _example_answer_type(example: Any) -> str:
    value = getattr(example, "answer_type", None)
    if value is None:
        return "option"
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"number", "numeric", "scalar"}:
        return "number"
    return "option letter"


def _example_relation_mode(example: Any, args: argparse.Namespace) -> str:
    answer_type = _example_answer_type(example)
    requested = str(getattr(args, "relation_mode", "auto")).strip().lower()
    if answer_type == "number":
        if requested == "option_permutation":
            raise ValueError(
                f"{getattr(example, 'dataset_name', '<example>')} is numeric; "
                "option_permutation relation is undefined"
            )
        return "identity"
    if requested == "identity":
        raise ValueError(
            f"{getattr(example, 'dataset_name', '<example>')} is option-MCQ; "
            "identity relation mode is only for numeric examples"
        )
    return "option_permutation"


def _sampling_profile_for_example(
    example: Any, args: argparse.Namespace
) -> tuple[str, int, int, int]:
    """Return the effective ``(mode, views, per_view, budget)`` profile.

    The command-line defaults describe the ordinary MCQ profile (4x8=32),
    because one invocation can contain several datasets.  WinoGrande is the
    one built-in exception: with two answer choices it uses the two unique
    permutations only, allocating 16 responses to identity and 16 to swap.
    In the nonstandard smoke path we preserve the same two-view shape while
    scaling the per-view count to the requested total budget.
    """

    mode = _example_relation_mode(example, args)
    allow_nonstandard = bool(getattr(args, "allow_nonstandard_budget", False))
    if mode == "identity":
        budget = int(args.total_budget) if allow_nonstandard else 32
        return mode, 1, budget, budget

    dataset_name = str(getattr(example, "dataset_name", "")).strip().lower()
    if dataset_name == "winogrande":
        if allow_nonstandard:
            total_budget = int(args.total_budget)
            if total_budget % 2:
                raise ValueError(
                    "WinoGrande nonstandard budget must be divisible by two "
                    "(identity and swapped-options views)"
                )
            return mode, 2, total_budget // 2, total_budget
        return mode, 2, 16, 32

    return (
        mode,
        int(args.num_views),
        int(args.samples_per_view),
        int(args.total_budget),
    )


def _views_for_example(example: Any, args: argparse.Namespace) -> tuple[Any, ...]:
    """Construct the appropriate relation profile for one example."""

    mode, num_views, samples_per_view, total_budget = _sampling_profile_for_example(
        example, args
    )
    question_seed = args.seed + int(stable_id(example.question_id, length=8), 16)
    if mode == "identity":
        return generate_identity_views(
            example.stem,
            samples_per_view=samples_per_view,
            total_budget=total_budget,
            seed=question_seed,
            answer_type="number",
            allow_nonstandard_budget=bool(
                getattr(args, "allow_nonstandard_budget", False)
            ),
        )
    return generate_option_permutation_views(
        example.stem,
        example.options,
        num_views=num_views,
        samples_per_view=samples_per_view,
        total_budget=total_budget,
        seed=question_seed,
        # WinoGrande's two-view profile is always unique.  Do not infer a
        # repeat permission from legacy example metadata.
        allow_repeated_views=False,
    )


def _question_path(output_root: Path, example: Any) -> Path:
    filename = f"{example.source_index:06d}_{stable_id(example.question_id)}.json"
    return output_root / example.dataset_name / "questions" / filename


def _validate_generation_metadata(path: Path, requested: dict[str, Any]) -> None:
    """Validate metadata while allowing legacy all-MCQ generation manifests.

    Version-1 manifests predate per-dataset relation profiles.  They are safe
    to resume only when the requested run contains no numeric identity dataset;
    a mixed/changed profile must use a new output root rather than silently
    reusing option-only artifacts.
    """

    if not path.exists():
        validate_or_write_metadata(path, requested)
        return
    existing = read_json(path)
    if not isinstance(existing, dict):
        raise RuntimeError(f"Existing generation metadata is not a JSON object: {path}")
    if existing == requested:
        return
    old_schema = str(existing.get("schema_version", ""))
    requested_profiles = requested.get("dataset_profiles", {})
    has_numeric = any(
        str(profile.get("answer_type", "option")) == "number"
        for profile in requested_profiles.values()
        if isinstance(profile, dict)
    )
    if old_schema.endswith("generation-config.1") and not has_numeric:
        # Compare every field that existed in v1; newly added profile fields
        # are informational and do not invalidate an otherwise resumable MCQ
        # run.
        mismatches = {
            key: (existing.get(key), requested.get(key))
            for key in existing
            if key in requested and existing.get(key) != requested.get(key)
        }
        if not mismatches:
            LOGGER.warning(
                "Resuming legacy all-MCQ generation metadata at %s; "
                "per-dataset profile fields are not rewritten",
                path,
            )
            return
    raise RuntimeError(
        f"Existing generation metadata differs at {path}. Use a new output directory.\n"
        f"existing={existing}\nrequested={requested}"
    )


def _sampling_params(cls: Any, **kwargs: Any) -> Any:
    """Keep vLLM construction in one testable seam."""

    return cls(**kwargs)


def _generate_question_batch_uniform(
    *,
    examples: Sequence[Any],
    llm: Any,
    tokenizer: Any,
    sampling_params_cls: Any,
    model_name: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Generate one homogeneous relation profile and score its responses.

    vLLM's ``n`` applies to every prompt in a call.  Numeric identity questions
    therefore cannot share a call with option-MCQ questions (1x32 versus 4x8);
    the public wrapper below groups mixed batches before reaching this helper.
    """

    all_views: list[tuple[Any, Any]] = []
    generation_prompts: list[str] = []
    for example in examples:
        views = _views_for_example(example, args)
        for view in views:
            all_views.append((example, view))
            answer_type = getattr(view, "answer_type", _example_answer_type(example))
            generation_prompts.append(
                build_reasoning_prompt(tokenizer, view.transformed_question, answer_type)
            )

    if not all_views:
        return []
    samples_per_view = int(all_views[0][1].samples_per_view)
    if any(int(view.samples_per_view) != samples_per_view for _, view in all_views):
        raise ValueError("generate_question_batch received mixed sampling profiles")

    generation = _sampling_params(
        sampling_params_cls,
        n=samples_per_view,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )
    generated = llm.generate(generation_prompts, generation, use_tqdm=False)
    if len(generated) != len(all_views):
        raise RuntimeError("vLLM returned a different number of prompt outputs")

    provisional: list[dict[str, Any]] = []
    confidence_prompts: list[str] = []
    suffix = confidence_suffix(model_name)
    for prompt_index, ((example, view), request_output) in enumerate(
        zip(all_views, generated)
    ):
        if len(request_output.outputs) != samples_per_view:
            raise RuntimeError(
                f"{example.question_id}/{view.relation_id}: expected "
                f"{samples_per_view} generations, got {len(request_output.outputs)}"
            )
        prompt = generation_prompts[prompt_index]
        metadata = view.to_metadata()
        answer_type = getattr(view, "answer_type", _example_answer_type(example))
        options = tuple(getattr(example, "options", ()) or ())
        option_permutation = getattr(view, "option_permutation", None)
        for sample_index, candidate in enumerate(request_output.outputs):
            response = str(candidate.text)
            if answer_type == "number":
                extracted = extract_numeric_answer(response)
                canonical = canonicalize_answer(
                    extracted, metadata, answer_type="number"
                )
            else:
                extracted = extract_option_answer(response, len(options))
                canonical = canonicalize_answer(extracted, metadata, answer_type="option")
            record: dict[str, Any] = {
                "sample_id": stable_id(
                    example.question_id, view.relation_id, sample_index, length=24
                ),
                "question_id": example.question_id,
                "source_index": example.source_index,
                "dataset_name": example.dataset_name,
                "split": example.split,
                "relation_type": view.relation_type,
                "relation_mode": getattr(view, "relation_mode", "option_permutation"),
                "answer_type": answer_type,
                "is_duplicate_view": bool(getattr(view, "is_duplicate_view", False)),
                "relation_id": view.relation_id,
                "view_index": int(view.relation_id.removeprefix("g")),
                "sample_index_in_view": sample_index,
                "original_question": view.original_question,
                "transformed_question": view.transformed_question,
                "original_prompt": build_reasoning_prompt(
                    tokenizer, view.original_question, answer_type
                ),
                "transformed_prompt": prompt,
                "option_labels": list(option_permutation.labels)
                if option_permutation is not None
                else [],
                "original_options": list(view.original_options),
                "transformed_options": list(view.transformed_options),
                "permutation": view.permutation if option_permutation is not None else None,
                "inverse_permutation": (
                    view.inverse_permutation if option_permutation is not None else None
                ),
                "response": response,
                "gold_original_answer": example.correct_answer,
                "relation_weight": 1.0,
                "dependency_weight": 1.0,
                "finish_reason": getattr(candidate, "finish_reason", None),
                "generated_token_count": len(getattr(candidate, "token_ids", []) or []),
            }
            record.update(canonical.to_record_fields())
            provisional.append(record)
            confidence_prompts.append(f"{prompt}{response} {suffix}")

    confidence_temperature = float(args.confidence_temperature)
    confidence_sampling = _sampling_params(
        sampling_params_cls,
        max_tokens=1,
        temperature=confidence_temperature,
        logprobs=20,
        seed=args.seed,
    )
    confidence_outputs = llm.generate(
        confidence_prompts, confidence_sampling, use_tqdm=False
    )
    if len(confidence_outputs) != len(provisional):
        raise RuntimeError("vLLM returned a different number of confidence outputs")
    for record, output in zip(provisional, confidence_outputs):
        candidate = output.outputs[0]
        if not candidate.logprobs or not candidate.logprobs[0]:
            raise RuntimeError(f"No confidence logprobs for {record['sample_id']}")
        yes, no, yes_found, no_found = confidence_from_logprobs(
            candidate.logprobs[0]
        )
        record.update(
            {
                "confidence": yes,
                "true_prob": yes,
                "false_prob": no,
                "yes_token_found": yes_found,
                "no_token_found": no_found,
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = {
        example.question_id: [] for example in examples
    }
    for record in provisional:
        grouped[record["question_id"]].append(record)

    payloads: list[dict[str, Any]] = []
    for example in examples:
        samples = grouped[example.question_id]
        valid_count = sum(bool(sample["is_valid_answer"]) for sample in samples)
        payloads.append(
            {
                "schema_version": "relacats-v1.raw-question.1",
                "question_id": example.question_id,
                "source_index": example.source_index,
                "dataset_name": example.dataset_name,
                "split": example.split,
                "original_question": samples[0]["original_question"],
                "original_options": list(getattr(example, "options", ()) or ()),
                "gold_original_answer": example.correct_answer,
                "answer_type": _example_answer_type(example),
                "relation_mode": samples[0].get(
                    "relation_mode", "option_permutation"
                ),
                "allow_repeated_views": any(
                    bool(sample.get("is_duplicate_view", False)) for sample in samples
                ),
                "num_views": len({sample["relation_id"] for sample in samples}),
                "samples_per_view": samples_per_view,
                "attempted_budget": len(samples),
                "valid_response_count": valid_count,
                "invalid_response_count": len(samples) - valid_count,
                "samples": samples,
            }
        )
    return payloads


def generate_question_batch(
    *,
    examples: Sequence[Any],
    llm: Any,
    tokenizer: Any,
    sampling_params_cls: Any,
    model_name: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Generate a possibly mixed batch without mixing vLLM ``n`` profiles."""

    if not examples:
        return []
    groups: dict[tuple[str, str, int, int, int], list[Any]] = {}
    for example in examples:
        mode, num_views, samples_per_view, total_budget = (
            _sampling_profile_for_example(example, args)
        )
        # vLLM's ``n`` is the per-view count.  Include the complete effective
        # profile so WinoGrande (2x16 formal) cannot be mixed with an ordinary
        # MCQ group (4x8), even though both calls have the same total budget.
        key = (
            _example_answer_type(example),
            mode,
            num_views,
            samples_per_view,
            total_budget,
        )
        groups.setdefault(key, []).append(example)

    by_question_id: dict[str, dict[str, Any]] = {}
    for group in groups.values():
        for payload in _generate_question_batch_uniform(
            examples=group,
            llm=llm,
            tokenizer=tokenizer,
            sampling_params_cls=sampling_params_cls,
            model_name=model_name,
            args=args,
        ):
            question_id = payload["question_id"]
            if question_id in by_question_id:
                raise ValueError(f"duplicate question_id in generation batch: {question_id}")
            by_question_id[question_id] = payload
    return [by_question_id[example.question_id] for example in examples]


def load_examples(args: argparse.Namespace) -> list[Any]:
    if args.candidate_file:
        examples = load_candidate_examples(resolve_path(args.candidate_file), args.split)
        if args.max_questions is not None:
            examples = examples[: args.max_questions]
        return examples
    result: list[Any] = []
    for dataset_name in args.datasets:
        if load_dataset_examples is not None:
            result.extend(
                load_dataset_examples(dataset_name, args.split, args.max_questions)
            )
        elif dataset_name in NUMERIC_DATASETS:
            raise RuntimeError(
                "The dataset adapter does not expose load_dataset_examples; "
                f"numeric dataset {dataset_name!r} needs the updated adapter"
            )
        else:
            # Keep compatibility with older adapters that only expose the
            # original MCQ loader.
            result.extend(
                load_mcq_examples(dataset_name, args.split, args.max_questions)
            )
    return result


def main() -> None:
    args = parse_args()
    validate_args(args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    output_root = resolve_path(args.output_root)
    examples = load_examples(args)
    dataset_profiles: dict[str, dict[str, Any]] = {}
    for example in examples:
        mode, num_views, samples_per_view, total_budget = (
            _sampling_profile_for_example(example, args)
        )
        answer_type = _example_answer_type(example)
        if mode == "identity":
            profile = {
                "answer_type": "number",
                "relation_mode": "identity_only",
                "num_views": 1,
                "samples_per_view": samples_per_view,
                "total_budget": total_budget,
            }
        else:
            profile = {
                "answer_type": answer_type,
                "relation_mode": "option_permutation",
                "num_views": num_views,
                "samples_per_view": samples_per_view,
                "total_budget": total_budget,
                "allow_repeated_views": False,
            }
        previous = dataset_profiles.setdefault(example.dataset_name, profile)
        if previous != profile:
            raise ValueError(
                f"Inconsistent relation profiles for dataset {example.dataset_name!r}: "
                f"{previous!r} versus {profile!r}"
            )
    selected = [
        example
        for global_index, example in enumerate(examples)
        if global_index % args.num_shards == args.shard_index
    ]
    metadata = {
        # Keep the v1 schema name for readers that only inspect the original
        # fields; the optional dataset_profiles field carries the mixed-mode
        # extension and is validated by _validate_generation_metadata.
        "schema_version": "relacats-v1.generation-config.1",
        "model_name": str(Path(args.model_name).expanduser()),
        "datasets": list(args.datasets or ["<candidate-file>"]),
        "split": args.split,
        "max_questions_per_dataset": args.max_questions,
        "candidate_file": str(resolve_path(args.candidate_file))
        if args.candidate_file
        else None,
        "num_views": args.num_views,
        "samples_per_view": args.samples_per_view,
        "total_budget": args.total_budget,
        "temperature": args.temperature,
        # ``None`` means reuse generation temperature (the CaTS default).
        "confidence_temperature": float(
            args.temperature
            if args.confidence_temperature is None
            else args.confidence_temperature
        ),
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "seed": args.seed,
        "relation_mode": args.relation_mode,
        "dataset_profiles": dataset_profiles,
    }
    _validate_generation_metadata(output_root / "generation_metadata.json", metadata)

    pending = [example for example in selected if not _question_path(output_root, example).exists()]
    LOGGER.info(
        "worker %d/%d owns %d questions (%d pending, %d resumed)",
        args.shard_index,
        args.num_shards,
        len(selected),
        len(pending),
        len(selected) - len(pending),
    )
    if not pending:
        return

    # Imports are intentionally delayed so dataset construction and CPU unit
    # tests do not require a working vLLM CUDA runtime.
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=False,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        trust_remote_code=args.trust_remote_code,
    )

    completed = 0
    for question_batch in batched(pending, args.question_batch_size):
        payloads = generate_question_batch(
            examples=question_batch,
            llm=llm,
            tokenizer=tokenizer,
            sampling_params_cls=SamplingParams,
            model_name=args.model_name,
            args=args,
        )
        for example, payload in zip(question_batch, payloads):
            atomic_write_json(_question_path(output_root, example), payload)
            completed += 1
        LOGGER.info(
            "worker %d/%d checkpoint %d/%d pending questions",
            args.shard_index,
            args.num_shards,
            completed,
            len(pending),
        )


if __name__ == "__main__":
    main()
