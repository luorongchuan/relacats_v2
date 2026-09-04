"""Chunked, resumable shared-response generation for RelaCaTS-v2 evaluation.

No relational transformation is applied here.  RelaCaTS-v2 uses relational
views only while constructing its training targets; at test time this module
preserves the original CaTS CoT prompt and ordinary stochastic sampling.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Sequence

from relacats_v2.common import (
    atomic_write_json,
    batched,
    build_reasoning_prompt,
    generation_defaults,
    stable_id,
    validate_or_write_metadata,
)
from relacats_v2.evaluation._artifacts import (
    SUPPORTED_EVAL_DATASETS,
    complete_chunk,
    merge_chunks,
    require_local_model,
    scalar_json,
    write_chunk,
    write_manifest,
)
from relacats_v2.evaluation.answer_parsing import (
    extract_gold_answer,
    parser_version,
)


LOGGER = logging.getLogger("relacats_v2.generate_responses")


def load_questions(
    dataset_name: str,
    split: str,
    tokenizer: Any,
    max_questions: int | None = None,
) -> list[dict[str, Any]]:
    """Load and format test questions through the original CaTS adapter."""

    # Keep datasets/transformers imports out of CPU aggregation and smoke tests.
    from utils.dataset_loader import get_dataset

    handler = get_dataset(dataset_name)
    split_map, answer_type = handler.load_data()
    if split not in split_map:
        raise KeyError(
            f"Split {split!r} is unavailable for {dataset_name}; "
            f"available={list(split_map)}"
        )
    source = split_map[split]
    if max_questions is not None:
        if max_questions <= 0:
            raise ValueError("max_questions must be positive")
        source = source.select(range(min(max_questions, len(source))))
    qa_data = handler.prepare_qa_data(source)
    if not qa_data:
        raise RuntimeError(f"No questions prepared for {dataset_name}/{split}")

    # The upstream MathQA adapter labels this task as ``number`` even though
    # its gold answers and extractor are option letters (A--E).  Evaluation
    # must use the same option-letter prompt as the other multiple-choice
    # handlers; otherwise the model is instructed to emit a different answer
    # space than the scorer.
    if dataset_name == "math_qa":
        answer_type = "option letter"

    questions: list[dict[str, Any]] = []
    for question_index, (question, gold_text) in enumerate(qa_data.items()):
        question_id = (
            f"{dataset_name}:{split}:{question_index}:"
            f"{stable_id(question, gold_text, length=12)}"
        )
        questions.append(
            {
                "question_id": question_id,
                "question_index": question_index,
                "dataset_name": dataset_name,
                "split": split,
                "source_question": str(question),
                "prompt": build_reasoning_prompt(tokenizer, str(question), answer_type),
                "gold_text": str(gold_text),
                # Keep MathQA's official handler untouched.  The local
                # evaluator accepts its paper-mode ``Answer: (A)`` spelling
                # through the versioned compatibility parser.
                "correct_answer": scalar_json(
                    extract_gold_answer(dataset_name, str(gold_text), handler)
                ),
                "answer_type": answer_type,
                "answer_parser_version": parser_version(dataset_name),
            }
        )
    return questions


def _engine_and_params(args: argparse.Namespace, model: str) -> tuple[Any, Any]:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:  # pragma: no cover - exercised only on GPU hosts
        raise RuntimeError("vLLM is required for response generation") from exc

    engine_kwargs: dict[str, Any] = {
        "model": model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
        "seed": args.seed,
    }
    if args.max_model_len is not None:
        engine_kwargs["max_model_len"] = args.max_model_len
    if args.enforce_eager:
        engine_kwargs["enforce_eager"] = True
    engine = LLM(**engine_kwargs)
    sampling = SamplingParams(
        n=args.num_generations,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    return engine, sampling


def generate(args: argparse.Namespace) -> Path:
    if args.dataset not in SUPPORTED_EVAL_DATASETS:
        raise ValueError(
            f"Unsupported dataset {args.dataset!r}; choose from "
            f"{SUPPORTED_EVAL_DATASETS}"
        )
    if args.num_generations <= 0:
        raise ValueError("num_generations must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require 0 <= shard_index < num_shards")

    model = require_local_model(args.model, args.allow_remote_model)
    defaults = generation_defaults(model)
    if args.max_tokens is None:
        args.max_tokens = defaults["max_new_tokens"]
    if args.max_model_len is None:
        args.max_model_len = defaults["max_model_len"]
    if args.max_tokens <= 0 or args.max_model_len <= 0:
        raise ValueError("max_tokens and max_model_len must be positive")
    if args.max_tokens >= args.max_model_len:
        raise ValueError("max_tokens must be smaller than max_model_len")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised only on GPU hosts
        raise RuntimeError("transformers is required for response generation") from exc
    # Older RelaCaTS-v1 merged directories may have a separate
    # ``chat_template.jinja`` but ``tokenizer_config.json`` with
    # ``chat_template=null``.  In that case use the corresponding untouched
    # base-model tokenizer for prompt rendering while keeping the merged model
    # as the vLLM generation model.  This is read-only and preserves the exact
    # base tokenizer protocol without rewriting an old checkpoint.
    tokenizer_source = args.tokenizer_source or model
    if not args.allow_remote_model:
        tokenizer_source = require_local_model(tokenizer_source)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=args.trust_remote_code,
        local_files_only=not args.allow_remote_model,
    )
    all_questions = load_questions(
        args.dataset, args.split, tokenizer, args.max_questions
    )
    selected = all_questions[args.shard_index :: args.num_shards]
    if not selected:
        raise ValueError(
            f"Shard {args.shard_index}/{args.num_shards} contains no questions; "
            f"dataset has {len(all_questions)}"
        )

    artifact_dir = (
        Path(args.output_dir).expanduser().resolve()
        / f"shard-{args.shard_index:05d}-of-{args.num_shards:05d}"
    )
    chunk_dir = artifact_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "artifact_type": "relacats_v2_responses",
        # This stage produces the response pool consumed by both the original
        # baselines and the RelaCaTS aggregators.  Keep the prompt identifier
        # below for protocol compatibility, but explicitly identify the
        # report namespace so downstream tools do not label our rows as
        # ``CaTS-*``.
        "evaluation_namespace": "RelaCaTS",
        "evaluation_implementation": "RelaCaTS-v2",
        "test_time_relational_transformation": False,
        "model": model,
        "tokenizer_source": tokenizer_source,
        "tokenizer_chat_template_present": bool(
            getattr(tokenizer, "chat_template", None)
        ),
        "dataset": args.dataset,
        "split": args.split,
        "dataset_questions_total": len(all_questions),
        "shard_questions": len(selected),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "num_generations": args.num_generations,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "max_questions": args.max_questions,
        "question_batch_size": args.question_batch_size,
        "prompt_protocol": "original_cats_cot",
        "prompt_protocol_owner": "CaTS",
        "shared_response_consumers": ["CaTS", "RelaCaTS"],
        "forced_assistant_think": False,
        "answer_parser_version": parser_version(args.dataset),
    }
    validate_or_write_metadata(artifact_dir / "response_metadata.json", metadata)

    question_batches = list(batched(selected, args.question_batch_size))
    missing_batch_indices: list[int] = []
    for batch_index, question_batch in enumerate(question_batches):
        expected_ids = [
            stable_id(question["question_id"], generation_index, length=24)
            for question in question_batch
            for generation_index in range(args.num_generations)
        ]
        chunk_path = chunk_dir / f"chunk-{batch_index:06d}.jsonl"
        if complete_chunk(
            chunk_path,
            expected_ids,
            required_fields=("prompt", "response", "correct_answer", "response_model"),
        ):
            LOGGER.info(
                "Resume: valid response chunk %d/%d already exists",
                batch_index + 1,
                len(question_batches),
            )
        else:
            missing_batch_indices.append(batch_index)

    engine = sampling = None
    if missing_batch_indices:
        engine, sampling = _engine_and_params(args, model)

    for batch_index in missing_batch_indices:
        question_batch = question_batches[batch_index]
        prompts = [question["prompt"] for question in question_batch]
        outputs = engine.generate(prompts, sampling)
        if len(outputs) != len(question_batch):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} prompt outputs for "
                f"{len(question_batch)} prompts"
            )
        records: list[dict[str, Any]] = []
        for question, prompt_output in zip(question_batch, outputs):
            candidates = list(prompt_output.outputs)
            if len(candidates) != args.num_generations:
                raise RuntimeError(
                    f"vLLM returned {len(candidates)} generations for "
                    f"{question['question_id']}, expected {args.num_generations}"
                )
            for generation_index, candidate in enumerate(candidates):
                records.append(
                    {
                        "schema_version": 1,
                        "artifact_type": "relacats_v2_response",
                        "response_model": model,
                        "sample_id": stable_id(
                            question["question_id"], generation_index, length=24
                        ),
                        **question,
                        "generation_index": generation_index,
                        "response": candidate.text,
                        "finish_reason": getattr(candidate, "finish_reason", None),
                    }
                )
        chunk_path = chunk_dir / f"chunk-{batch_index:06d}.jsonl"
        write_chunk(chunk_path, records)
        LOGGER.info(
            "Checkpoint %d/%d: %d questions, %d responses",
            batch_index + 1,
            len(question_batches),
            len(question_batch),
            len(records),
        )

    chunk_paths = [
        chunk_dir / f"chunk-{index:06d}.jsonl"
        for index in range(len(question_batches))
    ]
    stats = merge_chunks(chunk_paths, artifact_dir / "responses.jsonl")
    manifest_stats = {
        **stats,
        "dataset": args.dataset,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "expected_questions": len(selected),
        "expected_samples": len(selected) * args.num_generations,
        "complete": (
            stats["questions"] == len(selected)
            and stats["samples"] == len(selected) * args.num_generations
        ),
    }
    write_manifest(
        artifact_dir / "response_manifest.json",
        "relacats_v2_responses",
        manifest_stats,
    )
    if not manifest_stats["complete"]:
        raise RuntimeError(f"Incomplete response artifact: {manifest_stats}")
    atomic_write_json(
        Path(args.output_dir).expanduser().resolve()
        / f"last_response_artifact_shard-{args.shard_index:05d}-of-{args.num_shards:05d}.json",
        {"path": str(artifact_dir), **manifest_stats},
    )
    LOGGER.info("Complete response artifact: %s", artifact_dir)
    return artifact_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate shared ordinary (non-relational) test responses for "
            "CaTS baselines and RelaCaTS aggregators with vLLM"
        )
    )
    parser.add_argument("--model", required=True, help="Local trained/merged model path")
    parser.add_argument(
        "--tokenizer-source",
        help=(
            "Optional local model directory used only to render the chat prompt. "
            "Useful for legacy merged checkpoints whose tokenizer_config has a "
            "null chat_template; model weights still come from --model."
        ),
    )
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_EVAL_DATASETS)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-generations", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Defaults to 2048 for DeepSeek and 1024 for Qwen/Llama",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--question-batch-size", type=int, default=8)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument(
        "--max-model-len", type=int, help="Defaults to 8192 for every family"
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--allow-remote-model",
        action="store_true",
        help="Allow a Hugging Face model id; local directories are required by default",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = build_parser().parse_args(argv)
    generate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
