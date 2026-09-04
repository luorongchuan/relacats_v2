"""Independently calculate shared P(Yes) confidence for saved responses.

The one-token confidence definition follows CaTS so that the original
baselines and RelaCaTS rows are evaluated from exactly the same artifacts.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any, Iterator, Sequence

from relacats_v2.common import (
    confidence_from_logprobs,
    confidence_suffix,
    read_json,
    read_jsonl,
    model_family as infer_model_family,
    validate_or_write_metadata,
)
from relacats_v2.evaluation._artifacts import (
    complete_chunk,
    merge_chunks,
    require_local_model,
    response_sources,
    scalar_json,
    source_signature,
    write_chunk,
    write_manifest,
)
from relacats_v2.evaluation.answer_parsing import (
    extract_dataset_answer,
    extract_gold_answer,
    parser_version,
)


LOGGER = logging.getLogger("relacats_v2.calculate_confidence")


def _selected_batches(
    paths: Sequence[Path],
    batch_size: int,
    num_shards: int,
    shard_index: int,
    input_already_sharded: bool = False,
) -> Iterator[tuple[int, list[dict[str, Any]]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batch: list[dict[str, Any]] = []
    batch_index = 0
    selection_shards = 1 if input_already_sharded else num_shards
    selection_index = 0 if input_already_sharded else shard_index
    global_index = 0
    for path in paths:
        for record in read_jsonl(path):
            selected = global_index % selection_shards == selection_index
            global_index += 1
            if not selected:
                continue
            if not record.get("sample_id"):
                raise ValueError(f"Response record has no sample_id in {path}")
            batch.append(record)
            if len(batch) == batch_size:
                yield batch_index, batch
                batch_index += 1
                batch = []
    if batch:
        yield batch_index, batch


def _normalise_arc_label(label: str) -> str:
    label = label.upper()
    if label in {"1", "2", "3", "4", "5"}:
        return chr(ord("A") + int(label) - 1)
    return label


def extract_answer(
    dataset_name: str,
    text: str,
    handler: Any,
    *,
    answer_type: str | None = None,
) -> Any:
    """Strictly parse a model response from its explicit final-answer field."""

    extracted = extract_dataset_answer(
        dataset_name, text, handler, answer_type=answer_type
    )
    if extracted is None:
        return None
    if str(dataset_name).strip().lower() in {"arc_challenge", "arc_easy"}:
        return extracted
    return scalar_json(extracted)


def _engine_and_params(args: argparse.Namespace, model: str) -> tuple[Any, Any]:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:  # pragma: no cover - exercised only on GPU hosts
        raise RuntimeError("vLLM is required for confidence calculation") from exc

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
    # This is intentionally the original CaTS confidence query: one generated
    # token and top-20 next-token probabilities at temperature zero.
    sampling = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    return engine, sampling


def calculate(args: argparse.Namespace) -> Path:
    responses_already_sharded = bool(
        getattr(args, "responses_already_sharded", False)
    )
    model_family_override = getattr(args, "model_family", None)
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require 0 <= shard_index < num_shards")
    model = require_local_model(args.model, args.allow_remote_model)
    resolved_model_family = model_family_override or infer_model_family(model)
    paths = response_sources(args.responses)
    producer_manifest: dict[str, Any] | None = None
    if responses_already_sharded:
        # The launcher gives each worker exactly one response shard.  Do not
        # apply a second modulo split to that shard (which would silently drop
        # half of its samples).  Check the producer metadata when available so
        # a shard from another run cannot be mixed into this output.
        metadata_path = paths[0].parent / "response_metadata.json"
        if not metadata_path.is_file() and paths[0].parent.name == "chunks":
            metadata_path = paths[0].parent.parent / "response_metadata.json"
        if not metadata_path.is_file():
            raise ValueError(
                "--responses-already-sharded requires response_metadata.json "
                "beside the response artifact"
            )
        if metadata_path.is_file():
            response_metadata = read_json(metadata_path)
            if int(response_metadata.get("num_shards", 1)) != args.num_shards:
                raise ValueError(
                    "Response shard count does not match confidence launcher: "
                    f"{response_metadata.get('num_shards')} != {args.num_shards}"
                )
            if int(response_metadata.get("shard_index", -1)) != args.shard_index:
                raise ValueError(
                    "Response shard index does not match confidence launcher: "
                    f"{response_metadata.get('shard_index')} != {args.shard_index}"
                )
            producer_manifest_path = metadata_path.parent / "response_manifest.json"
            if not producer_manifest_path.is_file():
                raise ValueError(
                    "--responses-already-sharded requires response_manifest.json "
                    f"beside the response artifact: {producer_manifest_path}"
                )
            producer_manifest = read_json(producer_manifest_path)
            if producer_manifest.get("complete") is not True:
                raise ValueError(
                    f"Producer response manifest is incomplete: {producer_manifest_path}"
                )
            if producer_manifest.get("expected_samples") != producer_manifest.get("samples"):
                raise ValueError(
                    f"Producer response sample count mismatch: {producer_manifest_path}"
                )
            if producer_manifest.get("expected_questions") != producer_manifest.get("questions"):
                raise ValueError(
                    f"Producer response question count mismatch: {producer_manifest_path}"
                )
    artifact_dir = (
        Path(args.output_dir).expanduser().resolve()
        / f"shard-{args.shard_index:05d}-of-{args.num_shards:05d}"
    )
    chunk_dir = artifact_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "artifact_type": "relacats_v2_confidence",
        # Confidence records are shared by the original baselines and the
        # RelaCaTS rows.  Preserve the original one-token confidence
        # definition, while recording the namespace used by report writers.
        "evaluation_namespace": "RelaCaTS",
        "evaluation_implementation": "RelaCaTS-v2",
        "test_time_relational_transformation": False,
        "model": model,
        "model_family": resolved_model_family,
        "confidence_definition": "P(Yes) from one-token top-20 logprobs",
        "confidence_suffix": confidence_suffix(resolved_model_family),
        "shared_response_consumers": ["CaTS", "RelaCaTS"],
        # Confidence input can be a single dataset or a parent containing
        # several shards.  Record the deterministic dispatch policy rather
        # than guessing a dataset name from a filesystem directory.
        "answer_parser_versions": {
            "math_qa": parser_version("math_qa"),
            "other_datasets": parser_version("other"),
        },
        "response_sources": source_signature(paths),
        "batch_size": args.batch_size,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "responses_already_sharded": responses_already_sharded,
        "producer_expected_questions": (
            producer_manifest.get("expected_questions")
            if producer_manifest is not None
            else None
        ),
        "seed": args.seed,
    }
    validate_or_write_metadata(artifact_dir / "confidence_metadata.json", metadata)

    total_batches = 0
    missing: set[int] = set()
    selected_samples = 0
    mismatched_models: set[str] = set()
    for batch_index, batch in _selected_batches(
        paths,
        args.batch_size,
        args.num_shards,
        args.shard_index,
        responses_already_sharded,
    ):
        total_batches = batch_index + 1
        selected_samples += len(batch)
        mismatched_models.update(
            str(record["response_model"])
            for record in batch
            if record.get("response_model") not in (None, model)
        )
        expected_ids = [str(record["sample_id"]) for record in batch]
        chunk_path = chunk_dir / f"chunk-{batch_index:06d}.jsonl"
        if complete_chunk(
            chunk_path,
            expected_ids,
            required_fields=(
                "extracted_answer",
                "correct_answer",
                "confidence",
                "yes_token_found_top20",
            ),
        ):
            LOGGER.info(
                "Resume: valid confidence chunk %d already exists", batch_index + 1
            )
        else:
            missing.add(batch_index)
    if mismatched_models:
        raise ValueError(
            "Confidence must use the same model that generated the responses; "
            f"response models={sorted(mismatched_models)}, confidence model={model}"
        )
    if total_batches == 0:
        raise ValueError("The selected confidence shard contains no responses")

    engine = sampling = None
    if missing:
        engine, sampling = _engine_and_params(args, model)

    # The second streaming pass bounds host memory even for 100k+ long CoTs.
    from utils.dataset_loader import get_dataset

    handlers: dict[str, Any] = {}
    for batch_index, batch in _selected_batches(
        paths,
        args.batch_size,
        args.num_shards,
        args.shard_index,
        responses_already_sharded,
    ):
        if batch_index not in missing:
            continue
        prompts = [
            f"{record['prompt']} {record['response']} "
                f"{confidence_suffix(resolved_model_family)}"
            for record in batch
        ]
        outputs = engine.generate(prompts, sampling)
        if len(outputs) != len(batch):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} confidence outputs for "
                f"{len(batch)} prompts"
            )
        result_records: list[dict[str, Any]] = []
        for record, output in zip(batch, outputs):
            if not output.outputs or not output.outputs[0].logprobs:
                raise RuntimeError(
                    f"Missing confidence logprobs for sample {record['sample_id']}"
                )
            token_logprobs = output.outputs[0].logprobs[0]
            yes_prob, no_prob, yes_found, no_found = confidence_from_logprobs(
                token_logprobs
            )
            if not (math.isfinite(yes_prob) and math.isfinite(no_prob)):
                raise RuntimeError(
                    f"Non-finite confidence for sample {record['sample_id']}"
                )
            dataset_name = str(record["dataset_name"])
            if dataset_name not in handlers:
                handlers[dataset_name] = get_dataset(dataset_name)
            handler = handlers[dataset_name]
            gold = record.get("correct_answer")
            if gold is None:
                gold = extract_gold_answer(
                    dataset_name, str(record["gold_text"]), handler
                )
            elif dataset_name in {"arc_challenge", "arc_easy"}:
                gold = _normalise_arc_label(str(gold))
            answer = extract_answer(
                dataset_name,
                str(record["response"]),
                handler,
                answer_type=record.get("answer_type"),
            )
            result_records.append(
                {
                    **record,
                    "artifact_type": "relacats_v2_confidence_record",
                    "correct_answer": scalar_json(gold),
                    "extracted_answer": scalar_json(answer),
                    "is_correct": bool(
                        gold is not None
                        and answer is not None
                        and handler.check(gold, answer)
                    ),
                    "confidence": float(yes_prob),
                    "true_prob": float(yes_prob),
                    "false_prob": float(no_prob),
                    "yes_token_found_top20": bool(yes_found),
                    "no_token_found_top20": bool(no_found),
                    "confidence_valid": bool(yes_found),
                }
            )
        chunk_path = chunk_dir / f"chunk-{batch_index:06d}.jsonl"
        write_chunk(chunk_path, result_records)
        LOGGER.info(
            "Confidence checkpoint %d/%d: %d responses",
            batch_index + 1,
            total_batches,
            len(result_records),
        )

    chunk_paths = [
        chunk_dir / f"chunk-{index:06d}.jsonl" for index in range(total_batches)
    ]
    stats = merge_chunks(chunk_paths, artifact_dir / "confidence.jsonl")
    expected_questions = (
        producer_manifest.get("expected_questions")
        if producer_manifest is not None
        else stats["questions"]
    )
    manifest_stats = {
        **stats,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "expected_samples": selected_samples,
        "expected_questions": expected_questions,
        "complete": (
            stats["samples"] == selected_samples
            and stats["questions"] == expected_questions
        ),
    }
    write_manifest(
        artifact_dir / "confidence_manifest.json",
        "relacats_v2_confidence",
        manifest_stats,
    )
    if not manifest_stats["complete"]:
        raise RuntimeError(f"Incomplete confidence artifact: {manifest_stats}")
    LOGGER.info("Complete confidence artifact: %s", artifact_dir)
    return artifact_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate shared CaTS-compatible P(Yes) scores for saved "
            "responses (used by CaTS baselines and RelaCaTS methods)"
        )
    )
    parser.add_argument("--model", required=True, help="Same local model used to respond")
    parser.add_argument(
        "--model-family",
        choices=("llama", "qwen", "deepseek"),
        help="Override suffix auto-detection for generically named merged model paths",
    )
    parser.add_argument("--responses", required=True, help="Response artifact file/directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--responses-already-sharded",
        action="store_true",
        help="Treat --responses as exactly one producer shard; do not modulo-split it again",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--allow-remote-model", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    calculate(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
