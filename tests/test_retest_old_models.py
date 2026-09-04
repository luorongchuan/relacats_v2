from __future__ import annotations

import json
from pathlib import Path

import pytest

from relacats_v2.common import atomic_write_jsonl
from relacats_v2.evaluation.retest_old_models import (
    ModelSpec,
    _is_validation_question,
    _partition_records,
    _validate_bundle,
    parse_model_specs,
)


def test_model_spec_parser_accepts_serial_shell_forms():
    specs = parse_model_specs(["qwen_cats llama_relacats_v1"])
    assert [spec.tag for spec in specs] == ["qwen_cats", "llama_relacats_v1"]
    assert specs[0].family == "qwen"
    assert specs[1].kind == "relacats_v1"
    pipe = parse_model_specs(["x|cats|deepseek|/models/x"])[0]
    assert pipe == ModelSpec("x", "cats", "deepseek", "/models/x")


def _write_bundle(root: Path, *, nan_confidence: bool = False) -> tuple[Path, Path]:
    tag = "qwen2_5_7b_instruct_cats"
    dataset = "object_counting"
    response_dir = root / tag / "responses" / dataset / "shard-00000-of-00001"
    confidence_dir = root / tag / "confidence" / dataset / "shard-00000-of-00001"
    response_dir.mkdir(parents=True)
    confidence_dir.mkdir(parents=True)
    responses = []
    confidences = []
    for question_index in range(20):
        question_id = f"{dataset}:test:{question_index}"
        for generation_index in range(32):
            sample_id = f"sample-{question_index}-{generation_index}"
            response = {
                "sample_id": sample_id,
                "question_id": question_id,
                "generation_index": generation_index,
                "dataset_name": dataset,
                "split": "test",
                "answer_type": "number",
                "correct_answer": 1,
                "response": "Explanation: x\nAnswer: 1",
                "response_model": "/models/qwen",
            }
            confidence = {
                **response,
                "extracted_answer": 1,
                "confidence": float("nan") if nan_confidence else 0.9,
                "confidence_valid": True,
            }
            responses.append(response)
            confidences.append(confidence)
    atomic_write_jsonl(response_dir / "responses.jsonl", responses)
    atomic_write_jsonl(confidence_dir / "confidence.jsonl", confidences)
    response_manifest = {
        "schema_version": 1,
        "artifact_type": "relacats_v2_responses",
        "complete": True,
        "questions": 20,
        "samples": 640,
        "expected_questions": 20,
        "expected_samples": 640,
        "num_shards": 1,
        "shard_index": 0,
    }
    confidence_manifest = {
        **response_manifest,
        "artifact_type": "relacats_v2_confidence",
    }
    (response_dir / "response_manifest.json").write_text(
        json.dumps(response_manifest), encoding="utf-8"
    )
    (confidence_dir / "confidence_manifest.json").write_text(
        json.dumps(confidence_manifest), encoding="utf-8"
    )
    metadata = {"model": "/models/qwen", "model_family": "qwen", "num_generations": 32}
    (response_dir / "response_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    (confidence_dir / "confidence_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return response_dir, confidence_dir


def test_bundle_validation_pairs_ids_and_enforces_32_per_question(tmp_path):
    _write_bundle(tmp_path)
    spec = ModelSpec("qwen2_5_7b_instruct_cats", "cats", "qwen")
    bundle = _validate_bundle(tmp_path, spec, "object_counting", expected_generations=32)
    assert bundle.questions == 20
    assert bundle.samples == 640
    rows = list(
        _partition_records(
            bundle.confidence_files,
            "object_counting",
            "validation",
            seed=42,
            fraction=0.2,
        )
    )
    assert len({row["question_id"] for row in rows}) > 0
    assert all(
        sum(row["question_id"] == question for row in rows) == 32
        for question in {row["question_id"] for row in rows}
    )
    assert not (
        {
            row["question_id"]
            for row in _partition_records(
                bundle.confidence_files,
                "object_counting",
                "validation",
                seed=42,
                fraction=0.2,
            )
        }
        & {
            row["question_id"]
            for row in _partition_records(
                bundle.confidence_files,
                "object_counting",
                "test",
                seed=42,
                fraction=0.2,
            )
        }
    )


def test_bundle_validation_rejects_nan(tmp_path):
    _write_bundle(tmp_path, nan_confidence=True)
    spec = ModelSpec("qwen2_5_7b_instruct_cats", "cats", "qwen")
    with pytest.raises(ValueError, match="Non-finite"):
        _validate_bundle(tmp_path, spec, "object_counting", expected_generations=32)


def test_holdout_is_deterministic():
    values = [
        _is_validation_question(f"q{i}", "object_counting", seed=42, fraction=0.2)
        for i in range(100)
    ]
    assert values == [
        _is_validation_question(f"q{i}", "object_counting", seed=42, fraction=0.2)
        for i in range(100)
    ]
