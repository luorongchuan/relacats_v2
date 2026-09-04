"""CPU aggregation for a fresh six-model RelaCaTS-v2 old-model retest.

The GPU stages write one ordinary response pool and one confidence pool for
each model tag.  This module deliberately does not load a model and never
modifies a checkpoint: it validates the immutable GPU artifacts, derives a
deterministic question-level calibration holdout, selects dynamic thresholds
on that holdout only, and evaluates the remaining questions with the
persisted thresholds.

The input layout used by :mod:`scripts.14_retest_old_models_gpu67.sh` is::

    <artifact-root>/<tag>/responses/<dataset>/shard-*/response_manifest.json
    <artifact-root>/<tag>/confidence/<dataset>/shard-*/confidence_manifest.json

The manifest discovery is intentionally a little more permissive so that a
completed run can be moved without changing its report, but a missing or
incomplete manifest is always an error.  The output directory may already
contain ``artifacts/`` and ``logs/`` from the GPU stage; only this helper's
owned reports are created or replaced (with ``--resume``).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from relacats_v2.common import atomic_write_json, read_json, read_jsonl
from relacats_v2.evaluation.aggregate import (
    AggregateConfig,
    build_threshold_calibration,
    evaluate_records,
    write_reports,
)
from relacats_v2.evaluation.method_names import (
    TABLE2_METHOD_ORDER,
    canonical_method_name,
)


DEFAULT_DATASETS = ("object_counting", "math_qa", "arc_challenge")
DEFAULT_MODEL_TAGS = (
    "qwen2_5_7b_instruct_cats",
    "qwen2_5_7b_instruct_relacats_v1",
    "llama3_1_8b_instruct_cats",
    "llama3_1_8b_instruct_relacats_v1",
    "deepseek_r1_distill_qwen_1_5b_cats",
    "deepseek_r1_distill_qwen_1_5b_relacats_v1",
)
EXPECTED_SCHEMA = "relacats-v2.old-model-retest.1"
DEFAULT_NUM_GENERATIONS = 32


@dataclass(frozen=True)
class ModelSpec:
    """A named artifact set and optional model metadata supplied by the CLI."""

    tag: str
    kind: str = "unknown"
    family: str = "unknown"
    model_path: str | None = None


@dataclass(frozen=True)
class ArtifactBundle:
    """Validated response/confidence files for one model and dataset."""

    response_files: tuple[Path, ...]
    confidence_files: tuple[Path, ...]
    questions: int
    samples: int
    sample_ids_sha256: str
    manifest_paths: tuple[Path, ...]
    model: str | None
    family: str | None


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    return True


def _assert_finite(value: Any, where: str) -> None:
    if not _finite(value):
        raise ValueError(f"Non-finite value (NaN/Inf) in artifact record: {where}")


def _safe_int(value: Any, where: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected integer {where}, got {value!r}") from exc
    return result


def _sha256_ids(ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in ids:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _is_validation_question(
    question_id: str,
    dataset: str,
    *,
    seed: int,
    fraction: float,
) -> bool:
    """Stable question-level split independent of file/shard ordering."""

    # Keep the namespace identical to ``reaggregate_existing.py`` so that a
    # retest and the earlier v2 CPU reaggregation use the same partition when
    # their question IDs come from the same dataset pool.
    digest = hashlib.sha256(
        f"relacats-v2-holdout\0{seed}\0{dataset}\0{question_id}".encode(
            "utf-8"
        )
    ).digest()
    unit = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return unit < fraction


def _partition_records(
    files: Sequence[Path],
    dataset: str,
    role: str,
    *,
    seed: int,
    fraction: float,
) -> Iterator[dict[str, Any]]:
    if role not in {"validation", "test"}:
        raise ValueError(role)
    for path in files:
        for raw in read_jsonl(path):
            question_id = str(raw.get("question_id", "")).strip()
            if not question_id:
                raise ValueError(f"Missing question_id in {path}")
            source_split = str(raw.get("split", "")).strip().lower()
            if source_split != "test":
                raise ValueError(
                    f"Fresh retest artifacts must be split=test; {path} has {source_split!r}"
                )
            selected = _is_validation_question(
                question_id, dataset, seed=seed, fraction=fraction
            )
            if selected != (role == "validation"):
                continue
            record = dict(raw)
            record["source_split"] = source_split
            record["split"] = role
            yield record


def _manifest_dirs(root: Path, prefix: str, dataset: str) -> list[Path]:
    """Find manifest parent directories without accepting unrelated datasets."""

    if not root.exists():
        return []
    result: list[Path] = []
    filename = f"{prefix}_manifest.json"
    for path in sorted(root.rglob(filename)):
        if path.is_file() and dataset in path.parts:
            result.append(path.parent.resolve())
    # A caller may point directly at one shard directory.  In that case the
    # dataset name need not be a path component, so accept its manifest too.
    direct = root / filename
    if direct.is_file() and direct.parent not in result:
        result.append(direct.parent.resolve())
    return list(dict.fromkeys(result))


def _candidate_roots(tag_root: Path, kind: str, dataset: str) -> list[Path]:
    names = (kind, "response" if kind == "responses" else "confidence")
    candidates: list[Path] = []
    for name in dict.fromkeys(names):
        candidates.extend(
            (
                tag_root / name / dataset,
                tag_root / dataset / name,
                tag_root / name,
                tag_root / dataset,
            )
        )
    return list(dict.fromkeys(path for path in candidates if path.exists()))


def _data_file(shard_dir: Path, kind: str) -> Path:
    name = "responses.jsonl" if kind == "responses" else "confidence.jsonl"
    path = shard_dir / name
    if not path.is_file():
        raise FileNotFoundError(
            f"Completed {kind} manifest has no merged {name}: {shard_dir}"
        )
    if path.stat().st_size == 0:
        raise ValueError(f"Empty {kind} artifact: {path}")
    return path.resolve()


def _load_manifest(path: Path, kind: str) -> dict[str, Any]:
    value = read_json(path / f"{kind[:-1] if kind.endswith('s') else kind}_manifest.json")
    if not isinstance(value, dict):
        raise ValueError(f"Manifest is not an object: {path}")
    _assert_finite(value, str(path))
    if value.get("complete") is not True:
        raise ValueError(f"Incomplete {kind} manifest: {path}")
    for field in ("expected_questions", "expected_samples", "questions", "samples"):
        if field not in value:
            raise ValueError(f"Missing {field} in {kind} manifest: {path}")
        if _safe_int(value[field], f"{path}/{field}") < 0:
            raise ValueError(f"Negative {field} in manifest: {path}")
    if _safe_int(value["expected_questions"], "expected_questions") != _safe_int(
        value["questions"], "questions"
    ):
        raise ValueError(f"Manifest question count mismatch: {path}")
    if _safe_int(value["expected_samples"], "expected_samples") != _safe_int(
        value["samples"], "samples"
    ):
        raise ValueError(f"Manifest sample count mismatch: {path}")
    return value


def _record_signatures(
    path: Path,
    *,
    kind: str,
    dataset: str,
    expected_generations: int,
) -> tuple[dict[str, tuple[str, int]], dict[str, str], set[str], int]:
    """Read one JSONL and return ID/signature maps plus per-question counts."""

    ids: dict[str, tuple[str, int]] = {}
    responses: dict[str, str] = {}
    questions: set[str] = set()
    counts: dict[str, int] = defaultdict(int)
    indices_by_question: dict[str, list[int]] = defaultdict(list)
    total = 0
    for row_number, raw in enumerate(read_jsonl(path), start=1):
        where = f"{path}:{row_number}"
        _assert_finite(raw, where)
        sample_id = str(raw.get("sample_id", "")).strip()
        question_id = str(raw.get("question_id", "")).strip()
        if not sample_id or not question_id:
            raise ValueError(f"Missing sample_id/question_id at {where}")
        if sample_id in ids:
            raise ValueError(f"Duplicate sample_id in artifact: {sample_id} ({where})")
        generation_index = _safe_int(raw.get("generation_index"), f"{where}/generation_index")
        if generation_index < 0 or generation_index >= expected_generations:
            raise ValueError(
                f"generation_index out of range at {where}: {generation_index}"
            )
        record_dataset = str(raw.get("dataset_name", dataset)).strip()
        if record_dataset != dataset:
            raise ValueError(
                f"Dataset mismatch at {where}: {record_dataset!r} != {dataset!r}"
            )
        if str(raw.get("split", "")).strip().lower() != "test":
            raise ValueError(f"Artifact record is not split=test at {where}")
        if kind == "responses":
            if "response" not in raw or raw.get("response") is None:
                raise ValueError(f"Missing response text at {where}")
            responses[sample_id] = str(raw.get("response", ""))
        else:
            confidence = raw.get("confidence")
            if confidence is None:
                raise ValueError(f"Missing confidence at {where}")
            try:
                confidence_value = float(confidence)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid confidence at {where}: {confidence!r}") from exc
            if not math.isfinite(confidence_value):
                raise ValueError(f"Non-finite confidence at {where}")
        ids[sample_id] = (question_id, generation_index)
        questions.add(question_id)
        counts[question_id] += 1
        indices_by_question[question_id].append(generation_index)
        total += 1
    for question_id, count in counts.items():
        if count != expected_generations:
            raise ValueError(
                f"Question {question_id} has {count} {kind} records; "
                f"expected exactly {expected_generations}"
            )
    expected_indices = list(range(expected_generations))
    for question_id in questions:
        indices = sorted(indices_by_question[question_id])
        if indices != expected_indices:
            raise ValueError(
                f"Question {question_id} does not have generation indices "
                f"0..{expected_generations - 1} in {path}"
            )
    return ids, responses, questions, total


def _validate_bundle(
    artifact_root: Path,
    spec: ModelSpec,
    dataset: str,
    *,
    expected_generations: int,
) -> ArtifactBundle:
    tag_root = artifact_root / spec.tag
    response_dirs: list[Path] = []
    confidence_dirs: list[Path] = []
    for root in _candidate_roots(tag_root, "responses", dataset):
        response_dirs.extend(_manifest_dirs(root, "response", dataset))
    for root in _candidate_roots(tag_root, "confidence", dataset):
        confidence_dirs.extend(_manifest_dirs(root, "confidence", dataset))
    response_dirs = list(dict.fromkeys(response_dirs))
    confidence_dirs = list(dict.fromkeys(confidence_dirs))
    if not response_dirs:
        raise FileNotFoundError(
            f"No response_manifest.json found for {spec.tag}/{dataset} below {tag_root}"
        )
    if not confidence_dirs:
        raise FileNotFoundError(
            f"No confidence_manifest.json found for {spec.tag}/{dataset} below {tag_root}"
        )

    def shard_key(path: Path, manifest: Mapping[str, Any]) -> tuple[int, int]:
        return (
            _safe_int(manifest.get("num_shards", 1), f"{path}/num_shards"),
            _safe_int(manifest.get("shard_index", 0), f"{path}/shard_index"),
        )

    response_by_shard: dict[tuple[int, int], tuple[Path, dict[str, Any]]] = {}
    confidence_by_shard: dict[tuple[int, int], tuple[Path, dict[str, Any]]] = {}
    for directory in response_dirs:
        manifest = _load_manifest(directory, "responses")
        key = shard_key(directory, manifest)
        if key in response_by_shard:
            raise ValueError(f"Duplicate response shard {key} for {spec.tag}/{dataset}")
        response_by_shard[key] = (_data_file(directory, "responses"), manifest)
    for directory in confidence_dirs:
        manifest = _load_manifest(directory, "confidence")
        key = shard_key(directory, manifest)
        if key in confidence_by_shard:
            raise ValueError(f"Duplicate confidence shard {key} for {spec.tag}/{dataset}")
        confidence_by_shard[key] = (_data_file(directory, "confidence"), manifest)
    if set(response_by_shard) != set(confidence_by_shard):
        raise ValueError(
            f"Response/confidence shard mismatch for {spec.tag}/{dataset}: "
            f"responses={sorted(response_by_shard)}, confidence={sorted(confidence_by_shard)}"
        )
    shard_counts = {key[0] for key in response_by_shard}
    if len(shard_counts) != 1:
        raise ValueError(f"Mixed num_shards in artifacts for {spec.tag}/{dataset}")
    if any(key[1] < 0 or key[1] >= key[0] for key in response_by_shard):
        raise ValueError(f"Invalid shard index in artifacts for {spec.tag}/{dataset}")

    response_files: list[Path] = []
    confidence_files: list[Path] = []
    all_response_ids: dict[str, tuple[str, int]] = {}
    all_confidence_ids: dict[str, tuple[str, int]] = {}
    all_questions: set[str] = set()
    model_values: set[str] = set()
    family_values: set[str] = set()
    total_responses = total_confidence = 0
    for key in sorted(response_by_shard):
        response_path, response_manifest = response_by_shard[key]
        confidence_path, confidence_manifest = confidence_by_shard[key]
        r_ids, _r_text, r_questions, r_total = _record_signatures(
            response_path,
            kind="responses",
            dataset=dataset,
            expected_generations=expected_generations,
        )
        c_ids, _, c_questions, c_total = _record_signatures(
            confidence_path,
            kind="confidence",
            dataset=dataset,
            expected_generations=expected_generations,
        )
        if r_ids != c_ids:
            missing = sorted(set(r_ids) - set(c_ids))[:3]
            extra = sorted(set(c_ids) - set(r_ids))[:3]
            raise ValueError(
                f"Response/confidence sample IDs differ for {spec.tag}/{dataset}/{key}; "
                f"missing_confidence={missing}, extra_confidence={extra}"
            )
        response_digest = response_manifest.get("sample_id_sha256")
        if response_digest is not None and str(response_digest) != _sha256_ids(r_ids):
            raise ValueError(f"Response manifest sample_id_sha256 mismatch: {response_path}")
        confidence_digest = confidence_manifest.get("sample_id_sha256")
        if confidence_digest is not None and str(confidence_digest) != _sha256_ids(c_ids):
            raise ValueError(f"Confidence manifest sample_id_sha256 mismatch: {confidence_path}")
        if r_questions != c_questions:
            raise ValueError(f"Response/confidence question IDs differ for {spec.tag}/{dataset}/{key}")
        if r_total != _safe_int(response_manifest["expected_samples"], "response expected_samples"):
            raise ValueError(f"Response manifest count disagrees with JSONL: {response_path}")
        if c_total != _safe_int(confidence_manifest["expected_samples"], "confidence expected_samples"):
            raise ValueError(f"Confidence manifest count disagrees with JSONL: {confidence_path}")
        if len(r_questions) != _safe_int(
            response_manifest["expected_questions"], "response expected_questions"
        ):
            raise ValueError(f"Response manifest question count disagrees with JSONL: {response_path}")
        if len(c_questions) != _safe_int(
            confidence_manifest["expected_questions"], "confidence expected_questions"
        ):
            raise ValueError(f"Confidence manifest question count disagrees with JSONL: {confidence_path}")
        duplicate_questions = all_questions.intersection(r_questions)
        if duplicate_questions:
            raise ValueError(
                f"Question IDs occur in multiple response shards for {spec.tag}/{dataset}: "
                f"{sorted(duplicate_questions)[:3]}"
            )
        for sample_id, signature in r_ids.items():
            if sample_id in all_response_ids:
                raise ValueError(f"Duplicate response sample_id across shards: {sample_id}")
            all_response_ids[sample_id] = signature
        for sample_id, signature in c_ids.items():
            if sample_id in all_confidence_ids:
                raise ValueError(f"Duplicate confidence sample_id across shards: {sample_id}")
            all_confidence_ids[sample_id] = signature
        all_questions.update(r_questions)
        total_responses += r_total
        total_confidence += c_total
        response_files.append(response_path)
        confidence_files.append(confidence_path)

        response_meta = response_path.parent / "response_metadata.json"
        confidence_meta = confidence_path.parent / "confidence_metadata.json"
        for metadata_path in (response_meta, confidence_meta):
            if metadata_path.is_file():
                metadata = read_json(metadata_path)
                if isinstance(metadata, Mapping):
                    _assert_finite(metadata, str(metadata_path))
                    if metadata.get("model") is not None:
                        model_values.add(str(metadata["model"]))
                    if metadata.get("model_family") is not None:
                        family_values.add(str(metadata["model_family"]))
                    if metadata.get("num_generations") is not None and _safe_int(
                        metadata["num_generations"], f"{metadata_path}/num_generations"
                    ) != expected_generations:
                        raise ValueError(
                            f"num_generations mismatch in metadata: {metadata_path}"
                        )
    if total_responses != total_confidence or set(all_response_ids) != set(all_confidence_ids):
        raise ValueError(f"Final response/confidence count mismatch for {spec.tag}/{dataset}")
    expected_total = len(all_questions) * expected_generations
    if total_responses != expected_total:
        raise ValueError(
            f"Expected {expected_total} samples ({expected_generations}/question) for "
            f"{spec.tag}/{dataset}, found {total_responses}"
        )
    if spec.family != "unknown" and family_values and spec.family not in family_values:
        raise ValueError(
            f"Model family mismatch for {spec.tag}: spec={spec.family}, artifacts={sorted(family_values)}"
        )
    if spec.model_path and model_values:
        expected_path = str(Path(spec.model_path).expanduser().resolve())
        resolved_values = {str(Path(value).expanduser().resolve()) for value in model_values}
        if expected_path not in resolved_values:
            raise ValueError(
                f"Model path mismatch for {spec.tag}: spec={expected_path}, artifacts={sorted(resolved_values)}"
            )
    return ArtifactBundle(
        response_files=tuple(response_files),
        confidence_files=tuple(confidence_files),
        questions=len(all_questions),
        samples=total_responses,
        sample_ids_sha256=_sha256_ids(sorted(all_response_ids)),
        manifest_paths=tuple(
            [directory / "response_manifest.json" for directory in response_dirs]
            + [directory / "confidence_manifest.json" for directory in confidence_dirs]
        ),
        model=next(iter(model_values), None),
        family=next(iter(family_values), None),
    )


def _flatten_model_values(values: Sequence[str] | None) -> list[str]:
    result: list[str] = []
    for value in values or ():
        # The serial shell wrapper passes ``"${MODEL_TAGS[*]}"`` as one
        # argument.  Splitting only plain tag lists keeps pipe/equal specs
        # intact while accepting that convenient invocation too.
        if "|" not in value and "=" not in value and any(ch.isspace() for ch in value):
            result.extend(value.split())
        else:
            result.append(value)
    return result


def _infer_family(text: str) -> str:
    lowered = text.casefold()
    if "deepseek" in lowered:
        return "deepseek"
    if "llama" in lowered:
        return "llama"
    if "qwen" in lowered:
        return "qwen"
    return "unknown"


def _infer_kind(tag: str) -> str:
    lowered = tag.casefold()
    if "relacats" in lowered:
        return "relacats_v1"
    if lowered.endswith("_cats") or "_cats_" in lowered:
        return "cats"
    return "unknown"


def _parse_model_spec(value: str) -> ModelSpec:
    text = value.strip()
    if not text:
        raise ValueError("Empty model specification")
    if "|" in text:
        fields = text.split("|", 3)
        if len(fields) != 4:
            raise ValueError(
                "Pipe model spec must be TAG|KIND|FAMILY|MODEL_PATH: " + text
            )
        tag, kind, family, path = (item.strip() for item in fields)
        if not tag:
            raise ValueError("Model spec has an empty tag")
        return ModelSpec(tag, kind or "unknown", family or _infer_family(tag), path or None)
    if "=" in text:
        fields = text.split("=", 3)
        if len(fields) == 2:
            tag, path = (item.strip() for item in fields)
            return ModelSpec(tag, _infer_kind(tag), _infer_family(tag), path or None)
        if len(fields) == 4:
            tag, kind, family, path = (item.strip() for item in fields)
            return ModelSpec(tag, kind or "unknown", family or _infer_family(tag), path or None)
        raise ValueError(
            "Equal model spec must be TAG=MODEL_PATH or TAG=KIND=FAMILY=MODEL_PATH: "
            + text
        )
    return ModelSpec(text, _infer_kind(text), _infer_family(text), None)


def parse_model_specs(values: Sequence[str] | None) -> list[ModelSpec]:
    raw_values = _flatten_model_values(values)
    if not raw_values:
        raw_values = list(DEFAULT_MODEL_TAGS)
    specs = [_parse_model_spec(value) for value in raw_values]
    tags = [spec.tag for spec in specs]
    if len(set(tags)) != len(tags):
        raise ValueError(f"Duplicate model tags: {tags}")
    return specs


def _selected_rows(report: Mapping[str, Any], budget: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in report.get("fixed_budget_results", ()):
        if int(raw.get("budget", -1)) == budget:
            rows.append(dict(raw))
    for raw in report.get("dynamic_budget_matches", ()):
        if int(raw.get("budget_target", raw.get("budget", -1))) == budget:
            rows.append(dict(raw))
    by_method = {canonical_method_name(row.get("method", "")): row for row in rows}
    missing = set(TABLE2_METHOD_ORDER) - set(by_method)
    if missing:
        raise ValueError(f"Missing budget-{budget} methods: {sorted(missing)}")
    return [by_method[name] for name in TABLE2_METHOD_ORDER]


def _metric(row: Mapping[str, Any] | None, name: str) -> float | int | None:
    if not row:
        return None
    aliases = {
        "accuracy": ("accuracy", "accuracy_pct", "accuracy_percent"),
        "actual_avg_samples": ("actual_avg_samples", "avg_samples_used"),
        "valid_samples": ("valid_samples",),
        "invalid_rate": ("invalid_rate", "invalid_response_rate"),
    }
    for key in aliases[name]:
        if key not in row or row[key] is None:
            continue
        value = row[key]
        try:
            result = float(value)
        except (TypeError, ValueError):
            continue
        if key in {"accuracy_pct", "accuracy_percent"}:
            result /= 100.0
        if name == "valid_samples":
            return int(result)
        return result
    return None


def _reference_report_paths(reference_root: Path, model_id: str, dataset: str) -> list[Path]:
    return [
        reference_root / model_id / dataset / "test" / "evaluation.json",
        reference_root / model_id / "results" / dataset / "evaluation.json",
        reference_root / model_id / dataset / "evaluation.json",
    ]


def _load_reference_v2(
    reference_root: Path,
    base_model: str,
    dataset: str,
    budget: int,
) -> dict[str, dict[str, Any]]:
    candidates = [path for path in _reference_report_paths(reference_root, base_model, dataset) if path.is_file()]
    if not candidates:
        for filename in ("summary.json", "model_method_summary.json"):
            path = reference_root / filename
            if not path.is_file():
                continue
            value = read_json(path)
            if not isinstance(value, list):
                continue
            rows = [
                row
                for row in value
                if isinstance(row, Mapping)
                and str(row.get("model_id", "")) == base_model
                and str(row.get("dataset_name", row.get("dataset", ""))) == dataset
            ]
            if rows:
                return {
                    canonical_method_name(row.get("method", "")): dict(row) for row in rows
                }
        return {}
    report = read_json(candidates[0])
    if not isinstance(report, Mapping):
        return {}
    try:
        return {
            canonical_method_name(row.get("method", "")): row
            for row in _selected_rows(report, budget)
        }
    except ValueError:
        return {}


def _legacy_model_aliases(spec: ModelSpec) -> set[str]:
    lowered = spec.tag.casefold()
    aliases = {spec.tag, spec.tag.removesuffix("_cats"), spec.tag.removesuffix("_relacats_v1")}
    if "qwen" in lowered:
        aliases.add("qwen")
    if "llama" in lowered:
        aliases.add("llama")
    if "deepseek" in lowered:
        aliases.add("deepseek")
    return {item for item in aliases if item}


def _load_legacy_rows(
    path: Path,
    spec: ModelSpec,
    dataset: str,
    budget: int,
) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    value = read_json(path)
    if not isinstance(value, list):
        return {}
    aliases = _legacy_model_aliases(spec)
    result: dict[str, dict[str, Any]] = {}
    for raw in value:
        if not isinstance(raw, Mapping):
            continue
        model = str(raw.get("model", raw.get("model_id", ""))).strip()
        row_dataset = str(raw.get("dataset", raw.get("dataset_name", ""))).strip()
        if model not in aliases or row_dataset != dataset:
            continue
        target = raw.get("target_budget", raw.get("budget", budget))
        try:
            if float(target) != float(budget):
                continue
        except (TypeError, ValueError):
            continue
        method = canonical_method_name(raw.get("method", ""))
        if method:
            result[method] = dict(raw)
    return result


def _base_model_id(spec: ModelSpec) -> str:
    tag = spec.tag
    for suffix in ("_relacats_v1", "_cats", "_self_calibration"):
        if tag.endswith(suffix):
            return tag[: -len(suffix)]
    return tag


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _aggregate_rows(rows: Sequence[Mapping[str, Any]], specs: Sequence[ModelSpec]) -> list[dict[str, Any]]:
    state: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        key = (str(row["model_id"]), str(row["method"]))
        target = state[key]
        questions = int(row["questions_total"])
        target["questions"] += questions
        target["correct"] += int(row["correct"])
        target["generated"] += int(row["generated_samples"])
        target["valid"] += int(row["valid_samples"])
        target["invalid"] += int(row["invalid_samples"])
    result: list[dict[str, Any]] = []
    for spec in specs:
        for method in TABLE2_METHOD_ORDER:
            item = state.get((spec.tag, method))
            if not item:
                continue
            questions = int(item["questions"])
            generated = int(item["generated"])
            result.append(
                {
                    "model_id": spec.tag,
                    "method": method,
                    "questions_total": questions,
                    "accuracy": item["correct"] / questions if questions else 0.0,
                    "actual_avg_samples": generated / questions if questions else 0.0,
                    "generated_samples": generated,
                    "valid_samples": int(item["valid"]),
                    "invalid_rate": item["invalid"] / generated if generated else 0.0,
                }
            )
    return result


def _report_markdown(
    manifest: Mapping[str, Any],
    summary_rows: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
) -> str:
    excel_path = manifest.get("legacy_excel_path")
    excel_found = bool(manifest.get("legacy_excel_found"))
    if excel_found:
        excel_note = f"- Legacy Excel reference: `{excel_path}` (available; the machine-readable JSON/CSV reference was used for exact row matching)."
    else:
        excel_note = "- Legacy Excel reference: not found/supplied; comparisons use the existing machine-readable `table2_results.json/csv` only."
    lines = [
        "# RelaCaTS-v2 old-model retest",
        "",
        "This report was produced by the CPU aggregation stage after fresh GPU response and confidence generation.",
        "No model merge, checkpoint write, or training was performed.",
        "",
        "## Protocol",
        "",
        f"- Candidate responses per question: `{manifest['num_generations']}`",
        f"- Dynamic target and hard cap: `{manifest['target_budget']}`",
        f"- Calibration holdout: `{manifest['validation_fraction']:.3f}` of questions, SHA-256 question-id split, seed `{manifest['seed']}`",
        "- Thresholds are selected on validation only and reloaded from `thresholds/` for held-out test.",
        "- Invalid answers remain in the strict question/sample denominators.",
        excel_note,
        "",
        "## Test metrics",
        "",
        "| Model | Dataset | Method | Accuracy | Actual avg samples | Valid samples | Invalid rate |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['model_id']} | {row['dataset_name']} | {row['method']} | "
            f"{100.0 * float(row['accuracy']):.3f}% | {float(row['actual_avg_samples']):.3f} | "
            f"{int(row['valid_samples'])} | {100.0 * float(row['invalid_rate']):.3f}% |"
        )
    lines.extend(
        [
            "",
            "## Comparisons",
            "",
            "The old pools and `eval_outputs_v2` use different response generations and/or a full-test versus held-out denominator. Their deltas are diagnostic and are marked `directly_comparable=false`.",
            "",
            "| Model | Dataset | Method | Reference | New accuracy | Reference accuracy | Delta |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        new_value = float(row["new_accuracy"]) * 100.0
        reference = row.get("reference_accuracy")
        reference_value = "—" if reference is None else f"{100.0 * float(reference):.3f}%"
        delta = row.get("accuracy_delta")
        delta_value = "—" if delta is None else f"{100.0 * float(delta):+.3f} pp"
        lines.append(
            f"| {row['model_id']} | {row['dataset_name']} | {row['method']} | "
            f"{row['reference_name']} | {new_value:.3f}% | {reference_value} | {delta_value} |"
        )
    lines.extend(
        [
            "",
            "## Artifact audit",
            "",
            f"- Validated model/dataset bundles: `{manifest['validated_bundles']}`",
            f"- Validated questions: `{manifest['validated_questions']}`",
            f"- Validated response/confidence samples: `{manifest['validated_samples']}`",
            f"- Every question has exactly `{manifest['num_generations']}` response and confidence records; sample IDs are paired one-to-one; non-finite numeric values are rejected.",
            "",
        ]
    )
    return "\n".join(lines)


def _safe_write_text(path: Path, text: str, *, resume: bool) -> None:
    if path.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _compatible_existing(manifest: Mapping[str, Any], args: argparse.Namespace, specs: Sequence[ModelSpec]) -> bool:
    requested_excel = (
        str(Path(args.legacy_excel).expanduser().resolve())
        if getattr(args, "legacy_excel", None)
        else None
    )
    return (
        manifest.get("schema_version") == EXPECTED_SCHEMA
        and list(manifest.get("models", ())) == [spec.tag for spec in specs]
        and list(manifest.get("datasets", ())) == list(args.datasets)
        and int(manifest.get("num_generations", -1)) == args.num_generations
        and int(manifest.get("target_budget", -1)) == args.target_budget
        and int(manifest.get("seed", -1)) == args.seed
        and abs(float(manifest.get("validation_fraction", -1.0)) - args.validation_fraction) < 1e-12
        and manifest.get("legacy_excel_path") == requested_excel
    )


def reaggregate(args: argparse.Namespace) -> Path:
    artifact_root = Path(args.artifact_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    reference_root = Path(args.reference_root).expanduser().resolve()
    legacy_results = Path(args.legacy_results).expanduser().resolve()
    legacy_excel = (
        Path(args.legacy_excel).expanduser().resolve()
        if args.legacy_excel
        else None
    )
    specs = parse_model_specs(args.model_specs or args.models)
    args.datasets = tuple(
        item
        for value in (args.datasets or DEFAULT_DATASETS)
        for item in str(value).split()
        if item
    )
    if not args.datasets:
        raise ValueError("datasets must not be empty")
    if not artifact_root.is_dir():
        raise FileNotFoundError(f"Artifact root not found: {artifact_root}")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between 0 and 1")
    if args.num_generations <= 0 or args.target_budget <= 0:
        raise ValueError("num_generations and target_budget must be positive")
    if args.target_budget > args.num_generations:
        raise ValueError("target_budget cannot exceed num_generations")

    marker = output_root / "manifest.json"
    docs_path = Path(__file__).resolve().parents[1] / "docs" / "relacats_v2_retest_old_models_report.md"
    owned_paths = (
        output_root / "results",
        output_root / "thresholds",
        output_root / "summary.json",
        output_root / "summary.csv",
        output_root / "retest_summary.json",
        output_root / "retest_summary.csv",
        output_root / "model_method_summary.json",
        output_root / "model_method_summary.csv",
        output_root / "comparisons.json",
        output_root / "comparisons.csv",
        marker,
        output_root / "retest_report.md",
    )
    if docs_path.exists() and not args.resume:
        raise FileExistsError(
            f"Retest report already exists: {docs_path}; pass --resume or choose a new run"
        )
    if output_root.exists() and not args.resume:
        if marker.is_file():
            existing = read_json(marker)
            if isinstance(existing, Mapping) and existing.get("complete") is True:
                raise FileExistsError(
                    f"Completed retest exists: {output_root}; pass --resume or choose a new output root"
                )
        if any(path.exists() for path in owned_paths):
            raise FileExistsError(
                f"Retest output already contains helper-owned files: {output_root}; pass --resume"
            )
    if marker.is_file() and args.resume:
        existing = read_json(marker)
        if isinstance(existing, Mapping) and existing.get("complete") is True and not _compatible_existing(existing, args, specs):
            raise ValueError("--resume requested, but existing retest manifest is incompatible with this invocation")
    output_root.mkdir(parents=True, exist_ok=True)

    # Keep all writes off to the side until every model/dataset has passed
    # validation and aggregation.  Existing GPU artifacts/logs remain intact.
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.cpu-building.", dir=output_root.parent)
    )
    config = AggregateConfig(
        budgets=tuple(value for value in (1, 2, 4, 8, args.target_budget) if value <= args.target_budget),
        curve_max_budget=args.target_budget,
        budget_targets=(args.target_budget,),
        esc_window_sizes=tuple(range(2, args.target_budget + 1)),
    )
    summary_rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    bundle_stats: dict[str, dict[str, Any]] = {}
    validated_questions = validated_samples = 0
    try:
        for spec in specs:
            for dataset in args.datasets:
                print(f"[retest CPU] {spec.tag}/{dataset}: validating manifests and IDs", flush=True)
                bundle = _validate_bundle(
                    artifact_root,
                    spec,
                    dataset,
                    expected_generations=args.num_generations,
                )
                validated_questions += bundle.questions
                validated_samples += bundle.samples
                bundle_stats[f"{spec.tag}/{dataset}"] = {
                    "questions": bundle.questions,
                    "samples": bundle.samples,
                    "sample_ids_sha256": bundle.sample_ids_sha256,
                    "response_files": [str(path) for path in bundle.response_files],
                    "confidence_files": [str(path) for path in bundle.confidence_files],
                    "manifest_paths": [str(path) for path in bundle.manifest_paths],
                    "model": bundle.model,
                    "family": bundle.family,
                }
                confidence_files = bundle.confidence_files
                result_root = staging / "results" / spec.tag / dataset
                threshold_path = staging / "thresholds" / spec.tag / dataset / "dynamic_thresholds.json"
                reported_threshold_path = output_root / "thresholds" / spec.tag / dataset / "dynamic_thresholds.json"
                provenance = {
                    "source_artifacts": [str(path) for path in confidence_files],
                    "source_split": "test",
                    "partition_strategy": "sha256 question-id holdout",
                    "validation_fraction": args.validation_fraction,
                    "partition_seed": args.seed,
                    "candidate_pool": args.num_generations,
                    "warning": "Validation/test are deterministic partitions of the fresh test pool; validation is not an official dataset split.",
                }
                print(f"[retest CPU] {spec.tag}/{dataset}: validation-only threshold selection", flush=True)
                validation = evaluate_records(
                    _partition_records(
                        confidence_files,
                        dataset,
                        "validation",
                        seed=args.seed,
                        fraction=args.validation_fraction,
                    ),
                    config=config,
                    phase="validation",
                    model_id=spec.tag,
                    dataset_name=dataset,
                )
                validation["partition_provenance"] = provenance
                threshold_doc = build_threshold_calibration(
                    validation, model_id=spec.tag, dataset_name=dataset
                )
                threshold_doc["partition_provenance"] = provenance
                atomic_write_json(threshold_path, threshold_doc)
                validation["threshold_calibration_file"] = str(reported_threshold_path)
                write_reports(validation, result_root / "validation")
                validation_rows.extend(
                    {
                        "model_id": spec.tag,
                        "dataset_name": dataset,
                        "method": row["method"],
                        "partition": "validation",
                        **{key: row.get(key) for key in ("accuracy", "actual_avg_samples", "valid_samples", "invalid_rate", "questions_total")},
                    }
                    for row in _selected_rows(validation, args.target_budget)
                )

                persisted = read_json(threshold_path)
                print(f"[retest CPU] {spec.tag}/{dataset}: held-out test using persisted thresholds", flush=True)
                test = evaluate_records(
                    _partition_records(
                        confidence_files,
                        dataset,
                        "test",
                        seed=args.seed,
                        fraction=args.validation_fraction,
                    ),
                    config=config,
                    phase="test",
                    threshold_calibration=persisted,
                    model_id=spec.tag,
                    dataset_name=dataset,
                )
                test["partition_provenance"] = provenance
                test["threshold_calibration_file"] = str(reported_threshold_path)
                write_reports(test, result_root / "test")
                new_rows = _selected_rows(test, args.target_budget)
                for row in new_rows:
                    if float(row["actual_avg_samples"]) > float(args.target_budget) + 1e-12:
                        raise ValueError(
                            f"Dynamic budget cap exceeded for {spec.tag}/{dataset}/{row['method']}: "
                            f"{row['actual_avg_samples']} > {args.target_budget}"
                        )
                for row in new_rows:
                    summary_rows.append(
                        {
                            "model_id": spec.tag,
                            "model_kind": spec.kind,
                            "model_family": spec.family if spec.family != "unknown" else (bundle.family or "unknown"),
                            "dataset_name": dataset,
                            "partition": "held_out_test",
                            **row,
                        }
                    )

                base_model = _base_model_id(spec)
                reference_sets: list[tuple[str, dict[str, dict[str, Any]]]] = []
                # ``eval_outputs_v2`` was generated from the already merged
                # RelaCaTS-v1 checkpoints.  Compare it with the matching
                # RelaCaTS-v1 retest tag, never with the independent author
                # CaTS/Self-Calibration tag.  Unknown custom specs retain the
                # historical fallback so a caller can still opt into that
                # comparison explicitly.
                if spec.kind in {"relacats_v1", "unknown"}:
                    reference_sets.append(
                        (
                            "eval_outputs_v2",
                            _load_reference_v2(reference_root, base_model, dataset, args.target_budget),
                        )
                    )
                reference_sets.append(
                    (
                        "legacy_table2_results",
                        _load_legacy_rows(legacy_results, spec, dataset, args.target_budget),
                    )
                )
                for row in new_rows:
                    method = canonical_method_name(row["method"])
                    for reference_name, reference_map in reference_sets:
                        old = reference_map.get(method)
                        old_accuracy = _metric(old, "accuracy")
                        new_accuracy = float(row["accuracy"])
                        comparisons.append(
                            {
                                "model_id": spec.tag,
                                "dataset_name": dataset,
                                "method": method,
                                "reference_name": reference_name,
                                "reference_found": old is not None,
                                "directly_comparable": False,
                                "comparison_note": "Fresh response pool and deterministic held-out test versus a prior aggregation; diagnostic only.",
                                "new_accuracy": new_accuracy,
                                "new_actual_avg_samples": float(row["actual_avg_samples"]),
                                "new_valid_samples": int(row["valid_samples"]),
                                "new_invalid_rate": float(row["invalid_rate"]),
                                "reference_accuracy": old_accuracy,
                                "reference_actual_avg_samples": _metric(old, "actual_avg_samples"),
                                "reference_valid_samples": _metric(old, "valid_samples"),
                                "reference_invalid_rate": _metric(old, "invalid_rate"),
                                "accuracy_delta": (
                                    new_accuracy - float(old_accuracy)
                                    if old_accuracy is not None
                                    else None
                                ),
                            }
                        )

        model_method_rows = _aggregate_rows(summary_rows, specs)
        retest_rows = [
            {
                "model_id": row["model_id"],
                "model_kind": row["model_kind"],
                "model_family": row["model_family"],
                "dataset_name": row["dataset_name"],
                "method": row["method"],
                "accuracy": row["accuracy"],
                "actual_avg_samples": row["actual_avg_samples"],
                "valid_samples": row["valid_samples"],
                "invalid_rate": row["invalid_rate"],
                "questions_total": row["questions_total"],
                "budget_target": args.target_budget,
                "budget_cap": row.get("budget_cap", args.target_budget),
            }
            for row in summary_rows
        ]
        atomic_write_json(staging / "summary.json", summary_rows)
        _write_csv(staging / "summary.csv", summary_rows)
        atomic_write_json(staging / "retest_summary.json", retest_rows)
        _write_csv(staging / "retest_summary.csv", retest_rows)
        atomic_write_json(staging / "validation_summary.json", validation_rows)
        _write_csv(staging / "validation_summary.csv", validation_rows)
        atomic_write_json(staging / "model_method_summary.json", model_method_rows)
        _write_csv(staging / "model_method_summary.csv", model_method_rows)
        atomic_write_json(staging / "comparisons.json", comparisons)
        _write_csv(staging / "comparisons.csv", comparisons)
        manifest = {
            "schema_version": EXPECTED_SCHEMA,
            "complete": True,
            "models": [spec.tag for spec in specs],
            "model_specs": [asdict(spec) for spec in specs],
            "datasets": list(args.datasets),
            "num_generations": args.num_generations,
            "target_budget": args.target_budget,
            "validation_fraction": args.validation_fraction,
            "seed": args.seed,
            "validation_only_threshold_selection": True,
            "test_reads_persisted_thresholds": True,
            "hard_dynamic_budget_cap": args.target_budget,
            "artifact_root": str(artifact_root),
            "reference_root": str(reference_root),
            "legacy_results": str(legacy_results),
            "legacy_excel_path": str(legacy_excel) if legacy_excel else None,
            "legacy_excel_found": bool(legacy_excel and legacy_excel.is_file()),
            "comparison_reference_policy": (
                "eval_outputs_v2 is matched to relacats_v1 tags; legacy table2 JSON/CSV is retained for all tags"
            ),
            "validated_bundles": len(bundle_stats),
            "validated_questions": validated_questions,
            "validated_samples": validated_samples,
            "bundle_stats": bundle_stats,
            "summary_rows": len(summary_rows),
            "comparison_rows": len(comparisons),
        }
        atomic_write_json(staging / "manifest.json", manifest)
        report_text = _report_markdown(manifest, retest_rows, comparisons)
        (staging / "retest_report.md").write_text(report_text, encoding="utf-8")

        # Commit only helper-owned paths.  Artifacts and logs generated by the
        # GPU phase are deliberately left untouched.
        for directory_name in ("results", "thresholds"):
            destination = output_root / directory_name
            if destination.exists():
                if not args.resume:
                    raise FileExistsError(f"Refusing to overwrite {destination}; pass --resume")
                shutil.rmtree(destination)
            os.replace(staging / directory_name, destination)
        for filename in (
            "summary.json",
            "summary.csv",
            "retest_summary.json",
            "retest_summary.csv",
            "validation_summary.json",
            "validation_summary.csv",
            "model_method_summary.json",
            "model_method_summary.csv",
            "comparisons.json",
            "comparisons.csv",
            "manifest.json",
            "retest_report.md",
        ):
            destination = output_root / filename
            if destination.exists() and not args.resume:
                raise FileExistsError(f"Refusing to overwrite {destination}; pass --resume")
            os.replace(staging / filename, destination)
        # Keep the report at the stable docs path used by the shell wrapper.
        _safe_write_text(docs_path, report_text, resume=args.resume)
    except BaseException:
        print(f"Incomplete CPU staging retained for diagnosis: {staging}", flush=True)
        raise
    else:
        shutil.rmtree(staging, ignore_errors=True)
    print(f"Complete old-model retest aggregation: {output_root}", flush=True)
    return output_root


def build_parser() -> argparse.ArgumentParser:
    package_root = Path(__file__).resolve().parents[1]
    default_reference = package_root / "outputs/eval_outputs_v2"
    default_legacy = package_root / "outputs/table2_baseline_self_calibration_tp2_serial_n32_budget16/table2_results.json"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        help="Model tags or TAG|KIND|FAMILY|MODEL_PATH specs",
    )
    parser.add_argument(
        "--model-spec",
        dest="model_specs",
        action="append",
        help="Repeatable TAG|KIND|FAMILY|MODEL_PATH model specification",
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--num-generations", type=int, default=DEFAULT_NUM_GENERATIONS)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-budget", type=int, default=16)
    parser.add_argument("--reference-root", default=str(default_reference))
    parser.add_argument("--legacy-results", default=str(default_legacy))
    parser.add_argument(
        "--legacy-excel",
        help=(
            "Optional path to the legacy Table-2 Excel file. It is recorded in "
            "the report for provenance; JSON/CSV remains the machine-readable reference."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.model_specs and args.models:
        args.model_specs = list(args.model_specs) + list(args.models)
    elif not args.model_specs:
        args.model_specs = args.models
    reaggregate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
