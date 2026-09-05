"""Reproduce CaTS Table 1 calibration metrics and extend it with RelaCaTS.

This evaluator is intentionally CPU-only.  It consumes the question-sharded
JSON files produced by ``data_creation.generate_relational_data`` and reports
calibration quality for:

    P(True), SC, SSC, RelSSC, Full-RelSSC

The first three rows are the CaTS Table-1 baselines.  RelSSC and Full-RelSSC
are the RelaCaTS extensions.

Why this file exists
--------------------
The public CaTS implementation computes SC/SSC calibration at *question level*:
for each question it selects the majority/weighted-majority answer, uses that
answer's support ratio as confidence, and then computes 10-bin ECE across
questions.  ``utils/metric.py`` in the public CaTS repository uses equal-width
bins and multiplies ECE by 100 for reporting.

This script follows that public implementation as closely as possible while
also providing a corrected ECE implementation for diagnostics.

Important reproduction notes
----------------------------
1. The paper reports for Llama-3.1-8B-Instruct:

       Method      GSM8K   SVAMP
       P(True)      12.03   28.94
       SC            4.48    4.94
       SSC           3.42    3.75

   Exact numerical reproduction also depends on response generation, EDT,
   prompts, dataset revision, seed, answer extraction, and vLLM/model version.
   This file reproduces the *metric/protocol*; it cannot make mismatched
   generated responses equal the paper's released responses.

2. The public CaTS SC/SSC generator evaluates one question-level winner.  The
   public P(True) evaluator is a single-response evaluator.  When only a
   32-response question pool is available, this script uses a deterministic
   response slot (default slot 0) as the single-response P(True) baseline.  It
   additionally reports ``P(True)-all`` over every response as a diagnostic.

3. For GSM8K/SVAMP, the current RelaCaTS-v2 generator uses identity-only
   1x32 sampling.  Therefore conservative RelSSC (r_g=d_i=1) is mathematically
   the same confidence-weighted aggregation as SSC, apart from invalid-answer
   and tie-handling details.  A lower ECE on these two datasets should mainly
   be expected from Full-RelSSC dependency correction, not conservative
   RelSSC.  Relation-view gains should be tested on MCQ datasets separately.

Typical usage
-------------
Generate Llama teacher responses (current RelaCaTS generator; fixed-temperature
sampling rather than the paper's EDT implementation):

    python -m relacats_v2.data_creation.generate_relational_data \
      --model-name /path/to/meta-llama/Llama-3.1-8B-Instruct \
      --datasets gsm8k svamp \
      --split train \
      --max-questions 2000 \
      --num-views 4 --samples-per-view 8 --total-budget 32 \
      --relation-mode auto \
      --temperature 0.8 \
      --confidence-temperature 0.0 \
      --output-root relacats_v2/outputs/table1_llama_raw

Then evaluate:

    python -m relacats_v2.evaluation.reproduce_table1 \
      --input-root relacats_v2/outputs/table1_llama_raw \
      --datasets gsm8k svamp \
      --output-dir relacats_v2/outputs/table1_llama

For an MCQ experiment, ``--baseline-root`` may point to a separately generated
identity-only 32-response pool while ``--input-root`` points to the RelaCaTS
relation-view pool.  This keeps SC/SSC and RelSSC at the same total response
budget while allowing their sampling designs to differ.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import OrderedDict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from relacats_v2.core import (
    annotate_dependency_weights,
    compute_relssc,
    compute_relssc_full,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "relacats_v2/outputs/generated_data"
DEFAULT_OUTPUT = REPO_ROOT / "relacats_v2/outputs/table1_calibration"

PAPER_REFERENCE_ECE: dict[str, dict[str, float]] = {
    "gsm8k": {"P(True)": 12.03, "SC": 4.48, "SSC": 3.42},
    "svamp": {"P(True)": 28.94, "SC": 4.94, "SSC": 3.75},
}

INVALID_ANSWER = "<INVALID>"


@dataclass(frozen=True)
class Observation:
    method: str
    dataset: str
    question_id: str
    confidence: float
    correct: int
    predicted_answer: str | None
    gold_answer: str | None


@dataclass(frozen=True)
class MetricRow:
    dataset: str
    method: str
    n: int
    accuracy: float
    ece_paper: float
    ece_strict: float
    brier: float
    paper_reference_ece: float | None
    delta_from_paper: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        default=str(DEFAULT_INPUT),
        help="RelaCaTS question-sharded generated-data root.",
    )
    parser.add_argument(
        "--baseline-root",
        default=None,
        help=(
            "Optional separate root for P(True)/SC/SSC.  If omitted, the same "
            "--input-root is used for all methods."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=("gsm8k", "svamp"),
        help="Dataset directory names under <root>/<dataset>/questions/.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument(
        "--ptrue-slot",
        type=int,
        default=0,
        help=(
            "Deterministic response slot used for the single-response P(True) "
            "baseline.  Samples are sorted by (view_index, sample_index_in_view)."
        ),
    )
    parser.add_argument(
        "--baseline-scope",
        choices=("all", "identity"),
        default="all",
        help=(
            "Samples used by SC/SSC. 'all' matches the complete pool in the "
            "baseline root; 'identity' restricts to relation_id=g0."
        ),
    )
    parser.add_argument(
        "--invalid-policy",
        choices=("paper", "valid-only"),
        default="paper",
        help=(
            "'paper' keeps invalid extracted answers as an answer class for "
            "SC/SSC, matching the public CaTS generator most closely. "
            "'valid-only' drops them before aggregation."
        ),
    )
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--strategy-similarity-threshold", type=float, default=0.86)
    parser.add_argument(
        "--include-ptrue-all",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also report response-level P(True)-all as a diagnostic row.",
    )
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _finite_probability(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        return None
    return result


def _normalize_answer(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) == 1 and text.upper() in "ABCDE":
        return text.upper()
    # The generator already canonicalizes numeric answers.  Keep normalization
    # conservative here so evaluation never silently changes the task answer.
    return text


def _sample_answer(sample: Mapping[str, Any], *, keep_invalid: bool) -> str | None:
    valid = bool(sample.get("is_valid_answer", True))
    answer = _normalize_answer(
        sample.get("canonicalized_answer", sample.get("canonical_answer"))
    )
    if valid and answer is not None:
        return answer
    return INVALID_ANSWER if keep_invalid else None


def _sample_sort_key(sample: Mapping[str, Any]) -> tuple[int, int, str]:
    try:
        view = int(sample.get("view_index", 0))
    except (TypeError, ValueError):
        view = 0
    try:
        index = int(sample.get("sample_index_in_view", 0))
    except (TypeError, ValueError):
        index = 0
    return view, index, str(sample.get("sample_id", ""))


def _load_payloads(root: Path, dataset: str) -> list[dict[str, Any]]:
    question_dir = root / dataset / "questions"
    if not question_dir.exists():
        raise FileNotFoundError(f"Missing question directory: {question_dir}")
    paths = sorted(question_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No question JSON files under {question_dir}")
    payloads: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Question payload is not a JSON object: {path}")
        samples = payload.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"Question payload has no samples: {path}")
        payloads.append(payload)
    return payloads


def _index_payloads(payloads: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        question_id = str(payload.get("question_id", "")).strip()
        if not question_id:
            raise ValueError("Question payload missing question_id")
        if question_id in result:
            raise ValueError(f"Duplicate question_id: {question_id}")
        result[question_id] = payload
    return result


def _filtered_samples(
    payload: Mapping[str, Any],
    *,
    scope: str,
) -> list[dict[str, Any]]:
    samples = [dict(sample) for sample in payload["samples"] if isinstance(sample, dict)]
    samples.sort(key=_sample_sort_key)
    if scope == "identity":
        samples = [
            sample
            for sample in samples
            if str(sample.get("relation_id", "g0")) == "g0"
            or str(sample.get("relation_type", "")).strip().lower() == "identity"
        ]
    return samples


def _argmax_first(support: Mapping[str, float]) -> str | None:
    if not support:
        return None
    best_answer: str | None = None
    best_value = -math.inf
    for answer, value in support.items():
        if value > best_value:
            best_answer = answer
            best_value = value
    return best_answer


def _correct(prediction: str | None, gold: str | None) -> int:
    if prediction is None or prediction == INVALID_ANSWER or gold is None:
        return 0
    return int(_normalize_answer(prediction) == _normalize_answer(gold))


def _ptrue_observation(
    dataset: str,
    payload: Mapping[str, Any],
    *,
    slot: int,
    scope: str,
) -> Observation | None:
    samples = _filtered_samples(payload, scope=scope)
    if not samples:
        return None
    chosen = samples[slot % len(samples)]
    confidence = _finite_probability(chosen.get("confidence"))
    if confidence is None:
        return None
    predicted = _sample_answer(chosen, keep_invalid=False)
    gold = _normalize_answer(payload.get("gold_original_answer"))
    return Observation(
        method="P(True)",
        dataset=dataset,
        question_id=str(payload["question_id"]),
        confidence=confidence,
        correct=_correct(predicted, gold),
        predicted_answer=predicted,
        gold_answer=gold,
    )


def _ptrue_all_observations(
    dataset: str,
    payload: Mapping[str, Any],
    *,
    scope: str,
) -> list[Observation]:
    gold = _normalize_answer(payload.get("gold_original_answer"))
    result: list[Observation] = []
    for sample in _filtered_samples(payload, scope=scope):
        confidence = _finite_probability(sample.get("confidence"))
        if confidence is None:
            continue
        predicted = _sample_answer(sample, keep_invalid=False)
        result.append(
            Observation(
                method="P(True)-all",
                dataset=dataset,
                question_id=f"{payload['question_id']}::{sample.get('sample_id', len(result))}",
                confidence=confidence,
                correct=_correct(predicted, gold),
                predicted_answer=predicted,
                gold_answer=gold,
            )
        )
    return result


def _sc_observation(
    dataset: str,
    payload: Mapping[str, Any],
    *,
    scope: str,
    invalid_policy: str,
) -> Observation | None:
    keep_invalid = invalid_policy == "paper"
    samples = _filtered_samples(payload, scope=scope)
    support: "OrderedDict[str, float]" = OrderedDict()
    denominator = 0
    for sample in samples:
        answer = _sample_answer(sample, keep_invalid=keep_invalid)
        if answer is None:
            continue
        support.setdefault(answer, 0.0)
        support[answer] += 1.0
        denominator += 1
    if denominator <= 0:
        return None
    winner = _argmax_first(support)
    assert winner is not None
    confidence = support[winner] / denominator
    gold = _normalize_answer(payload.get("gold_original_answer"))
    return Observation(
        method="SC",
        dataset=dataset,
        question_id=str(payload["question_id"]),
        confidence=confidence,
        correct=_correct(winner, gold),
        predicted_answer=None if winner == INVALID_ANSWER else winner,
        gold_answer=gold,
    )


def _ssc_observation(
    dataset: str,
    payload: Mapping[str, Any],
    *,
    scope: str,
    invalid_policy: str,
) -> Observation | None:
    keep_invalid = invalid_policy == "paper"
    samples = _filtered_samples(payload, scope=scope)
    support: "OrderedDict[str, float]" = OrderedDict()
    total_weight = 0.0
    for sample in samples:
        confidence = _finite_probability(sample.get("confidence"))
        if confidence is None:
            continue
        answer = _sample_answer(sample, keep_invalid=keep_invalid)
        if answer is None:
            continue
        support.setdefault(answer, 0.0)
        support[answer] += confidence
        total_weight += confidence
    if total_weight <= 0 or not support:
        return None
    winner = _argmax_first(support)
    assert winner is not None
    score = support[winner] / total_weight
    gold = _normalize_answer(payload.get("gold_original_answer"))
    return Observation(
        method="SSC",
        dataset=dataset,
        question_id=str(payload["question_id"]),
        confidence=score,
        correct=_correct(winner, gold),
        predicted_answer=None if winner == INVALID_ANSWER else winner,
        gold_answer=gold,
    )


def _relssc_observation(
    dataset: str,
    payload: Mapping[str, Any],
) -> Observation | None:
    samples = [dict(sample) for sample in payload["samples"] if isinstance(sample, dict)]
    result = compute_relssc(samples, zero_weight_policy="skip", enforce_v1_weights=True)
    if not result.defined or result.top_answer is None:
        return None
    winner = result.top_answer
    score = result.score(winner)
    if score is None:
        return None
    gold = _normalize_answer(payload.get("gold_original_answer"))
    return Observation(
        method="RelSSC",
        dataset=dataset,
        question_id=str(payload["question_id"]),
        confidence=float(score),
        correct=_correct(winner, gold),
        predicted_answer=winner,
        gold_answer=gold,
    )


def _full_relssc_observation(
    dataset: str,
    payload: Mapping[str, Any],
    *,
    beta: float,
    similarity_threshold: float,
) -> Observation | None:
    samples = [dict(sample) for sample in payload["samples"] if isinstance(sample, dict)]
    weighted, _summary = annotate_dependency_weights(
        samples,
        beta=beta,
        similarity_threshold=similarity_threshold,
        answer_sensitive=True,
    )
    result = compute_relssc_full(weighted, zero_weight_policy="skip")
    if not result.defined or result.top_answer is None:
        return None
    winner = result.top_answer
    score = result.score(winner)
    if score is None:
        return None
    gold = _normalize_answer(payload.get("gold_original_answer"))
    return Observation(
        method="Full-RelSSC",
        dataset=dataset,
        question_id=str(payload["question_id"]),
        confidence=float(score),
        correct=_correct(winner, gold),
        predicted_answer=winner,
        gold_answer=gold,
    )


def calculate_ece_paper(
    y_true: Sequence[int], y_scores: Sequence[float], *, n_bins: int
) -> float:
    """Match the public CaTS ``utils.metric.calculate_ece`` implementation.

    Note: ``np.digitize(score=1.0, linspace(0,1,n_bins+1))-1`` gives index
    ``n_bins`` and the original loop does not include that bin.  We preserve
    that edge behavior here solely for paper reproduction.
    """

    if not y_true:
        return float("nan")
    scores = np.asarray(y_scores, dtype=float)
    labels = np.asarray(y_true, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(scores, bins) - 1
    ece = 0.0
    for index in range(n_bins):
        mask = bin_indices == index
        if np.any(mask):
            bin_accuracy = float(np.mean(labels[mask]))
            bin_confidence = float(np.mean(scores[mask]))
            bin_size = float(np.sum(mask)) / len(labels)
            ece += abs(bin_accuracy - bin_confidence) * bin_size
    return ece


def calculate_ece_strict(
    y_true: Sequence[int], y_scores: Sequence[float], *, n_bins: int
) -> float:
    """Conventional equal-width ECE with confidence=1 assigned to the last bin."""

    if not y_true:
        return float("nan")
    scores = np.asarray(y_scores, dtype=float)
    labels = np.asarray(y_true, dtype=float)
    clipped = np.clip(scores, 0.0, 1.0)
    # floor(score*n_bins), then clip 1.0 back into the final valid bin.
    indices = np.minimum((clipped * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for index in range(n_bins):
        mask = indices == index
        if np.any(mask):
            bin_accuracy = float(np.mean(labels[mask]))
            bin_confidence = float(np.mean(clipped[mask]))
            ece += abs(bin_accuracy - bin_confidence) * (float(np.sum(mask)) / len(labels))
    return ece


def _metric_row(dataset: str, method: str, observations: Sequence[Observation], n_bins: int) -> MetricRow:
    labels = [obs.correct for obs in observations]
    scores = [obs.confidence for obs in observations]
    if not labels:
        return MetricRow(
            dataset=dataset,
            method=method,
            n=0,
            accuracy=float("nan"),
            ece_paper=float("nan"),
            ece_strict=float("nan"),
            brier=float("nan"),
            paper_reference_ece=PAPER_REFERENCE_ECE.get(dataset, {}).get(method),
            delta_from_paper=None,
        )
    accuracy = 100.0 * sum(labels) / len(labels)
    paper_ece = 100.0 * calculate_ece_paper(labels, scores, n_bins=n_bins)
    strict_ece = 100.0 * calculate_ece_strict(labels, scores, n_bins=n_bins)
    brier = 100.0 * sum((score - label) ** 2 for score, label in zip(scores, labels)) / len(labels)
    reference = PAPER_REFERENCE_ECE.get(dataset, {}).get(method)
    delta = paper_ece - reference if reference is not None else None
    return MetricRow(
        dataset=dataset,
        method=method,
        n=len(labels),
        accuracy=accuracy,
        ece_paper=paper_ece,
        ece_strict=strict_ece,
        brier=brier,
        paper_reference_ece=reference,
        delta_from_paper=delta,
    )


def _read_generation_metadata(root: Path) -> dict[str, Any] | None:
    path = root / "generation_metadata.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _metadata_warnings(root: Path, datasets: Sequence[str]) -> list[str]:
    metadata = _read_generation_metadata(root)
    if metadata is None:
        return [f"No generation_metadata.json found under {root}"]
    warnings: list[str] = []
    model = str(metadata.get("model_name", ""))
    if "llama" not in model.lower() or "8b" not in model.lower():
        warnings.append(
            "Paper Table 1 uses Llama-3.1-8B-Instruct, but metadata model is " + model
        )
    temperature = metadata.get("temperature")
    if temperature is not None and not math.isclose(float(temperature), 0.8, abs_tol=1e-9):
        warnings.append(f"Paper generation temperature is 0.8; metadata has {temperature}")
    total_budget = metadata.get("total_budget")
    if total_budget is not None and int(total_budget) != 32:
        warnings.append(f"Paper Table-1 SSC/SC uses N=32; metadata has {total_budget}")
    profiles = metadata.get("dataset_profiles")
    if isinstance(profiles, dict):
        for dataset in datasets:
            profile = profiles.get(dataset)
            if isinstance(profile, dict) and int(profile.get("total_budget", 32)) != 32:
                warnings.append(
                    f"{dataset}: profile total_budget={profile.get('total_budget')}, expected 32"
                )
    return warnings


def _write_csv(path: Path, rows: Sequence[MetricRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "method",
        "n",
        "accuracy",
        "ece_paper",
        "ece_strict",
        "brier",
        "paper_reference_ece",
        "delta_from_paper",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record = asdict(row)
            for field in ("accuracy", "ece_paper", "ece_strict", "brier"):
                value = record[field]
                if isinstance(value, float) and math.isfinite(value):
                    record[field] = round(value, 4)
            if record["delta_from_paper"] is not None:
                record["delta_from_paper"] = round(float(record["delta_from_paper"]), 4)
            writer.writerow(record)


def _write_observations(path: Path, observations: Sequence[Observation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for observation in observations:
            handle.write(json.dumps(asdict(observation), ensure_ascii=False) + "\n")


def _print_matrix(rows: Sequence[MetricRow], *, title: str) -> None:
    print("\n" + title)
    print("=" * len(title))
    datasets = list(dict.fromkeys(row.dataset for row in rows))
    methods = list(dict.fromkeys(row.method for row in rows))
    width = max(14, max((len(method) for method in methods), default=6) + 2)
    header = f"{'Method':<{width}}" + "".join(f"{dataset:>12}" for dataset in datasets)
    print(header)
    print("-" * len(header))
    lookup = {(row.dataset, row.method): row for row in rows}
    for method in methods:
        line = f"{method:<{width}}"
        for dataset in datasets:
            row = lookup.get((dataset, method))
            value = row.ece_paper if row is not None else float("nan")
            cell = f"{value:.2f}" if math.isfinite(value) else "N/A"
            line += f"{cell:>12}"
        print(line)
    print("\nECE values are percentages; lower is better.")


def main() -> None:
    args = parse_args()
    if args.n_bins <= 0:
        raise ValueError("--n-bins must be positive")
    if args.ptrue_slot < 0:
        raise ValueError("--ptrue-slot must be non-negative")
    if not 0.0 <= args.beta <= 1.0:
        raise ValueError("--beta must lie in [0,1]")
    if not 0.0 <= args.strategy_similarity_threshold <= 1.0:
        raise ValueError("--strategy-similarity-threshold must lie in [0,1]")

    relational_root = resolve_path(args.input_root)
    baseline_root = resolve_path(args.baseline_root) if args.baseline_root else relational_root
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    warnings.extend(_metadata_warnings(baseline_root, args.datasets))
    if relational_root != baseline_root:
        warnings.extend(_metadata_warnings(relational_root, args.datasets))

    all_rows: list[MetricRow] = []
    all_observations: list[Observation] = []

    for dataset in args.datasets:
        baseline_payloads = _load_payloads(baseline_root, dataset)
        relational_payloads = _load_payloads(relational_root, dataset)
        baseline_index = _index_payloads(baseline_payloads)
        relational_index = _index_payloads(relational_payloads)

        shared = sorted(set(baseline_index) & set(relational_index))
        if not shared:
            # Different generation roots may use different stable IDs even for
            # the same source examples.  In that case align by source_index.
            base_by_source = {
                int(payload.get("source_index", -1)): payload
                for payload in baseline_payloads
                if payload.get("source_index") is not None
            }
            rel_by_source = {
                int(payload.get("source_index", -1)): payload
                for payload in relational_payloads
                if payload.get("source_index") is not None
            }
            common_sources = sorted(set(base_by_source) & set(rel_by_source))
            if not common_sources:
                raise ValueError(
                    f"{dataset}: baseline and relational roots have no shared question_id/source_index"
                )
            pairs = [(base_by_source[index], rel_by_source[index]) for index in common_sources]
            warnings.append(
                f"{dataset}: aligned baseline/relational payloads by source_index, not question_id"
            )
        else:
            pairs = [(baseline_index[qid], relational_index[qid]) for qid in shared]
            if len(shared) != len(baseline_index) or len(shared) != len(relational_index):
                warnings.append(
                    f"{dataset}: using {len(shared)} shared questions; baseline={len(baseline_index)}, "
                    f"relational={len(relational_index)}"
                )

        method_observations: dict[str, list[Observation]] = {
            "P(True)": [],
            "SC": [],
            "SSC": [],
            "RelSSC": [],
            "Full-RelSSC": [],
        }
        if args.include_ptrue_all:
            method_observations["P(True)-all"] = []

        identity_only_dataset = True
        for baseline_payload, relational_payload in pairs:
            relation_ids = {
                str(sample.get("relation_id", "g0"))
                for sample in relational_payload["samples"]
                if isinstance(sample, dict)
            }
            if relation_ids != {"g0"}:
                identity_only_dataset = False

            observation = _ptrue_observation(
                dataset,
                baseline_payload,
                slot=args.ptrue_slot,
                scope=args.baseline_scope,
            )
            if observation is not None:
                method_observations["P(True)"].append(observation)

            if args.include_ptrue_all:
                method_observations["P(True)-all"].extend(
                    _ptrue_all_observations(
                        dataset,
                        baseline_payload,
                        scope=args.baseline_scope,
                    )
                )

            observation = _sc_observation(
                dataset,
                baseline_payload,
                scope=args.baseline_scope,
                invalid_policy=args.invalid_policy,
            )
            if observation is not None:
                method_observations["SC"].append(observation)

            observation = _ssc_observation(
                dataset,
                baseline_payload,
                scope=args.baseline_scope,
                invalid_policy=args.invalid_policy,
            )
            if observation is not None:
                method_observations["SSC"].append(observation)

            observation = _relssc_observation(dataset, relational_payload)
            if observation is not None:
                method_observations["RelSSC"].append(observation)

            observation = _full_relssc_observation(
                dataset,
                relational_payload,
                beta=args.beta,
                similarity_threshold=args.strategy_similarity_threshold,
            )
            if observation is not None:
                method_observations["Full-RelSSC"].append(observation)

        if identity_only_dataset and baseline_root == relational_root:
            warnings.append(
                f"{dataset}: relational pool is identity-only; conservative RelSSC is expected "
                "to equal SSC up to invalid/tie policy."
            )

        method_order = ["P(True)", "SC", "SSC", "RelSSC", "Full-RelSSC"]
        if args.include_ptrue_all:
            method_order.append("P(True)-all")
        for method in method_order:
            observations = method_observations[method]
            all_observations.extend(observations)
            all_rows.append(_metric_row(dataset, method, observations, args.n_bins))

    paper_rows = [row for row in all_rows if row.method in {"P(True)", "SC", "SSC"}]
    extended_rows = [row for row in all_rows if row.method != "P(True)-all"]

    _write_csv(output_dir / "table1_paper_reproduction.csv", paper_rows)
    _write_csv(output_dir / "table1_relacats_extended.csv", extended_rows)
    if args.include_ptrue_all:
        _write_csv(output_dir / "table1_all_metrics.csv", all_rows)
    _write_observations(output_dir / "table1_observations.jsonl", all_observations)

    report = {
        "input_root": str(relational_root),
        "baseline_root": str(baseline_root),
        "datasets": list(args.datasets),
        "n_bins": args.n_bins,
        "ptrue_slot": args.ptrue_slot,
        "baseline_scope": args.baseline_scope,
        "invalid_policy": args.invalid_policy,
        "beta": args.beta,
        "strategy_similarity_threshold": args.strategy_similarity_threshold,
        "paper_reference_ece": PAPER_REFERENCE_ECE,
        "warnings": warnings,
        "rows": [asdict(row) for row in all_rows],
    }
    with (output_dir / "table1_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    _print_matrix(paper_rows, title="CaTS Table 1 protocol reproduction (paper-style ECE)")
    _print_matrix(extended_rows, title="Extended Table 1 with RelaCaTS (paper-style ECE)")

    print("\nOutputs:")
    print(f"  {output_dir / 'table1_paper_reproduction.csv'}")
    print(f"  {output_dir / 'table1_relacats_extended.csv'}")
    print(f"  {output_dir / 'table1_report.json'}")
    print(f"  {output_dir / 'table1_observations.jsonl'}")
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
