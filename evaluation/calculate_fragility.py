"""Predict the full RelaCaTS fragility score f_hat for saved responses.

The same LM is queried with the fragility instruction I_frag.  This script is
separate from ``calculate_confidence.py`` so old CaTS/RelaCaTS-v2 confidence
artifacts remain immutable.  Output rows preserve every input field and add
``fragility`` / ``predicted_fragility``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from relacats_v2.common import confidence_from_logprobs, read_jsonl
from relacats_v2.core import fragility_suffix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, help="confidence JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def _batches(values: list[dict[str, Any]], size: int):
    if size <= 0:
        raise ValueError("batch-size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def main() -> None:
    args = parse_args()
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError("vLLM is required for fragility inference") from exc

    rows = list(read_jsonl(Path(args.input).expanduser().resolve()))
    if not rows:
        raise ValueError("input confidence artifact is empty")
    engine = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        seed=args.seed,
    )
    sampling = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    suffix = fragility_suffix(args.model)
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for batch in _batches(rows, args.batch_size):
            prompts = [
                f"{record.get('prompt', record.get('transformed_prompt', ''))} "
                f"{record.get('response', '')} {suffix}"
                for record in batch
            ]
            outputs = engine.generate(prompts, sampling, use_tqdm=False)
            if len(outputs) != len(batch):
                raise RuntimeError("vLLM returned a mismatched fragility batch")
            for record, output in zip(batch, outputs):
                candidate = output.outputs[0]
                if not candidate.logprobs or not candidate.logprobs[0]:
                    raise RuntimeError(
                        f"missing fragility logprobs for {record.get('sample_id')}"
                    )
                yes, no, yes_found, no_found = confidence_from_logprobs(
                    candidate.logprobs[0]
                )
                if not math.isfinite(yes):
                    raise RuntimeError("non-finite fragility score")
                enriched = dict(record)
                enriched.update(
                    {
                        "fragility": float(yes),
                        "predicted_fragility": float(yes),
                        "fragility_false_prob": float(no),
                        "fragility_yes_token_found_top20": bool(yes_found),
                        "fragility_no_token_found_top20": bool(no_found),
                        "fragility_definition": "P(Yes | I_frag)",
                    }
                )
                handle.write(json.dumps(enriched, ensure_ascii=False) + "\n")
    print(f"Wrote full RelaCaTS fragility artifact: {output_path}")


if __name__ == "__main__":
    main()
