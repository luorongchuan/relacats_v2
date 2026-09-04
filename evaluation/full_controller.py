"""CPU controller evaluation for full RelaCaTS.

Input rows must contain an explicit parsed answer, ``confidence`` q_hat,
``fragility`` f_hat, and preferably the original ``response`` text so the
same dependency correction used offline can be applied online.  The controller
implements Eqs. (34)--(37): STOP / SAMPLE / INTERVENE.

This CPU stage does not fabricate a relational intervention response.  An
INTERVENE decision is emitted explicitly so a GPU orchestration layer can
construct a relation challenge.  That separation prevents CPU aggregation
from silently pretending an intervention was executed.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from relacats_v2.common import read_jsonl
from relacats_v2.core import ControllerAction, controller_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tau-support", type=float, default=0.8)
    parser.add_argument("--tau-fragility", type=float, default=0.25)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--strategy-similarity-threshold", type=float, default=0.86)
    parser.add_argument("--min-valid", type=int, default=2)
    parser.add_argument("--max-budget", type=int, default=16)
    return parser.parse_args()


def _generation_index(record: dict[str, Any], fallback: int) -> int:
    value = record.get("generation_index", record.get("sample_index_in_view", fallback))
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def main() -> None:
    args = parse_args()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(read_jsonl(Path(args.input).expanduser().resolve())):
        qid = str(record.get("question_id", f"legacy:{index}"))
        copied = dict(record)
        copied["_generation_index"] = _generation_index(copied, index)
        grouped[qid].append(copied)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {action.value: 0 for action in ControllerAction}
    used_total = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for question_id in sorted(grouped):
            records = sorted(
                grouped[question_id],
                key=lambda item: (item["_generation_index"], str(item.get("sample_id", ""))),
            )[: args.max_budget]
            final_state = None
            used = 0
            for index in range(len(records)):
                state = controller_state(
                    records[: index + 1],
                    tau_support=args.tau_support,
                    tau_fragility=args.tau_fragility,
                    beta=args.beta,
                    similarity_threshold=args.strategy_similarity_threshold,
                    min_valid=args.min_valid,
                )
                used = index + 1
                final_state = state
                if state.action in {ControllerAction.STOP, ControllerAction.INTERVENE}:
                    break
            if final_state is None:
                continue
            summary[final_state.action.value] += 1
            used_total += used
            row = {
                "question_id": question_id,
                "action": final_state.action.value,
                "leader": final_state.leader,
                "support_ratio": final_state.support_ratio,
                "leader_fragility": final_state.leader_fragility,
                "valid_sample_count": final_state.valid_sample_count,
                "actual_samples": used,
                "max_budget": args.max_budget,
                "intervention_executed": False,
                "intervention_required": final_state.action is ControllerAction.INTERVENE,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    questions = sum(summary.values())
    report = {
        "questions": questions,
        "actual_avg_samples": used_total / questions if questions else 0.0,
        "actions": summary,
        "tau_support": args.tau_support,
        "tau_fragility": args.tau_fragility,
        "beta": args.beta,
        "strategy_similarity_threshold": args.strategy_similarity_threshold,
        "max_budget": args.max_budget,
        "note": "INTERVENE is surfaced but not executed by this CPU stage",
    }
    report_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
