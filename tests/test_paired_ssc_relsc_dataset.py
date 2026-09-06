from relacats_v2.data_creation.build_paired_ssc_relsc_dataset import (
    build_question_rows,
    pure_relsc_scores,
    weighted_ssc_scores,
)


def _sample(answer, confidence, *, valid=True, qid="q1", sid="s0"):
    return {
        "question_id": qid,
        "sample_id": sid,
        "canonicalized_answer": answer if valid else None,
        "is_valid_answer": valid,
        "confidence": confidence,
        "transformed_prompt": "Question\n",
        "response": "Answer",
    }


def test_weighted_ssc_uses_ptrue_mass():
    samples = [
        _sample("A", 0.1, sid="a1"),
        _sample("A", 0.2, sid="a2"),
        _sample("B", 0.7, sid="b1"),
    ]
    scores = weighted_ssc_scores(samples)
    assert abs(scores["A"] - 0.3) < 1e-12
    assert abs(scores["B"] - 0.7) < 1e-12


def test_relsc_is_count_based_not_confidence_weighted():
    samples = [
        _sample("A", 0.01, sid="a1"),
        _sample("A", 0.01, sid="a2"),
        _sample("B", 0.99, sid="b1"),
    ]
    scores = pure_relsc_scores(samples)
    assert scores == {"A": 2 / 3, "B": 1 / 3}


def test_paired_rows_share_inputs_but_have_two_targets():
    identity = {
        "samples": [
            _sample("A", 0.2, sid="i1"),
            _sample("B", 0.8, sid="i2"),
        ]
    }
    relation = {
        "samples": [
            _sample("A", 0.1, sid="r1"),
            _sample("A", 0.1, sid="r2"),
            _sample("B", 0.9, sid="r3"),
        ]
    }
    rows, summary = build_question_rows(identity, relation, dataset="arc_easy")
    assert summary["defined"]
    assert len(rows) == 2
    row_a = next(row for row in rows if row["answer"] == "A")
    row_b = next(row for row in rows if row["answer"] == "B")
    assert abs(row_a["ssc_consistency"] - 0.2) < 1e-12
    assert abs(row_b["ssc_consistency"] - 0.8) < 1e-12
    assert abs(row_a["relsc_consistency"] - 2 / 3) < 1e-12
    assert abs(row_b["relsc_consistency"] - 1 / 3) < 1e-12
    assert row_a["paired_selection_consistency"] == row_a["ssc_consistency"]


def test_numeric_fallback_makes_targets_identical():
    identity = {
        "samples": [
            _sample("10", 0.25, sid="i1"),
            _sample("20", 0.75, sid="i2"),
        ]
    }
    rows, summary = build_question_rows(identity, None, dataset="gsm8k")
    assert summary["defined"]
    for row in rows:
        assert row["ssc_consistency"] == row["relsc_consistency"]
