from relacats_v2.data_creation.build_hybrid_relsc_ssc_dataset import (
    hybrid_scores,
    numeric_ssc_scores,
    pure_relsc_scores,
)
from relacats_v2.model_training.train_hybrid_relsc_ssc import (
    _exact_weighted_allocation,
)


def _sample(answer, confidence, valid=True, relation_id="g0"):
    return {
        "canonicalized_answer": answer if valid else None,
        "is_valid_answer": valid,
        "confidence": confidence,
        "relation_weight": 1.0,
        "dependency_weight": 1.0,
        "relation_id": relation_id,
        "relation_type": "identity" if relation_id == "g0" else "invariant",
        "view_index": int(relation_id[1:]),
    }


def test_mcq_relsc_is_count_based_not_confidence_weighted():
    samples = [
        _sample("A", 0.05, relation_id="g0"),
        _sample("A", 0.05, relation_id="g1"),
        _sample("B", 0.99, relation_id="g2"),
    ]
    scores = pure_relsc_scores(samples)
    assert scores == {"A": 2 / 3, "B": 1 / 3}
    method, hybrid = hybrid_scores(samples, answer_type="option")
    assert method == "relsc"
    assert hybrid == scores


def test_numeric_ssc_uses_ptrue_weighting():
    samples = [
        _sample("10", 0.05),
        _sample("10", 0.05),
        _sample("20", 0.90),
    ]
    scores = numeric_ssc_scores(samples)
    assert abs(scores["10"] - 0.10) < 1e-12
    assert abs(scores["20"] - 0.90) < 1e-12
    method, hybrid = hybrid_scores(samples, answer_type="number")
    assert method == "ssc"
    assert hybrid == scores


def test_invalid_answers_do_not_enter_target_denominators():
    samples = [
        _sample("A", 0.2, relation_id="g0"),
        _sample("B", 0.8, relation_id="g1"),
        _sample("X", 1.0, valid=False, relation_id="g2"),
    ]
    assert pure_relsc_scores(samples) == {"A": 0.5, "B": 0.5}

    numeric = [_sample("1", 0.2), _sample("2", 0.8), _sample("3", 1.0, False)]
    scores = numeric_ssc_scores(numeric)
    assert abs(scores["1"] - 0.2) < 1e-12
    assert abs(scores["2"] - 0.8) < 1e-12


def test_exact_weighted_allocation_preserves_global_budget():
    weights = [1, 1, 3, 1, 1, 1, 1, 3, 1]

    train = _exact_weighted_allocation(100_000, weights)
    assert sum(train) == 100_000
    assert train[2] in {23_076, 23_077}
    assert train[7] in {23_076, 23_077}
    assert all(value in {7_692, 7_693} for i, value in enumerate(train) if i not in {2, 7})

    evaluation = _exact_weighted_allocation(1_000, weights)
    assert sum(evaluation) == 1_000
