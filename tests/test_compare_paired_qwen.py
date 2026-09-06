from relacats_v2.evaluation.compare_paired_qwen import _auc_binary, _ece


def test_auc_binary_perfect_ranking():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert _auc_binary(labels, scores) == 1.0


def test_auc_binary_ties_are_half_credit():
    labels = [0, 1]
    scores = [0.5, 0.5]
    assert _auc_binary(labels, scores) == 0.5


def test_ece_zero_for_perfectly_matched_two_bins():
    labels = [0, 1]
    scores = [0.0, 1.0]
    assert _ece(labels, scores, bins=10) == 0.0
