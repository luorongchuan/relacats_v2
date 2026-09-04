from relacats_v2.evaluation.method_names import (
    TABLE2_METHOD_ORDER,
    canonical_method_name,
    canonicalize_report_methods,
)


def test_legacy_relacats_labels_are_normalized_without_touching_asc_baseline():
    assert canonical_method_name("CaTS-SC") == "RelaCaTS-SC"
    assert canonical_method_name("CaTS-ES") == "RelaCaTS-ES"
    assert canonical_method_name("CaTS-ASC") == "RelaCaTS-ASC"
    assert canonical_method_name("cats_asc") == "RelaCaTS-ASC"
    assert canonical_method_name("relacats-es") == "RelaCaTS-ES"
    assert canonical_method_name("ASC") == "ASC"


def test_report_normalization_handles_rows_and_curve_keys():
    report = {
        "fixed_budget_results": [{"method": "CaTS-SC", "budget": 16}],
        "dynamic_budget_matches": [{"method": "CaTS-ASC", "budget": 16}],
        "threshold_curves": {
            "CaTS-ES": [{"method": "CaTS-ES", "threshold": 0.5}],
            "RelaCaTS-ES": [{"method": "RelaCaTS-ES", "threshold": 0.6}],
        },
        "method_metadata": {
            "CaTS-SC": {"implementation_status": "legacy"},
            "RelaCaTS-SC": {"score_source": "calibrated confidence"},
        },
        "method_order": ["CaTS-SC", "RelaCaTS-SC", "ASC"],
    }
    normalized = canonicalize_report_methods(report)
    assert normalized["fixed_budget_results"][0]["method"] == "RelaCaTS-SC"
    assert normalized["dynamic_budget_matches"][0]["method"] == "RelaCaTS-ASC"
    assert list(normalized["threshold_curves"]) == ["RelaCaTS-ES"]
    assert len(normalized["threshold_curves"]["RelaCaTS-ES"]) == 2
    assert normalized["threshold_curves"]["RelaCaTS-ES"][0]["method"] == "RelaCaTS-ES"
    assert list(normalized["method_metadata"]) == ["RelaCaTS-SC"]
    assert normalized["method_metadata"]["RelaCaTS-SC"] == {
        "implementation_status": "legacy",
        "score_source": "calibrated confidence",
    }
    assert normalized["method_order"] == ["RelaCaTS-SC", "ASC"]


def test_table2_order_has_all_baselines_and_relacats_rows():
    assert TABLE2_METHOD_ORDER == (
        "SC",
        "CISC",
        "Self-Certainty",
        "RelaCaTS-SC",
        "Best-of-N",
        "RelaCaTS-ES",
        "ASC",
        "RelaCaTS-ASC",
        "ESC",
        "RASC",
    )
