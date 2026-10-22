import numpy as np
import pytest

from tvt.evaluation import (
    bootstrap_ci,
    evaluate,
    expected_calibration_error,
    macro_f1,
    mcnemar_test,
    paired_bootstrap_diff,
)


def test_ece_is_zero_for_perfectly_calibrated_confident_predictions():
    probs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    assert expected_calibration_error(probs, np.array([0, 1, 0])) == pytest.approx(0.0)


def test_ece_detects_overconfidence():
    # Always 90 % confident, right only half of the time -> ECE = 0.4
    probs = np.tile([0.9, 0.1], (10, 1))
    y = np.array([0, 1] * 5)
    assert expected_calibration_error(probs, y) == pytest.approx(0.4)


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    pred = np.where(rng.random(500) < 0.8, y, 1 - y)
    lo, hi = bootstrap_ci(y, pred, macro_f1, n_resamples=300)
    assert lo <= macro_f1(y, pred) <= hi
    assert hi - lo < 0.15


def test_mcnemar_identical_predictions():
    y = np.array([0, 1, 1, 0])
    assert mcnemar_test(y, y, y)["p_value"] == 1.0


def test_mcnemar_exact_matches_binomial():
    y = np.zeros(20, dtype=int)
    a = np.zeros(20, dtype=int)
    b = np.zeros(20, dtype=int)
    a[:10] = 1  # A wrong on 10 examples where B is right
    res = mcnemar_test(y, a, b)
    assert (res["a_only_correct"], res["b_only_correct"], res["method"]) == (0, 10, "exact")
    assert res["p_value"] == pytest.approx(2 * 0.5**10)


def test_mcnemar_chi2_branch():
    y = np.zeros(200, dtype=int)
    a = np.zeros(200, dtype=int)
    a[:80] = 1
    b = np.zeros(200, dtype=int)
    b[80:100] = 1
    res = mcnemar_test(y, a, b)  # b=20 (A right only), c=80 (B right only)
    assert res["method"] == "chi2_continuity"
    assert res["statistic"] == pytest.approx((60 - 1) ** 2 / 100)
    assert res["p_value"] < 1e-6


def test_paired_bootstrap_sign_and_significance():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 1000)
    weak = np.where(rng.random(1000) < 0.7, y, 1 - y)
    strong = np.where(rng.random(1000) < 0.95, y, 1 - y)
    res = paired_bootstrap_diff(y, weak, strong, n_resamples=300)
    assert res["diff"] > 0.15
    assert res["ci95"][0] > 0
    assert res["p_value"] < 0.01


def test_evaluate_outputs():
    y = np.array([0, 1, 2, 2])
    pred = np.array([0, 1, 2, 1])
    probs = np.eye(3)[pred]
    out = evaluate(y, pred, probs, ["a", "b", "c"], n_bootstrap=50)
    assert out["accuracy"] == pytest.approx(0.75)
    assert np.array(out["confusion_matrix"]).sum() == 4
    assert set(out) >= {"macro_f1", "macro_f1_ci95", "ece", "per_class"}
