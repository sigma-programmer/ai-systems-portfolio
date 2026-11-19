import numpy as np
import pytest

from trajectory_eval.calibration import (IsotonicCalibrator, PlattCalibrator, brier, calibrate_judge, cohen_kappa,
                                         ece, tune_threshold)


def test_kappa_known_values():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert cohen_kappa([1, 1, 1, 1], [1, 1, 1, 1]) == 1.0
    assert cohen_kappa([1, 1, 1, 1], [1, 0, 1, 0]) == 0.0
    # classic example: po=0.7, pe=0.5 -> 0.4
    a = [1] * 5 + [0] * 5
    b = [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]
    assert cohen_kappa(a, b) == pytest.approx(0.6)
    assert cohen_kappa([1, 0], [0, 1]) == pytest.approx(-1.0)


def test_platt_recovers_logistic():
    rng = np.random.default_rng(0)
    s = rng.uniform(-3, 3, 4000)
    y = (rng.random(4000) < 1 / (1 + np.exp(-(2 * s - 1)))).astype(int)
    c = PlattCalibrator().fit(s, y)
    assert c.a == pytest.approx(2, abs=0.25) and c.b == pytest.approx(-1, abs=0.25)


def test_isotonic_is_monotone_and_fits_steps():
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    y = np.array([0, 1, 0, 1, 1, 1])
    c = IsotonicCalibrator().fit(s, y)
    p = c.predict(np.linspace(0, 1, 50))
    assert np.all(np.diff(p) >= -1e-12)
    assert c.predict([0.2])[0] == pytest.approx(0.5)  # pooled block
    assert c.predict([0.6])[0] == 1.0


def test_threshold_tuning_and_metrics():
    p = np.array([0.6, 0.7, 0.8, 0.9])
    y = np.array([0, 0, 1, 1])
    assert tune_threshold(p, y) == 0.8
    assert brier([1, 0], [1, 0]) == 0.0
    assert ece([0.9, 0.9], [1, 1]) == pytest.approx(0.1)


def test_calibration_improves_biased_judge():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    s = np.clip(0.75 + 0.12 * y + rng.normal(0, 0.06, 300), 0, 1)  # always > 0.5
    for method in ("platt", "isotonic"):
        rep, cal, thr = calibrate_judge(s, y, method=method)
        assert rep.pre.kappa == pytest.approx(0.0, abs=0.02)
        assert rep.post.kappa > 0.6
        assert rep.post.ece < rep.pre.ece
        assert 0 <= thr <= 1


def test_calibrate_input_validation():
    with pytest.raises(ValueError):
        calibrate_judge(np.array([0.1, 0.2]), np.array([0, 1]), folds=5)
