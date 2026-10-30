import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from abtest.stats import cuped, holm, sample_size_per_arm, srm_pvalue, welch  # noqa: E402


def test_welch_matches_scipy():
    rng = np.random.default_rng(0)
    a, b = rng.normal(10, 2, 300), rng.normal(10.5, 3, 250)
    ours = welch(a, b).p_value
    ref = stats.ttest_ind(b, a, equal_var=False).pvalue
    assert ours == pytest.approx(ref, rel=1e-6)


def test_sample_size_textbook():
    # sd=1, MDE=0.2 sd, alpha=.05, power=.8  ->  ~393 por grupo
    assert sample_size_per_arm(1, 0.2) == 393


def test_srm_detects_imbalance():
    assert srm_pvalue(5000, 5000) == pytest.approx(1.0)
    assert srm_pvalue(5300, 4700) < 0.001


def test_cuped_reduces_variance_and_keeps_mean():
    rng = np.random.default_rng(1)
    x = rng.normal(100, 20, 5000)
    y = x + rng.normal(0, 5, 5000)
    adj, theta = cuped(y, x)
    assert adj.var() < 0.1 * y.var()
    assert adj.mean() == pytest.approx(y.mean())
    assert theta == pytest.approx(1, abs=0.05)


def test_holm_monotone_and_bounded():
    out = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert out["a"] == pytest.approx(0.03)
    assert out["c"] == pytest.approx(0.06)
    assert out["b"] == pytest.approx(0.06)   # nunca menor que el anterior
    assert all(0 <= v <= 1 for v in out.values())
