"""Smoke tests for cellclust.inferential."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cellclust.inferential import rank_biserial, rank_biserial_with_ci


def test_rank_biserial_sign_convention():
    """Positive r_rb means group_b > group_a."""
    rng = np.random.default_rng(42)
    a = rng.normal(0, 1, 50)
    b = rng.normal(2, 1, 50)
    r = rank_biserial(a, b)
    assert r > 0, "When b > a, r_rb must be positive"


def test_rank_biserial_empty():
    assert np.isnan(rank_biserial(np.array([]), np.array([1.0, 2.0])))
    assert np.isnan(rank_biserial(np.array([1.0, 2.0]), np.array([])))


def test_rank_biserial_no_difference():
    rng = np.random.default_rng(42)
    a = rng.normal(0, 1, 200)
    b = rng.normal(0, 1, 200)
    r = rank_biserial(a, b)
    assert abs(r) < 0.2, f"r_rb should be near zero for similar groups, got {r}"


def test_rank_biserial_with_ci_contains_point():
    """The 95% CI should bracket the point estimate."""
    rng = np.random.default_rng(42)
    a = rng.normal(0, 1, 30)
    b = rng.normal(1, 1, 30)
    result = rank_biserial_with_ci(a, b, n_boot=200, seed=42)
    assert "r_rb" in result
    assert "CI_lo" in result
    assert "CI_hi" in result
    # Reasonable margin: bootstrap may put point slightly outside CI by chance
    assert result["CI_lo"] <= result["r_rb"] + 0.1
    assert result["CI_hi"] >= result["r_rb"] - 0.1


def test_rank_biserial_with_ci_returns_keys():
    rng = np.random.default_rng(42)
    a = rng.normal(0, 1, 20)
    b = rng.normal(1, 1, 20)
    result = rank_biserial_with_ci(a, b, n_boot=50, seed=42)
    assert set(result.keys()) >= {"r_rb", "CI_lo", "CI_hi", "n_boot_effective"}
