"""Smoke tests for cellclust.clustering."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cellclust.clustering import fit_predict, reorder_clusters_by_mean
from cellclust.config import ALGORITHMS


@pytest.fixture
def synthetic_two_cluster():
    """1D synthetic data: two well-separated Gaussian clusters."""
    rng = np.random.default_rng(42)
    X = np.concatenate([
        rng.normal(loc=-2.0, scale=0.5, size=300),
        rng.normal(loc=+2.0, scale=0.5, size=300),
    ]).reshape(-1, 1)
    return X


@pytest.mark.parametrize("algorithm", [
    "K-Means", "GMM", "Spectral", "Agglom-Ward",
    "BIRCH", "Agglom-Average", "BayesianGMM",
    # SOM+Ward requires minisom; tested separately if installed
])
def test_fit_predict_two_clusters(synthetic_two_cluster, algorithm):
    X = synthetic_two_cluster
    labels = fit_predict(algorithm, X, k=2)
    assert isinstance(labels, np.ndarray)
    assert labels.shape == (X.shape[0],)
    assert set(np.unique(labels)).issubset({0, 1})


def test_fit_predict_invalid_algorithm():
    X = np.random.randn(100, 1)
    with pytest.raises(ValueError):
        fit_predict("NotARealAlgorithm", X, k=2)


def test_reorder_clusters_by_mean(synthetic_two_cluster):
    X = synthetic_two_cluster
    labels = fit_predict("K-Means", X, k=2)
    reordered = reorder_clusters_by_mean(labels, X.flatten())
    # After reordering, cluster 0 should have lower mean than cluster 1
    mean_0 = X[reordered == 0].mean()
    mean_1 = X[reordered == 1].mean()
    assert mean_0 < mean_1


def test_algorithms_list_complete():
    """Ensure no algorithm is forgotten from the canonical list."""
    expected = {
        "K-Means", "GMM", "Spectral", "Agglom-Ward",
        "BIRCH", "Agglom-Average", "SOM+Ward", "BayesianGMM",
    }
    assert set(ALGORITHMS) == expected
