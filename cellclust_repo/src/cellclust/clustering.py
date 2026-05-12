"""
cellclust.clustering
====================
Unified ``fit_predict`` API for the eight clustering algorithms
benchmarked in the manuscript.

Algorithms (4 paradigms):
  Centroid-based    : K-Means
  Probabilistic     : GMM, BayesianGMM
  Graph-based       : Spectral
  Hierarchical      : Agglom-Ward, Agglom-Average, BIRCH
  Self-organizing   : SOM+Ward

For tractability on large datasets, Spectral and Agglomerative methods
are fitted on a random sub-sample and propagated to the full dataset
by nearest-centroid Euclidean assignment in z-score space.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist

from sklearn.cluster import (
    KMeans, SpectralClustering, AgglomerativeClustering, Birch
)
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

try:
    from minisom import MiniSom
    _HAS_MINISOM = True
except ImportError:
    _HAS_MINISOM = False

from .config import RANDOM_STATE, ALGORITHMS


# ---------------------------------------------------------------
# Hyperparameter reference (matches the manuscript audit table)
# ---------------------------------------------------------------
HYPERPARAMETERS = {
    "K-Means": {
        "n_init":       30,
        "init":         "k-means++",
        "max_iter":     300,
        "tol":          1e-6,
        "algorithm":    "lloyd",
        "random_state": RANDOM_STATE,
    },
    "GMM": {
        "covariance_type": "full",
        "tol":             1e-3,
        "reg_covar":       1e-6,
        "max_iter":        300,
        "n_init":          10,
        "init_params":     "kmeans",
        "random_state":    RANDOM_STATE,
    },
    "Spectral": {
        "affinity":      "rbf",
        "gamma":         1.0,
        "assign_labels": "kmeans",
        "n_init":        10,
        "sub_sample":    4000,
        "random_state":  RANDOM_STATE,
    },
    "Agglom-Ward": {
        "linkage":      "ward",
        "metric":       "euclidean",
        "sub_sample":   5000,
        "random_state": RANDOM_STATE,
    },
    "BIRCH": {
        "threshold":        0.05,
        "branching_factor": 50,
    },
    "Agglom-Average": {
        "linkage":      "average",
        "metric":       "euclidean",
        "sub_sample":   5000,
        "random_state": RANDOM_STATE,
    },
    "SOM+Ward": {
        "topology":      "1xN rectangular grid (N=max(4*k, 12))",
        "sigma":         1.0,
        "learning_rate": 0.5,
        "iterations":    2000,
        "random_state":  RANDOM_STATE,
    },
    "BayesianGMM": {
        "covariance_type":                 "full",
        "weight_concentration_prior_type": "dirichlet_process",
        "weight_concentration_prior":      "1/k (default)",
        "tol":                             1e-3,
        "reg_covar":                       1e-6,
        "max_iter":                        400,
        "n_init":                          5,
        "random_state":                    RANDOM_STATE,
    },
}


def _subsample_and_propagate(X, fitter, k, cap, seed=RANDOM_STATE):
    """Fit on sub-sample, propagate to full dataset by nearest centroid."""
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(cap, n), replace=False)
    sub_labels = fitter.fit_predict(X[idx])
    centroids = np.array([
        X[idx][sub_labels == c].mean(axis=0) for c in range(k)
    ])
    return np.argmin(cdist(X, centroids), axis=1)


def fit_predict(algorithm: str, X: np.ndarray, k: int,
                 seed: int = RANDOM_STATE) -> np.ndarray:
    """Fit any of the 8 algorithms and return integer cluster labels.

    Parameters
    ----------
    algorithm : str
        One of ``ALGORITHMS``.
    X : np.ndarray, shape (n_samples, n_features)
        Z-scored feature matrix.
    k : int
        Number of clusters (2 to 8 in this study).
    seed : int
        Random seed; defaults to ``RANDOM_STATE``.

    Returns
    -------
    labels : np.ndarray, shape (n_samples,)
        Integer cluster assignments in ``[0, k-1]``.
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm: {algorithm!r}. "
            f"Must be one of {ALGORITHMS}."
        )

    if algorithm == "K-Means":
        return KMeans(
            n_clusters=k, n_init=30, init="k-means++",
            max_iter=300, tol=1e-6, algorithm="lloyd",
            random_state=seed,
        ).fit(X).predict(X)

    if algorithm == "GMM":
        return GaussianMixture(
            n_components=k, covariance_type="full",
            tol=1e-3, reg_covar=1e-6, max_iter=300,
            n_init=10, init_params="kmeans",
            random_state=seed,
        ).fit(X).predict(X)

    if algorithm == "Spectral":
        n = len(X)
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=min(4000, n), replace=False)
        model = SpectralClustering(
            n_clusters=k, affinity="rbf", gamma=1.0,
            assign_labels="kmeans", n_init=10,
            random_state=seed,
        )
        sub_labels = model.fit_predict(X[idx])
        centroids = np.array([
            X[idx][sub_labels == c].mean(axis=0) for c in range(k)
        ])
        return np.argmin(cdist(X, centroids), axis=1)

    if algorithm == "Agglom-Ward":
        fitter = AgglomerativeClustering(
            n_clusters=k, linkage="ward", metric="euclidean"
        )
        return _subsample_and_propagate(X, fitter, k, cap=5000, seed=seed)

    if algorithm == "BIRCH":
        return Birch(
            n_clusters=k, threshold=0.05, branching_factor=50,
        ).fit(X).predict(X)

    if algorithm == "Agglom-Average":
        fitter = AgglomerativeClustering(
            n_clusters=k, linkage="average", metric="euclidean"
        )
        return _subsample_and_propagate(X, fitter, k, cap=5000, seed=seed)

    if algorithm == "SOM+Ward":
        if not _HAS_MINISOM:
            raise ImportError(
                "SOM+Ward requires minisom. "
                "Install with: pip install minisom"
            )
        n_neurons = max(4 * k, 12)
        som = MiniSom(
            1, n_neurons, X.shape[1],
            sigma=1.0, learning_rate=0.5,
            random_seed=seed,
        )
        som.random_weights_init(X)
        som.train_random(X, 2000)
        protos = som.get_weights().reshape(-1, X.shape[1])
        proto_labels = AgglomerativeClustering(
            n_clusters=k, linkage="ward"
        ).fit_predict(protos)
        labels = np.zeros(len(X), dtype=int)
        for i, x in enumerate(X):
            bmu_idx = int(np.argmin(np.linalg.norm(protos - x, axis=1)))
            labels[i] = proto_labels[bmu_idx]
        return labels

    if algorithm == "BayesianGMM":
        return BayesianGaussianMixture(
            n_components=k, covariance_type="full",
            weight_concentration_prior_type="dirichlet_process",
            tol=1e-3, reg_covar=1e-6, max_iter=400,
            n_init=5, random_state=seed,
        ).fit(X).predict(X)

    raise RuntimeError(f"unreachable: {algorithm}")


def reorder_clusters_by_mean(labels: np.ndarray,
                               values: np.ndarray) -> np.ndarray:
    """Relabel clusters so 0 = smallest mean, k-1 = largest.

    Ensures a consistent Low/High interpretation across algorithms
    regardless of their internal labelling order.
    """
    k = len(np.unique(labels))
    means = [values[labels == c].mean() for c in range(k)]
    order = np.argsort(means)
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping[x] for x in labels])
