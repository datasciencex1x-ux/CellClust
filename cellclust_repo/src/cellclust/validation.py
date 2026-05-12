"""
cellclust.validation
====================
Internal validation metrics and composite score for the benchmark:

  - Silhouette coefficient (higher is better, range [-1, 1])
  - Calinski-Harabasz index (higher is better)
  - Davies-Bouldin index   (lower is better)
  - Composite score: panel-normalized average of the three metrics
  - Pairwise Adjusted Rand Index across algorithms
  - Bootstrap 95% CI on the silhouette coefficient

Composite construction
----------------------
Each index is min-max normalized within each (Muscle, k) panel. The
Davies-Bouldin index is inverted (1/DB) so all three metrics are
higher-is-better. Composite = (Sil_n + CH_n + (1/DB)_n) / 3, unit-less
on [0, 1].
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score,
    davies_bouldin_score, adjusted_rand_score,
)

from .clustering import fit_predict
from .config import RANDOM_STATE


def internal_indices(X: np.ndarray, labels: np.ndarray) -> dict:
    """Compute Silhouette, Calinski-Harabasz, and Davies-Bouldin for a partition.

    Returns NaN for any index that cannot be computed (e.g., single-cluster
    degenerate solution).
    """
    if len(np.unique(labels)) < 2:
        return {"Silhouette": np.nan, "CH": np.nan, "DB": np.nan}
    try:
        sil = silhouette_score(
            X, labels,
            sample_size=min(5000, len(X)),
            random_state=RANDOM_STATE,
        )
    except Exception:
        sil = np.nan
    try:
        ch = calinski_harabasz_score(X, labels)
    except Exception:
        ch = np.nan
    try:
        db = davies_bouldin_score(X, labels)
    except Exception:
        db = np.nan
    return {"Silhouette": sil, "CH": ch, "DB": db}


def composite_score(df_indices: pd.DataFrame) -> pd.DataFrame:
    """Add Sil_n, CH_n, DBinv_n, Composite columns to a benchmark frame.

    Parameters
    ----------
    df_indices : pd.DataFrame
        Must contain columns ``["Muscle", "k", "Algorithm",
        "Silhouette", "CH", "DB"]``.
    """
    df = df_indices.copy()

    def _minmax(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-12)

    df["Sil_n"]   = df.groupby(["Muscle", "k"])["Silhouette"].transform(_minmax)
    df["CH_n"]    = df.groupby(["Muscle", "k"])["CH"].transform(_minmax)
    df["DBinv_n"] = df.groupby(["Muscle", "k"])["DB"].transform(
        lambda s: _minmax(1.0 / s)
    )
    df["Composite"] = (df["Sil_n"] + df["CH_n"] + df["DBinv_n"]) / 3.0
    return df


def ari_matrix(labels_by_algorithm: dict[str, np.ndarray]) -> pd.DataFrame:
    """Pairwise Adjusted Rand Index between algorithms (square matrix)."""
    algs = list(labels_by_algorithm.keys())
    mat = pd.DataFrame(
        np.zeros((len(algs), len(algs))), index=algs, columns=algs
    )
    for a in algs:
        for b in algs:
            mat.loc[a, b] = adjusted_rand_score(
                labels_by_algorithm[a], labels_by_algorithm[b]
            )
    return mat


def bootstrap_silhouette_ci(algorithm: str, X: np.ndarray, k: int,
                              n_boot: int = 100,
                              seed: int = RANDOM_STATE) -> dict:
    """Bootstrap 95% confidence interval for the silhouette coefficient.

    In each iteration, n samples are drawn with replacement from X, the
    algorithm is refitted under its principal hyperparameters, and the
    silhouette coefficient is computed on the bootstrap sample.
    """
    n = len(X)
    sils = []
    for b in range(n_boot):
        seed_b = seed + b * 7
        rng_b = np.random.default_rng(seed_b)
        boot_idx = rng_b.choice(n, n, replace=True)
        X_b = X[boot_idx]
        try:
            labels = fit_predict(algorithm, X_b, k, seed=seed_b)
            if len(np.unique(labels)) < 2:
                continue
            sils.append(
                silhouette_score(
                    X_b, labels,
                    sample_size=min(2000, len(X_b)),
                    random_state=seed_b,
                )
            )
        except Exception:
            continue
    sils = np.array(sils)
    if len(sils) == 0:
        return {
            "mean": np.nan, "lo": np.nan, "hi": np.nan,
            "n_boot_effective": 0, "all_values": [],
        }
    return {
        "mean":             float(sils.mean()),
        "lo":               float(np.percentile(sils, 2.5)),
        "hi":               float(np.percentile(sils, 97.5)),
        "n_boot_effective": int(len(sils)),
        "all_values":       sils.tolist(),
    }
