"""
cellclust.k_selection
=====================
Formal selection of the number of clusters via four independent
computational criteria plus per-algorithm silhouette voting:

  1. Elbow on K-Means inertia (Kneedle method, Satopaa et al. 2011)
  2. Bayesian Information Criterion (BIC) on GMM
  3. Akaike Information Criterion (AIC) on GMM
  4. Gap statistic (Tibshirani 2001) with 1-standard-error rule
  5. Modal vote across the 8 algorithms (argmax silhouette per algorithm)

The consensus K is the modal vote across these 4 + 8 = 12 sources.
"""
from __future__ import annotations
from collections import Counter
import numpy as np

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

from .config import RANDOM_STATE


# ---------------------------------------------------------------
# Criterion 1: Elbow on K-Means inertia
# ---------------------------------------------------------------
def compute_inertia_curve(X: np.ndarray, k_range: list[int]) -> list[float]:
    """Within-cluster sum of squared distances (K-Means inertia)."""
    return [
        KMeans(n_clusters=k, n_init=30, random_state=RANDOM_STATE).fit(X).inertia_
        for k in k_range
    ]


def elbow_kneedle(k_values: list[int], y_values: list[float]) -> int:
    """Kneedle elbow: max perpendicular distance to the chord between
    the first and last points of the curve."""
    K = np.array(k_values, dtype=float)
    Y = np.array(y_values, dtype=float)
    p1 = np.array([K[0],  Y[0]])
    p2 = np.array([K[-1], Y[-1]])
    chord = p2 - p1
    chord_norm = chord / np.linalg.norm(chord)
    distances = []
    for i in range(len(K)):
        v = np.array([K[i], Y[i]]) - p1
        proj = np.dot(v, chord_norm) * chord_norm
        distances.append(np.linalg.norm(v - proj))
    return int(K[int(np.argmax(distances))])


# ---------------------------------------------------------------
# Criteria 2 & 3: BIC and AIC on GMM
# ---------------------------------------------------------------
def compute_bic_aic_curves(X: np.ndarray,
                              k_range: list[int]) -> tuple[list[float], list[float]]:
    """Full-covariance GMM with 10 restarts; return (BIC, AIC)."""
    bic_values, aic_values = [], []
    for k in k_range:
        gmm = GaussianMixture(
            n_components=k, covariance_type="full",
            n_init=10, random_state=RANDOM_STATE,
        ).fit(X)
        bic_values.append(gmm.bic(X))
        aic_values.append(gmm.aic(X))
    return bic_values, aic_values


# ---------------------------------------------------------------
# Criterion 4: Gap statistic (Tibshirani 1-SE rule)
# ---------------------------------------------------------------
def gap_statistic(X: np.ndarray, k_range: list[int],
                    n_refs: int = 100, seed: int = RANDOM_STATE) -> dict:
    """Gap statistic with Tibshirani's 1-standard-error rule.

    Reference distribution: uniform Monte Carlo over the bounding box
    of X. ``n_refs`` controls the number of Monte Carlo replicates.
    """
    rng = np.random.default_rng(seed)
    n, p = X.shape
    mins, maxs = X.min(axis=0), X.max(axis=0)

    Wk = np.zeros(len(k_range))
    Wk_ref = np.zeros((n_refs, len(k_range)))

    for i, k in enumerate(k_range):
        Wk[i] = np.log(
            KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X).inertia_ + 1e-10
        )

    for b in range(n_refs):
        Xb = rng.uniform(mins, maxs, size=(n, p))
        for i, k in enumerate(k_range):
            Wk_ref[b, i] = np.log(
                KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xb).inertia_ + 1e-10
            )

    Eref = Wk_ref.mean(axis=0)
    SDk  = Wk_ref.std(axis=0)
    sk   = SDk * np.sqrt(1.0 + 1.0 / n_refs)
    gap  = Eref - Wk

    k_opt = k_range[-1]
    for i in range(len(k_range) - 1):
        if gap[i] >= gap[i + 1] - sk[i + 1]:
            k_opt = k_range[i]
            break

    return {
        "k_values": list(k_range),
        "gap":      gap.tolist(),
        "se":       sk.tolist(),
        "k_opt":    int(k_opt),
    }


# ---------------------------------------------------------------
# Per-algorithm silhouette voting
# ---------------------------------------------------------------
def algorithm_silhouette_vote(labels_per_k: dict[int, np.ndarray],
                                X: np.ndarray) -> int:
    """Return the k that maximizes silhouette over the provided labels."""
    best_k, best_sil = None, -np.inf
    for k, labels in labels_per_k.items():
        if len(np.unique(labels)) < 2:
            continue
        sil = silhouette_score(
            X, labels,
            sample_size=min(5000, len(X)),
            random_state=RANDOM_STATE,
        )
        if sil > best_sil:
            best_sil, best_k = sil, k
    return int(best_k) if best_k is not None else int(min(labels_per_k.keys()))


# ---------------------------------------------------------------
# Consensus K
# ---------------------------------------------------------------
def consensus_k(formal_votes: list[int],
                  algorithm_votes: list[int]) -> tuple[int, dict]:
    """Combine formal and algorithm votes; return the modal K."""
    all_votes = list(formal_votes) + list(algorithm_votes)
    counts = Counter(all_votes)
    k_consensus = counts.most_common(1)[0][0]
    return int(k_consensus), {
        "formal_votes":    list(formal_votes),
        "algorithm_votes": list(algorithm_votes),
        "counts":          dict(counts),
        "k_consensus":     int(k_consensus),
    }
