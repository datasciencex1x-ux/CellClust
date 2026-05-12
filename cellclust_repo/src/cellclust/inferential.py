"""
cellclust.inferential
=====================
Animal-level inferential statistics.

All inferential tests are performed on animal-aggregated cluster
proportions, NOT on fibers, to avoid pseudoreplication. Fibers within
the same animal are not independent observations.

Tests reported in the manuscript:
  - Animal-stratified 5-fold cross-validation (silhouette stability)
  - Rank-biserial correlation r_rb with bootstrap 95% CI (diet effect)
  - Welch t-test, Mann-Whitney U as complementary statistics
  - Chi-square on the fiber-level (Muscle x Cluster) contingency table,
    stratified by diet to avoid diet x muscle confounding
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

from .clustering import fit_predict
from .config import RANDOM_STATE


def per_animal_proportions(assignments: pd.DataFrame,
                              cluster_names: list[str] | None = None) -> pd.DataFrame:
    """Compute per-animal cluster proportions.

    Parameters
    ----------
    assignments : pd.DataFrame
        Must contain columns ``["animal", "Musculo", "diet", "week",
        "cluster", "feret"]``.
    cluster_names : list[str] or None
        Cluster category order; inferred from the data if None.

    Returns
    -------
    pd.DataFrame
        One row per (animal x muscle) with columns:
        animal, Musculo, diet, week, <cluster_name>..., total,
        pct_<cluster_name>..., mean_feret.
    """
    if cluster_names is None:
        if hasattr(assignments["cluster"], "cat"):
            cluster_names = list(assignments["cluster"].cat.categories)
        else:
            cluster_names = list(assignments["cluster"].unique())

    pivot = (
        assignments
        .groupby(["animal", "Musculo", "diet", "week", "cluster"], observed=True)
        .size()
        .unstack("cluster", fill_value=0)
    )
    pivot["total"] = pivot.sum(axis=1)
    for cl in cluster_names:
        pivot[f"pct_{cl}"] = 100 * pivot[cl] / pivot["total"]

    per_animal = pivot.reset_index().rename_axis(None, axis=1)
    mean_feret = (
        assignments.groupby("animal")["feret"].mean().rename("mean_feret")
    )
    return per_animal.merge(mean_feret, on="animal")


def animal_stratified_cv(df_muscle: pd.DataFrame, algorithm: str,
                            k: int = 2, n_folds: int = 5,
                            seed: int = RANDOM_STATE) -> dict:
    """Animal-stratified k-fold cross-validation of cluster stability.

    Animals are randomly partitioned into ``n_folds`` folds. In each
    fold, the algorithm is fitted on the training animals (z-scored
    independently) and Silhouette + Davies-Bouldin are reported.
    """
    from sklearn.model_selection import KFold

    rng = np.random.default_rng(seed)
    animals = df_muscle["animal"].unique()
    animals_shuf = rng.permutation(animals)
    kf = KFold(n_splits=n_folds, shuffle=False)

    silhouettes, db_values = [], []
    for _, (train_idx, _) in enumerate(kf.split(animals_shuf)):
        train_animals = animals_shuf[train_idx]
        train_mask = df_muscle["animal"].isin(train_animals)
        Xz_tr = StandardScaler().fit_transform(
            df_muscle.loc[train_mask, "feret"].values.reshape(-1, 1)
        )
        try:
            labels = fit_predict(algorithm, Xz_tr, k=k, seed=seed)
            if len(np.unique(labels)) < 2:
                continue
            sil = silhouette_score(
                Xz_tr, labels,
                sample_size=min(3000, len(Xz_tr)),
                random_state=seed,
            )
            db = davies_bouldin_score(Xz_tr, labels)
            silhouettes.append(sil)
            db_values.append(db)
        except Exception:
            continue

    sils = np.array(silhouettes)
    dbs  = np.array(db_values)
    return {
        "n_folds_effective": len(sils),
        "Silhouette_mean":   float(sils.mean()) if len(sils) else np.nan,
        "Silhouette_sd":     float(sils.std())  if len(sils) else np.nan,
        "Silhouette_CI_lo":  float(np.percentile(sils, 2.5))  if len(sils) >= 5 else np.nan,
        "Silhouette_CI_hi":  float(np.percentile(sils, 97.5)) if len(sils) >= 5 else np.nan,
        "DB_mean":           float(dbs.mean()) if len(dbs) else np.nan,
        "DB_sd":             float(dbs.std())  if len(dbs) else np.nan,
        "fold_silhouettes":  sils.tolist(),
        "fold_dbs":          dbs.tolist(),
    }


def rank_biserial(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Rank-biserial correlation r_rb (effect size for Mann-Whitney U).

    Convention: positive r_rb means ``group_b > group_a``.
    Called as ``rank_biserial(ncd, hfd)``, positive => HFD > NCD.
    """
    if len(group_a) == 0 or len(group_b) == 0:
        return np.nan
    u, _ = mannwhitneyu(group_b, group_a, alternative="two-sided")
    r = 1.0 - 2.0 * u / (len(group_a) * len(group_b))
    return -float(r)


def rank_biserial_with_ci(group_a: np.ndarray, group_b: np.ndarray,
                            n_boot: int = 1000,
                            seed: int = RANDOM_STATE) -> dict:
    """Rank-biserial with bootstrap 95% confidence interval."""
    rng = np.random.default_rng(seed)
    point = rank_biserial(group_a, group_b)
    boot_values = []
    for _ in range(n_boot):
        a_b = rng.choice(group_a, size=len(group_a), replace=True)
        b_b = rng.choice(group_b, size=len(group_b), replace=True)
        try:
            r_b = rank_biserial(a_b, b_b)
            if not np.isnan(r_b):
                boot_values.append(r_b)
        except Exception:
            pass
    boot_values = np.array(boot_values)
    if len(boot_values) < 10:
        return {"r_rb": point, "CI_lo": np.nan, "CI_hi": np.nan,
                "n_boot_effective": len(boot_values)}
    return {
        "r_rb":             point,
        "CI_lo":            float(np.percentile(boot_values, 2.5)),
        "CI_hi":            float(np.percentile(boot_values, 97.5)),
        "n_boot_effective": int(len(boot_values)),
    }


def diet_effect_summary(per_animal: pd.DataFrame, muscles: list[str],
                          cluster_names: list[str], n_boot: int = 1000,
                          seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Per (Muscle x Cluster) NCD vs HFD effect-size summary at the animal level."""
    rows = []
    for m in muscles:
        sub = per_animal[per_animal["Musculo"] == m]
        for cl in cluster_names:
            col = f"pct_{cl}"
            ncd = sub.loc[sub["diet"] == "NCD", col].values
            hfd = sub.loc[sub["diet"] == "HFD", col].values
            if len(ncd) < 2 or len(hfd) < 2:
                continue
            rb = rank_biserial_with_ci(ncd, hfd, n_boot=n_boot, seed=seed)
            t,  p_t = ttest_ind(ncd, hfd, equal_var=False)
            u,  p_u = mannwhitneyu(ncd, hfd, alternative="two-sided")
            pooled_sd = np.sqrt((ncd.std() ** 2 + hfd.std() ** 2) / 2.0)
            cohen_d = (ncd.mean() - hfd.mean()) / (pooled_sd + 1e-12)
            rows.append({
                "Muscle":   m, "Cluster": cl,
                "NCD_mean": ncd.mean(), "NCD_sd": ncd.std(), "NCD_n": len(ncd),
                "HFD_mean": hfd.mean(), "HFD_sd": hfd.std(), "HFD_n": len(hfd),
                "r_rb":     rb["r_rb"],
                "CI_lo":    rb["CI_lo"],
                "CI_hi":    rb["CI_hi"],
                "Welch_t":  t,  "Welch_p":  p_t,
                "MannW_U":  u,  "MannW_p":  p_u,
                "Cohen_d":  cohen_d,
            })
    return pd.DataFrame(rows)


def chi2_stratified_by_diet(assignments: pd.DataFrame,
                              cluster_names: list[str]) -> pd.DataFrame:
    """Chi-square test on the (Muscle x Cluster) contingency table,
    stratified by diet (NCD only and HFD only).

    Reports chi-square, df, p-value, Cramer's V, and a verbal effect-size
    interpretation.
    """
    rows = []
    for d in assignments["diet"].unique():
        sub = assignments[assignments["diet"] == d]
        ct = (
            sub.groupby(["Musculo", "cluster"], observed=True)
            .size()
            .unstack("cluster")
            .reindex(columns=cluster_names)
            .fillna(0)
        )
        chi2, p, dof, _ = chi2_contingency(ct.values)
        v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
        if p < 0.05:
            if   v > 0.30: interp = "Significant; large"
            elif v > 0.15: interp = "Significant; small-moderate"
            elif v > 0.10: interp = "Significant; small"
            else:          interp = "Significant; trivial"
        else:
            interp = "Not significant"
        rows.append({
            "Condition":      d,
            "chi2":           float(chi2),
            "df":             int(dof),
            "p":              float(p),
            "Cramers_V":      float(v),
            "Interpretation": interp,
        })
    return pd.DataFrame(rows)
