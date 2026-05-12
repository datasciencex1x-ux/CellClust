#!/usr/bin/env python3
"""
run_analysis.py
================
End-to-end analytical pipeline. Single-command driver that reproduces
every numerical result and figure reported in the manuscript.

Usage
-----
    python run_analysis.py [--input data/VF_DATA.csv]
                           [--out results/]
                           [--boot-iters 100]
                           [--gap-refs 100]

All outputs are written to ``results/`` (subfolders ``tables/`` and
``figures/``).
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Make src/ importable when running without `pip install -e .`
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from cellclust import config
from cellclust.data import load_data, split_by_muscle, summarize_strata
from cellclust.clustering import fit_predict, reorder_clusters_by_mean
from cellclust.k_selection import (
    compute_inertia_curve, compute_bic_aic_curves,
    elbow_kneedle, gap_statistic,
    algorithm_silhouette_vote, consensus_k,
)
from cellclust.validation import (
    internal_indices, composite_score,
    ari_matrix, bootstrap_silhouette_ci,
)
from cellclust.inferential import (
    per_animal_proportions, animal_stratified_cv,
    diet_effect_summary, chi2_stratified_by_diet,
)
from cellclust.plots import (
    save_figure,
    plot_dataset_overview, plot_k_selection,
    plot_voting_matrix, plot_composite_benchmark,
    plot_ari_matrices, plot_cluster_signatures,
    plot_diet_x_week,
)


def banner(text):
    bar = "=" * 70
    print(f"\n{bar}\n  {text}\n{bar}")


def main(input_path, out_dir, boot_iters, gap_refs):
    # Override output paths if user supplied --out
    if out_dir is not None:
        config.RESULTS_DIR = Path(out_dir)
        config.TABLES_DIR  = config.RESULTS_DIR / "tables"
        config.FIGURES_DIR = config.RESULTS_DIR / "figures"
        for d in (config.RESULTS_DIR, config.TABLES_DIR, config.FIGURES_DIR):
            d.mkdir(parents=True, exist_ok=True)

    banner("CellSize-Clust analytical pipeline")
    print(f"Input          : {input_path}")
    print(f"Output         : {config.RESULTS_DIR}")
    print(f"Bootstrap iter : {boot_iters}")
    print(f"Gap refs       : {gap_refs}")
    print(f"Random seed    : {config.RANDOM_STATE}")

    # ---- Step 1 : load + QC
    banner("Step 1  Load and QC the dataset")
    df = load_data(input_path)
    data_by_muscle = split_by_muscle(df)
    strata = summarize_strata(df)
    for m in config.MUSCLES:
        s = data_by_muscle[m]
        print(f"  {m:>3s}: n={len(s):>6,}  animals={s['animal'].nunique():>3d}  "
              f"mu={s['feret'].mean():.2f}  sd={s['feret'].std():.2f}")
    print(f"  TOTAL: n={len(df):,} fibers, {df['animal'].nunique()} animals")
    strata.to_csv(config.TABLES_DIR / "T01_strata_summary.csv", index=False)

    # ---- Step 2 : 8 algos x 2 muscles x k=2..8
    banner("Step 2  Fit 8 algorithms across k = 2..8")
    bench_records = []
    labels_store  = {}
    inertia, bic_curves, aic_curves = {}, {}, {}
    for m in config.MUSCLES:
        feret = data_by_muscle[m]["feret"].values.reshape(-1, 1)
        Xz = StandardScaler().fit_transform(feret)
        inertia[m] = compute_inertia_curve(Xz, config.K_RANGE)
        bic_curves[m], aic_curves[m] = compute_bic_aic_curves(Xz, config.K_RANGE)
        print(f"\n  Fitting {m}...")
        for alg in config.ALGORITHMS:
            for k in config.K_RANGE:
                labels = fit_predict(alg, Xz, k=k).astype(np.int8)
                labels_store[(m, alg, k)] = labels
                idx = internal_indices(Xz, labels)
                bench_records.append({"Muscle": m, "Algorithm": alg, "k": k, **idx})
            print(f"    {alg:<16s} done")

    bench_df = pd.DataFrame(bench_records)
    bench_df = composite_score(bench_df)
    bench_df.to_csv(config.TABLES_DIR / "T02_benchmark_full.csv", index=False)

    # ---- Step 3 : Formal K-selection
    banner("Step 3  Formal K-selection (4 criteria + algorithm voting)")
    K_criteria = {}
    for m in config.MUSCLES:
        feret = data_by_muscle[m]["feret"].values.reshape(-1, 1)
        Xz = StandardScaler().fit_transform(feret)
        e = elbow_kneedle(config.K_RANGE, inertia[m])
        b = config.K_RANGE[int(np.argmin(bic_curves[m]))]
        a = config.K_RANGE[int(np.argmin(aic_curves[m]))]
        g = gap_statistic(Xz, config.K_RANGE, n_refs=gap_refs,
                           seed=config.RANDOM_STATE)
        algo_votes = []
        for alg in config.ALGORITHMS:
            lab_per_k = {k: labels_store[(m, alg, k)] for k in config.K_RANGE}
            algo_votes.append(algorithm_silhouette_vote(lab_per_k, Xz))
        K_criteria[m] = {
            "K":           config.K_RANGE,
            "inertia":     inertia[m],
            "bic":         bic_curves[m],
            "aic":         aic_curves[m],
            "gap":         g["gap"],
            "gap_se":      g["se"],
            "elbow_k_opt": e,
            "bic_k_opt":   int(b),
            "aic_k_opt":   int(a),
            "gap_k_opt":   g["k_opt"],
            "algo_votes":  algo_votes,
        }
        k_cons, summary = consensus_k(
            formal_votes=[e, b, a, g["k_opt"]],
            algorithm_votes=algo_votes,
        )
        K_criteria[m]["consensus_K"]  = k_cons
        K_criteria[m]["vote_summary"] = summary
        print(f"  {m}: elbow={e}  BIC={b}  AIC={a}  gap={g['k_opt']}  "
              f"algos={algo_votes}  -> consensus K = {k_cons}")
    with open(config.TABLES_DIR / "T03_K_selection.json", "w") as f:
        json.dump(K_criteria, f, indent=2, default=int)

    # ---- Step 4 : Bootstrap 95% CI on silhouette
    banner("Step 4  Bootstrap 95% CI for the silhouette coefficient")
    top4 = ["Agglom-Ward", "GMM", "Spectral", "K-Means"]
    boot_records = []
    for m in config.MUSCLES:
        Xz = StandardScaler().fit_transform(
            data_by_muscle[m]["feret"].values.reshape(-1, 1)
        )
        print(f"\n  {m}:")
        for alg in top4:
            ci = bootstrap_silhouette_ci(
                alg, Xz, k=config.K_MAIN,
                n_boot=boot_iters, seed=config.RANDOM_STATE,
            )
            boot_records.append({"Muscle": m, "Algorithm": alg, **ci})
            print(f"    {alg:<14s}  Sil = {ci['mean']:.3f}  "
                  f"[{ci['lo']:.3f}, {ci['hi']:.3f}]  n_boot={ci['n_boot_effective']}")
    pd.DataFrame(boot_records).drop(columns=["all_values"]).to_csv(
        config.TABLES_DIR / "T04_bootstrap_CI.csv", index=False
    )

    # ---- Step 5 : ARI matrices at K_MAIN
    banner(f"Step 5  ARI matrices at K = {config.K_MAIN}")
    ari_matrices = {}
    for m in config.MUSCLES:
        labs = {alg: labels_store[(m, alg, config.K_MAIN)]
                 for alg in config.ALGORITHMS}
        mat = ari_matrix(labs)
        ari_matrices[m] = mat
        mat.to_csv(config.TABLES_DIR / f"T05_ARI_{m}_k{config.K_MAIN}.csv")
        print(f"  {m}: ARI(Agglom-Ward vs GMM) = "
              f"{mat.loc['Agglom-Ward','GMM']:.3f}")

    # ---- Step 6 : Animal-level inference
    banner("Step 6  Animal-level inferential framework")
    principal = config.PRINCIPAL_ALGORITHM
    CLUSTER_NAMES = ["Low", "High"]
    assignment_frames = []
    for m in config.MUSCLES:
        sub = data_by_muscle[m].copy()
        feret = sub["feret"].values
        labels = labels_store[(m, principal, config.K_MAIN)]
        labels = reorder_clusters_by_mean(labels, feret)
        sub["cluster_idx"] = labels
        sub["cluster"] = pd.Categorical(
            [CLUSTER_NAMES[i] for i in labels],
            categories=CLUSTER_NAMES,
        )
        assignment_frames.append(sub)
    assignments = pd.concat(assignment_frames, ignore_index=True)
    assignments.to_csv(
        config.TABLES_DIR /
        f"T06_assignments_{principal.replace('+','_')}_k{config.K_MAIN}.csv",
        index=False,
    )

    per_animal = per_animal_proportions(assignments, cluster_names=CLUSTER_NAMES)
    per_animal.to_csv(config.TABLES_DIR / "T07_per_animal_proportions.csv",
                       index=False)

    cv_rows = []
    for m in config.MUSCLES:
        cv = animal_stratified_cv(
            data_by_muscle[m],
            algorithm=principal,
            k=config.K_MAIN,
            n_folds=5,
            seed=config.RANDOM_STATE,
        )
        cv_rows.append({
            "Muscle":     m,
            "n_folds":    cv["n_folds_effective"],
            "Sil_mean":   cv["Silhouette_mean"],
            "Sil_sd":     cv["Silhouette_sd"],
            "Sil_CI_lo":  cv["Silhouette_CI_lo"],
            "Sil_CI_hi":  cv["Silhouette_CI_hi"],
            "DB_mean":    cv["DB_mean"],
            "DB_sd":      cv["DB_sd"],
        })
        print(f"  {m} CV: Sil = {cv['Silhouette_mean']:.3f} "
              f"+/- {cv['Silhouette_sd']:.3f}")
    pd.DataFrame(cv_rows).to_csv(config.TABLES_DIR / "T08_CV_5fold.csv",
                                  index=False)

    diet_eff = diet_effect_summary(
        per_animal,
        muscles=config.MUSCLES,
        cluster_names=CLUSTER_NAMES,
        n_boot=1000,
        seed=config.RANDOM_STATE,
    )
    diet_eff.to_csv(config.TABLES_DIR / "T09_diet_effect.csv", index=False)
    print(f"\n  Diet effect (rank-biserial, animal-level):")
    for _, r in diet_eff.iterrows():
        sig = "*" if not (r["CI_lo"] <= 0 <= r["CI_hi"]) else ""
        print(f"    {r['Muscle']} {r['Cluster']:<5s}  r_rb = {r['r_rb']:+.3f}  "
              f"95% CI [{r['CI_lo']:+.3f}, {r['CI_hi']:+.3f}] {sig}")

    chi = chi2_stratified_by_diet(assignments, cluster_names=CLUSTER_NAMES)
    chi.to_csv(config.TABLES_DIR / "T10_chi2_stratified.csv", index=False)
    print(f"\n  Chi-square SOL vs GAS stratified by diet:")
    for _, r in chi.iterrows():
        print(f"    {r['Condition']}: chi2 = {r['chi2']:.2f}, "
              f"p = {r['p']:.2e}, V = {r['Cramers_V']:.3f}  "
              f"({r['Interpretation']})")

    # ---- Step 7 : Figures
    banner("Step 7  Generating figures (600 dpi RGB)")

    fig = plot_dataset_overview(data_by_muscle)
    save_figure(fig, str(config.FIGURES_DIR / "F1_dataset_overview.png"))
    print("  F1_dataset_overview.png")

    fig = plot_k_selection(K_criteria)
    save_figure(fig, str(config.FIGURES_DIR / "F2_k_selection.png"))
    print("  F2_k_selection.png")

    fig = plot_voting_matrix(bench_df)
    save_figure(fig, str(config.FIGURES_DIR / "F3_voting_matrix.png"))
    print("  F3_voting_matrix.png")

    bench_by_muscle = {
        m: bench_df[(bench_df["Muscle"] == m) & (bench_df["k"] == config.K_MAIN)]
            .assign(Rank=lambda d: d["Composite"].rank(ascending=False))
        for m in config.MUSCLES
    }
    fig = plot_composite_benchmark(bench_by_muscle)
    save_figure(fig, str(config.FIGURES_DIR / "F4_benchmark_composite.png"))
    print("  F4_benchmark_composite.png")

    fig = plot_ari_matrices(ari_matrices["SOL"], ari_matrices["GAS"])
    save_figure(fig, str(config.FIGURES_DIR / "F5_ARI_matrices.png"))
    print("  F5_ARI_matrices.png")

    labels_principal = {
        m: reorder_clusters_by_mean(
            labels_store[(m, principal, config.K_MAIN)],
            data_by_muscle[m]["feret"].values,
        )
        for m in config.MUSCLES
    }
    fig = plot_cluster_signatures(data_by_muscle, labels_principal,
                                    cluster_names=("Low", "High"))
    save_figure(fig, str(config.FIGURES_DIR / "F6_cluster_signatures.png"))
    print("  F6_cluster_signatures.png")

    fig = plot_diet_x_week(per_animal, diet_eff)
    save_figure(fig, str(config.FIGURES_DIR / "F7_diet_x_week.png"))
    print("  F7_diet_x_week.png")

    # ---- Done
    banner("Pipeline complete")
    print(f"All outputs written to {config.RESULTS_DIR}")
    print(f"  - {len(list(config.TABLES_DIR.glob('*.csv')))} CSV tables")
    print(f"  - {len(list(config.FIGURES_DIR.glob('*.png')))} PNG figures")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the CellSize-Clust analytical pipeline."
    )
    parser.add_argument("--input", type=Path, default=None,
                         help="Path to VF_DATA.csv")
    parser.add_argument("--out", type=Path, default=None,
                         help="Output directory (default: results/)")
    parser.add_argument("--boot-iters", type=int, default=100,
                         help="Bootstrap iterations for silhouette CI")
    parser.add_argument("--gap-refs", type=int, default=100,
                         help="Monte Carlo refs for gap statistic")
    args = parser.parse_args()

    # Load config to get default input path
    sys.path.insert(0, str(HERE.parent / "src"))
    from cellclust import config as _cfg
    input_path = args.input if args.input is not None else _cfg.RAW_CSV

    main(input_path, args.out, args.boot_iters, args.gap_refs)
