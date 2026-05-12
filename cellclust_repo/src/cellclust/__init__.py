"""
cellclust — unsupervised multi-algorithm clustering of skeletal
muscle fibers by minimum Feret diameter.

Public submodules
-----------------
config        : project constants, paths, palette, fixed seed
data          : raw CSV loading and biological-plausibility QC
clustering    : 8 clustering algorithms (unified fit_predict API)
k_selection   : 4 formal K-selection criteria + algorithm voting
validation    : internal validation indices, composite score, ARI, bootstrap
inferential   : animal-level inference (CV, rank-biserial, chi-square)
plots         : matplotlib helpers for the seven main-text figures

Quick start
-----------
    from cellclust import config
    from cellclust.data import load_data, split_by_muscle
    from cellclust.clustering import fit_predict

    df = load_data()
    Xz = (df["feret"].values.reshape(-1, 1) - mean) / sd
    labels = fit_predict("Agglom-Ward", Xz, k=2)
"""
__version__ = "1.0.0"
__author__  = "Llanos P., Russell-Guzman J., Monsalves-Alvarez M. et al."
__license__ = "MIT (code) + CC-BY-4.0 (data)"
