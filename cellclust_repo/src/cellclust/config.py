"""
cellclust.config
================
Project-wide constants, file paths, palette, and the canonical random
seed. Imported by every other submodule.
"""
from pathlib import Path

# ---------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------
# Paths (relative to the repository root)
# ---------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[2]
DATA_DIR    = ROOT / "data"
RESULTS_DIR = ROOT / "results"
TABLES_DIR  = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

RAW_CSV     = DATA_DIR / "VF_DATA.csv"

for d in (RESULTS_DIR, TABLES_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------
# Analytical decisions (manuscript anchors)
# ---------------------------------------------------------------
MUSCLES = ["SOL", "GAS"]
DIETS   = ["NCD", "HFD"]
WEEKS   = [8, 12, 16]

K_RANGE         = list(range(2, 9))
K_MAIN          = 2          # consensus K
K_EXPLORATORY   = 4          # elbow-suggested K (secondary analysis)
PRINCIPAL_ALGORITHM = "Agglom-Ward"

FERET_MIN_UM = 2.0
FERET_MAX_UM = 120.0
TREATNMD_THRESHOLD_UM = 20.0  # Treat-NMD atrophic-fiber threshold

# ---------------------------------------------------------------
# 8-algorithm panel (4 paradigms)
# ---------------------------------------------------------------
ALGORITHMS = [
    "K-Means",         # centroid-based
    "GMM",             # probabilistic
    "Spectral",        # graph-based
    "Agglom-Ward",     # hierarchical (Ward linkage)
    "BIRCH",           # tree-based
    "Agglom-Average",  # hierarchical (average linkage)
    "SOM+Ward",        # self-organizing map + hierarchical
    "BayesianGMM",     # nonparametric probabilistic
]

# ---------------------------------------------------------------
# Plotting palette
# ---------------------------------------------------------------
COLORS = {
    "ink":    "#1A1918",
    "muted":  "#7B7268",
    "sol":    "#C66B4A",
    "gas":    "#5B7A8F",
    "low":    "#7A9B7E",
    "high":   "#C66B4A",
    "ncd":    "#7A9B7E",
    "hfd":    "#C66B4A",
}

ALG_COLORS = {
    "K-Means":         "#C66B4A",
    "GMM":             "#7A9B7E",
    "Spectral":        "#C9A961",
    "Agglom-Ward":     "#5B7A8F",
    "BIRCH":           "#E08A6B",
    "Agglom-Average":  "#8E5B6B",
    "SOM+Ward":        "#3F6B5C",
    "BayesianGMM":     "#324858",
}

DPI_FINAL = 600
