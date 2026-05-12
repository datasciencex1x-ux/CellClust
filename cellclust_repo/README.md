# CellSize-Clust

[![pipeline](https://gitlab.com/rigo1983/cellclust/badges/main/pipeline.svg)](https://gitlab.com/rigo1983/cellclust/-/pipelines)
[![coverage](https://gitlab.com/rigo1983/cellclust/badges/main/coverage.svg)](https://gitlab.com/rigo1983/cellclust/-/pipelines)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![data: CC-BY-4.0](https://img.shields.io/badge/data-CC--BY--4.0-orange.svg)](LICENSE-DATA)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

**Unsupervised multi-algorithm classification of skeletal muscle fibers
based on the minimum Feret diameter.**

CellSize-Clust is a reproducible, label-free Python pipeline that
classifies skeletal muscle fibers into morphometric subpopulations
using only the minimum Feret diameter. The pipeline benchmarks eight
clustering algorithms across four algorithmic paradigms, determines
the optimal number of clusters via four formal computational criteria,
and performs animal-level inference to avoid pseudoreplication.

---

## Highlights

- **Eight algorithms** across four paradigms (centroid, probabilistic,
  graph-based, hierarchical, self-organizing).
- **Four formal K-selection criteria** (elbow, BIC, AIC, gap statistic)
  combined with per-algorithm silhouette voting.
- **Animal-stratified 5-fold cross-validation** of cluster stability.
- **Animal-level inference** (rank-biserial `r_rb` with bootstrap 95% CI,
  Welch t-test, Mann-Whitney U, Cohen's d) — no pseudoreplication.
- **Bootstrap 95% CI** on the silhouette coefficient (B = 100).
- **Fully reproducible** with `RANDOM_STATE = 42` throughout.
- **Open data** (CC-BY-4.0): 14,655 fibers from 42 mice.

---

## Repository layout

```
cellclust/
├── data/
│   └── VF_DATA.csv                   # 14,655 fibers (raw export)
├── src/
│   └── cellclust/                    # importable package
│       ├── __init__.py
│       ├── config.py                 # constants, paths, palette, seed
│       ├── data.py                   # CSV loading + QC
│       ├── clustering.py             # 8 algorithms, unified fit_predict API
│       ├── k_selection.py            # 4 formal criteria + voting
│       ├── validation.py             # internal indices, composite, ARI, bootstrap
│       ├── inferential.py            # animal-level inference (CV, r_rb, chi2)
│       └── plots.py                  # 7 main-text figures
├── notebooks/
│   └── 01_walkthrough.ipynb          # interactive demo
├── results/
│   ├── tables/                       # auto-generated CSVs
│   └── figures/                      # auto-generated PNGs (600 dpi)
├── tests/
│   ├── test_data.py
│   ├── test_clustering.py
│   └── test_inferential.py
├── docs/
│   └── HYPERPARAMETERS.md            # full per-algorithm hyperparameter audit
├── run_analysis.py                   # CLI driver
├── requirements.txt                  # pip dependencies
├── environment.yml                   # conda dependencies
├── pyproject.toml                    # build/install config
├── .gitlab-ci.yml                    # GitLab CI/CD pipeline
├── .gitignore
├── LICENSE                           # MIT (code)
├── LICENSE-DATA                      # CC-BY-4.0 (data)
├── CITATION.cff                      # citation metadata
└── README.md                         # this file
```

---

## Quick start

### 1. Clone from GitLab

```bash
git clone https://gitlab.com/rigo1983/cellclust.git
cd cellclust
```

### 2. Install dependencies

With pip:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate cellclust
```

Or install the package itself in editable mode:

```bash
pip install -e .
```

### 3. Run the full pipeline

```bash
python run_analysis.py
```

This single command reproduces every numerical result and figure in
the manuscript. Outputs are written to `results/tables/` (10 CSV
files) and `results/figures/` (7 PNG figures at 600 dpi).

Expected runtime on a laptop: 5–10 minutes.

### 4. CLI options

```bash
python run_analysis.py \
    --input data/VF_DATA.csv \
    --out results/ \
    --boot-iters 100 \
    --gap-refs 100
```

---

## Usage from Python

```python
from cellclust.data import load_data, split_by_muscle
from cellclust.clustering import fit_predict, reorder_clusters_by_mean
from sklearn.preprocessing import StandardScaler

# 1. Load & QC
df = load_data()                                  # default: data/VF_DATA.csv
data_by_muscle = split_by_muscle(df)

# 2. Z-score per muscle (independent normalization)
sol_feret = data_by_muscle["SOL"]["feret"].values.reshape(-1, 1)
Xz_sol = StandardScaler().fit_transform(sol_feret)

# 3. Cluster
labels = fit_predict("Agglom-Ward", Xz_sol, k=2)
labels = reorder_clusters_by_mean(labels, sol_feret.ravel())  # 0=Low, 1=High
```

For more advanced usage (cross-validation, bootstrap, animal-level
inference), see `notebooks/01_walkthrough.ipynb`.

---

## Dataset description

`data/VF_DATA.csv` contains **14,655 muscle fibers** from
**42 C57BL/6J male mice**, segmented from WGA-stained cryosections
with a U-Net model in Dragonfly v2022.2.

| Column                    | Description                                 |
|---------------------------|---------------------------------------------|
| `Min Feret Diameter (um)` | Minimum Feret diameter (float, decimal=`,`) |
| `Musculo`                 | `SOL` (soleus) or `GAS` (gastrocnemius)     |
| `Condicion`               | `NCD` or `HFD`                              |
| `Tiempo`                  | 8, 12, or 16 weeks                          |
| `Codigo animal`           | within-stratum integer animal identifier    |

CSV format: `;`-separated, decimal comma (European notation).

---

## Methodological decisions (manuscript anchors)

| Decision                     | Value                                       |
|------------------------------|---------------------------------------------|
| Random seed                  | `42`                                        |
| Muscles included             | SOL, GAS (FDB excluded)                     |
| Biological filter            | `2 um <= min Feret <= 120 um`               |
| Normalization                | z-score per muscle (independent)            |
| Principal K                  | `K = 2` (consensus of 4 criteria + 8 algos) |
| Exploratory K                | `K = 4` (elbow only, secondary analysis)    |
| Principal algorithm          | Agglomerative-Ward                          |
| Cross-validation             | Animal-stratified 5-fold                    |
| Bootstrap                    | B = 100 (silhouette); B = 1000 (r_rb)       |
| Gap statistic refs           | n_refs = 100 Monte Carlo replicates         |
| Inferential statistical unit | Animal (not fiber)                          |

For the per-algorithm hyperparameter audit, see [`docs/HYPERPARAMETERS.md`](docs/HYPERPARAMETERS.md).

---

## Results overview

Deterministic results from `python run_analysis.py` on the included
dataset (with `RANDOM_STATE = 42`):

| Muscle | Cluster | n      | Mean (um) | %     |
|--------|---------|--------|-----------|-------|
| SOL    | Low     | 2,989  | 30.25     | 68.8  |
| SOL    | High    | 1,356  | 45.74     | 31.2  |
| GAS    | Low     | 4,662  | 31.32     | 45.2  |
| GAS    | High    | 5,648  | 47.84     | 54.8  |

**Animal-stratified 5-fold CV:** SOL silhouette = 0.567 ± 0.013;
GAS silhouette = 0.548 ± 0.012.

**Chi-square SOL vs GAS, stratified by diet:**
- NCD: chi2 = 196.83, p = 1.0e-44, Cramer's V = 0.160
- HFD: chi2 = 535.55, p = 1.8e-118, Cramer's V = 0.277

---

## Reproducibility

All randomness is controlled by a single seed
(`config.RANDOM_STATE = 42`). Running `python run_analysis.py` on the
same machine should reproduce the exact same labels, indices, and
figures bit-for-bit, provided NumPy, scikit-learn, and SciPy versions
match those in `requirements.txt`.

Across different platforms, expect floating-point differences smaller
than 1e-4 in any reported metric.

---

## Testing

```bash
pytest tests/ -v
```

The test suite has 21 tests covering:

- Data loading produces the expected 14,655-fiber cohort
- Each of the 8 algorithms returns valid labels on synthetic data
- Composite score, ARI matrix, and bootstrap CI are well-defined
- The animal-level inferential pipeline runs without error

The GitLab CI pipeline runs all tests automatically on every push.

---

## GitLab CI/CD

This repository ships with a `.gitlab-ci.yml` that:

1. **`lint`** — runs `ruff` on the entire codebase
2. **`test`** — runs `pytest tests/ -v --cov=cellclust`
3. **`smoke`** — runs `run_analysis.py` with minimal iterations to
   confirm the pipeline executes end-to-end
4. **`pages`** (main branch only) — builds and deploys documentation
   to GitLab Pages

To enable: push to GitLab and the runner picks it up automatically.

---

## License

- **Code**: MIT (see [`LICENSE`](LICENSE))
- **Data**: CC-BY-4.0 (see [`LICENSE-DATA`](LICENSE-DATA))

---

## Citation

If you use this code or data, please cite:

```bibtex
@software{cellclust2026,
  author       = {Llanos, Paola and Russell-Guzm{\'a}n, Javier and
                    Monsalves-{\'A}lvarez, Mat{\'\i}as and others},
  title        = {CellSize-Clust: Unsupervised multi-algorithm
                    classification of skeletal muscle fibers},
  year         = {2026},
  publisher    = {GitLab},
  url          = {https://gitlab.com/rigo1983/cellclust},
  version      = {1.0.0}
}
```

See [`CITATION.cff`](CITATION.cff) for full bibliographic metadata.

---

## Funding

This work was supported by **FONDECYT 1231103** (Paola Llanos) and
**FONDECYT 11230186** (Matías Monsalves-Álvarez).

## Ethics

Animal protocol **CBA 240423 FOUCH** approved by the Animal Bioethics
Committee, Faculty of Dentistry, University of Chile.

## Contact

For questions or collaboration inquiries, please open an issue on the
[GitLab repository](https://gitlab.com/rigo1983/cellclust/-/issues).
