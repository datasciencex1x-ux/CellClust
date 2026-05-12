# Hyperparameter specification

This file consolidates the hyperparameter specification for the eight
algorithms benchmarked in CellSize-Clust. All hyperparameters are
implementation-ready (sklearn API where applicable). A single fixed
random seed (`RANDOM_STATE = 42`, set in `src/cellclust/config.py`)
governs every stochastic step for bitwise reproducibility.

## Algorithm panel (4 paradigms × 8 algorithms)

| Algorithm           | Paradigm                       | Library / Class                              |
|---------------------|--------------------------------|-----------------------------------------------|
| K-Means             | Centroid-based                 | `sklearn.cluster.KMeans`                      |
| GMM                 | Parametric mixture             | `sklearn.mixture.GaussianMixture`             |
| Spectral            | Graph-based                    | `sklearn.cluster.SpectralClustering`          |
| Agglom-Ward         | Hierarchical (Ward linkage)    | `sklearn.cluster.AgglomerativeClustering`     |
| BIRCH               | Tree-based incremental         | `sklearn.cluster.Birch`                       |
| Agglom-Average      | Hierarchical (avg linkage)     | `sklearn.cluster.AgglomerativeClustering`     |
| SOM+Ward            | Neural + hierarchical          | `minisom.MiniSom` + AgglomerativeClustering   |
| BayesianGMM         | Non-parametric Bayesian        | `sklearn.mixture.BayesianGaussianMixture`     |

## Detailed hyperparameters

### K-Means
- `n_init = 30`
- `init = "k-means++"`
- `max_iter = 300`
- `tol = 1e-6`
- `algorithm = "lloyd"`
- `random_state = 42`
- Fitted on full dataset.

### GMM (Gaussian Mixture)
- `n_components = k`
- `covariance_type = "full"`
- `tol = 1e-3`
- `reg_covar = 1e-6`
- `max_iter = 300`
- `n_init = 10`
- `init_params = "kmeans"`
- `random_state = 42`

### Spectral
- `n_clusters = k`
- `affinity = "rbf"`
- `gamma = 1.0`
- `assign_labels = "kmeans"`
- `n_init = 10`
- Sub-sampling: 4,000 fibers
- Label propagation: nearest-centroid in z-score space
- `random_state = 42`

### Agglom-Ward
- `n_clusters = k`
- `linkage = "ward"`
- `metric = "euclidean"` (required by Ward)
- Sub-sampling: 5,000 fibers
- Label propagation: nearest-centroid in z-score space
- `random_state = 42`

### BIRCH
- `n_clusters = k`
- `threshold = 0.05` (cluster-feature radius)
- `branching_factor = 50`
- Fitted on full dataset.

### Agglom-Average
- `n_clusters = k`
- `linkage = "average"`
- `metric = "euclidean"`
- Sub-sampling: 5,000 fibers
- Label propagation: nearest-centroid in z-score space
- `random_state = 42`

### SOM + Ward
- Topology: 1 × N rectangular grid (N = max(4 · k, 12) neurons)
- `sigma = 1.0`
- `learning_rate = 0.5`
- `iterations = 2000`
- Ward consolidation on prototype vectors
- Best-Matching Unit (BMU) assignment to fibers
- `random_seed = 42`

### Bayesian GMM
- `n_components = k` (truncation)
- `covariance_type = "full"`
- `weight_concentration_prior_type = "dirichlet_process"`
- `weight_concentration_prior = 1/k` (default)
- `tol = 1e-3`
- `reg_covar = 1e-6`
- `max_iter = 400`
- `n_init = 5`
- `random_state = 42`

## Internal validation indices

| Metric              | Library                                       | Direction      |
|---------------------|-----------------------------------------------|----------------|
| Silhouette          | `sklearn.metrics.silhouette_score`            | higher better  |
| Calinski–Harabasz   | `sklearn.metrics.calinski_harabasz_score`     | higher better  |
| Davies–Bouldin      | `sklearn.metrics.davies_bouldin_score`        | lower  better  |

The composite score normalizes all three within each `(Muscle, k)`
panel by min-max and inverts DB:

```
Sil_n    = minmax(Silhouette)
CH_n     = minmax(CH)
DBinv_n  = minmax(1 / DB)
Composite = (Sil_n + CH_n + DBinv_n) / 3
```

## Bootstrap & cross-validation

- Bootstrap iterations: `n_boot = 100` (per algorithm, per muscle).
  Each iteration uses `seed = RANDOM_STATE + b * 7`.
- 5-fold cross-validation: animals randomly permuted with seed 42,
  then split by `sklearn.model_selection.KFold(n_splits=5, shuffle=False)`.
- Gap statistic: `n_refs = 100` Monte Carlo replicates over the bounding-box
  uniform distribution. 1-SE rule for k selection.

## Inferential statistics

- Rank-biserial correlation r_rb with bootstrap 95% CI (B = 1,000).
- Welch t-test (`scipy.stats.ttest_ind`, `equal_var = False`).
- Mann-Whitney U test (`scipy.stats.mannwhitneyu`, two-sided).
- Chi-square (`scipy.stats.chi2_contingency`) stratified by diet,
  effect size as Cramér's V.

## Reproducibility checklist

- [x] Fixed `RANDOM_STATE = 42` everywhere.
- [x] Sub-sampling uses `numpy.random.default_rng(seed)`.
- [x] Bootstrap seeds are deterministic offsets from the base seed.
- [x] Z-scoring is per-muscle, computed from the muscle's mean and SD.
- [x] All outputs (tables, figures, labels) are deterministic given input.
