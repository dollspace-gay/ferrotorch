# Audit: ferrotorch ↔ ferray + ferrolearn integration

ferray and ferrolearn are sister projects (same author, parallel
ecosystems):
- **ferray** (v0.3.0, 2026-04-30): NumPy-equivalent in Rust. 17 crates.
  Major work just landed: hand-tuned BLAS kernels, OpenBLAS-port-quality
  GEMM, mixed-precision GEMM, AVX-512 / VNNI int8.
- **ferrolearn** (v0.3.0, 2026-04-29): scikit-learn-equivalent in Rust.
  21 crates. Just hit ~35% sklearn parity with 54 estimators bound to
  Python and head-to-head benched against sklearn 1.8.0 (geomean 5-10×
  faster across families).

ferrotorch lives one layer up: deep-learning-on-Rust. Currently it
depends on **only 4 ferray crates** and has **no integration with
ferrolearn**.

## Current ferrotorch ↔ ferray relationship

### What ferrotorch consumes today

| ferray crate | Usage in ferrotorch | Where |
|---|---|---|
| `ferray-core` | `Array<T, IxN>`, `DType`, `Element`, `FerrayError`, `Ix2`/`IxDyn` | `ferrotorch-core::dtype`, `linalg`, `error` |
| `ferray-ufunc` | universal-function kernels | (workspace dep, used internally) |
| `ferray-linalg` | `svd`, `solve`, `det`, `inv`, `qr`, `cholesky` | `ferrotorch-core::linalg` |
| `ferray-random` | RNG | (workspace dep, used internally) |

Used in **`ferrotorch-core` only.** All other ferrotorch crates depend on
`ferrotorch-core`, not directly on ferray.

### What ferrotorch does NOT consume

13 of the 17 ferray crates are unused by ferrotorch:

| ferray crate | Could fill ferrotorch gap |
|---|---|
| `ferray-fft` | **`fftn`, `ifftn`, `rfftn`, `irfftn`, `hfft`, `ihfft`, `fftfreq`, `rfftfreq`, `fftshift`, `ifftshift`** — exactly the gap in `ferrotorch-core::fft` (per audit #1) |
| `ferray-linalg` (more) | **`eig`, `eigh`, `eigvals`, `eigvalsh`, `lstsq`, `lu`, `matrix_power`, `tensorsolve`, `tensorinv`, `svdvals`** plus batched variants — closes the linalg gap from audit #1 |
| `ferray-stats` | correlation, histogram, sorting, set ops — could feed `ferrotorch-core::ops::search` and a future stats module |
| `ferray-polynomial` | `chebyshev`, `hermite`, `hermite_e`, `laguerre`, `legendre`, fitting, roots — exactly the orthogonal-polynomial families in `torch.special` (audit #1 special-fn gap) |
| `ferray-window` | `bartlett`, `blackman`, `hamming`, `hanning`, `kaiser` — `torch.signal.windows` (currently a top-level pytorch gap) |
| `ferray-ma` | masked arrays — `torch.masked` (currently a top-level pytorch gap) |
| `ferray-stride-tricks` | `as_strided` — currently missing in ferrotorch (audit #1 indexing gap) |
| `ferray-strings` | string ops — `torch.text` adjacent |
| `ferray-numpy-interop` | numpy/Arrow/Polars round-trip, RecordBatch, FromArrow — would unlock zero-copy data pipelines |
| `ferray-autodiff` | forward-mode dual numbers | **DO NOT integrate** — duplicates ferrotorch's own `DualTensor` (audit #1 strengths). |
| `ferray-test-oracle` | property-based testing oracle | testing infrastructure only |
| `ferray-core-macros` | macros | already pulled transitively |
| `compat` | numpy compat layer | not needed |

### Recent ferray work that helps ferrotorch directly

From the 2026-04-30 ferray 0.3.0 changelog:
- **OpenBLAS Haswell/Skylake-X SGEMM/DGEMM/CGEMM/ZGEMM kernel ports**
  (#654, #660-665) — `ferrotorch-core` matmul on CPU now has a top-tier
  fallback path
- **Mixed-precision GEMM (gemm_bf16_f32, gemm_f16_f32)** (#673) —
  directly relevant for Llama-class fp16/bf16 paths
- **Quantized GEMM (i8 × i8, u8 × i8, with VNNI fast path)** (#666-672)
  — feeds `ferrotorch-core::quantize::quantized_matmul`
- **TRSM primitives** (#674) — block-recursive triangular solve, useful
  for batched normal equations, kalman filter updates, etc.
- **Optional OpenBLAS backend** (#675) — perf floor guarantee in
  CPU-heavy environments

### Recommendation: deepen the ferray integration

Add these workspace dependencies to `ferrotorch/Cargo.toml`:

```toml
ferray-fft = "0.3.0"
ferray-stats = "0.3.0"
ferray-window = "0.3.0"
ferray-polynomial = "0.3.0"
ferray-ma = "0.3.0"
ferray-stride-tricks = "0.3.0"
ferray-numpy-interop = "0.3.0"  # optional, behind feature flag
```

Then:
1. **Expand `ferrotorch-core::fft`** by re-exporting / wrapping the
   ferray-fft API. Closes the FFT gap from audit #1 immediately.
2. **Expand `ferrotorch-core::linalg`** by wrapping the new
   ferray-linalg APIs (`eig`, `eigh`, `eigvals`, `eigvalsh`, `lstsq`,
   `lu`, `matrix_power`, `tensorsolve`, `tensorinv`, `svdvals`).
   Closes the linalg gap from audit #1 immediately.
3. **Add `ferrotorch-core::special::polynomials`** wrapping
   ferray-polynomial (chebyshev_polynomial_t/u/v/w, hermite_polynomial_h/he,
   laguerre_polynomial_l, legendre_polynomial_p — the long tail of
   `torch.special` that the original audit flagged).
4. **Add `ferrotorch-core::signal`** as a new module wrapping
   ferray-window — closes the missing `torch.signal.windows` gap.
5. **Add `ferrotorch-core::masked`** wrapping ferray-ma — closes
   `torch.masked`.
6. **Add `ferrotorch-core::stride_tricks::as_strided`** wrapping
   ferray-stride-tricks — closes the indexing gap.
7. **Optional: enable `ferray-numpy-interop`** behind a feature flag to
   unlock Arrow / Polars / numpy round-trips for `ferrotorch-data`.
   Big win for tabular ML pipelines.

**Cost**: a handful of thin-wrapper modules (no new logic, just type
adapters between `Tensor<T>` and `ferray::Array<T, IxN>`).

**Benefit**: closes ~60% of the gaps from the original 15-crate audit
in a few hours of plumbing.

## Current ferrotorch ↔ ferrolearn relationship

**No direct dependency.** ferrolearn is built on `ndarray::Array2`
(via `ndarray-linalg` + `faer`), not on ferray. ferrotorch is built on
`ferray::Array`. The two never meet.

### What ferrolearn provides (relative to ferrotorch needs)

| ferrolearn crate | Could fill ferrotorch gap |
|---|---|
| `ferrolearn-metrics` | sklearn metrics: classification (precision, recall, F1, AUC, log_loss, brier, ...), regression (MAE, MSE, R², MAPE, d2_*, pinball, tweedie), ranking (ndcg, map, lrap), clustering (ARI, NMI, silhouette), pairwise (cosine, manhattan, etc.). **`ferrotorch-train::metric` is much thinner** (only LossMetric, AccuracyMetric, TopKAccuracy, RunningAverage). Audit #5 flagged this as a gap. |
| `ferrolearn-preprocess` | StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer, KBinsDiscretizer, OneHotEncoder, OrdinalEncoder, LabelEncoder, PolynomialFeatures, KNNImputer, IterativeImputer, ColumnTransformer. **No ferrotorch counterpart.** |
| `ferrolearn-datasets` | `make_classification`, `make_regression`, `make_blobs`, `make_moons`, `make_circles`, `make_friedman1/2/3`, `load_iris`, `load_digits`, ..., `fetch_california_housing`, `fetch_openml`. **`ferrotorch-vision::datasets` only has MNIST / CIFAR**; tabular dataset gap. Audit #6 flagged ImageFolder / ImageNet but tabular is a separate domain. |
| `ferrolearn-decomp` | PCA, IncrementalPCA, TruncatedSVD, FastICA, NMF, KernelPCA, FactorAnalysis, SparsePCA, DictionaryLearning, t-SNE, UMAP, Isomap, LocallyLinearEmbedding. **Useful for activation analysis / feature engineering inside DL pipelines.** No ferrotorch counterpart. |
| `ferrolearn-cluster` | KMeans, MiniBatchKMeans, DBSCAN, OPTICS, AgglomerativeClustering, Birch, GaussianMixture, SpectralClustering, HDBSCAN, MeanShift. **Useful for embedding clustering.** |
| `ferrolearn-model-sel` | `KFold`, `StratifiedKFold`, `GroupKFold`, `ShuffleSplit`, `cross_val_score`, `GridSearchCV`, `RandomizedSearchCV`, `Pipeline`, `FeatureUnion`, `make_pipeline`, `make_union`. **No ferrotorch counterpart.** Big gap for hyperparameter search. |
| `ferrolearn-tree` | RandomForest, GradientBoosting, HistGradientBoosting, AdaBoost, ExtraTrees, Bagging, DecisionTree. **Useful as baselines / ensembles with deep models.** |
| `ferrolearn-linear` | LinearRegression, Ridge, Lasso, LogisticRegression, ElasticNet, SVMs (LinearSVC, SVC, SVR), HuberRegressor, QuantileRegressor, BayesianRidge, ARD, MLP. **Baselines.** |
| `ferrolearn-neighbors` | KNN classifier/regressor, NearestCentroid. Could power KNN-based retrieval / classification. |
| `ferrolearn-bayes` | GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, conjugate priors. Bayesian baselines. |
| `ferrolearn-kernel` | KernelRidge, GaussianProcessRegressor/Classifier, Nystroem, RBFSampler. Useful for small-data regimes / GP-based hyperparameter search. |
| `ferrolearn-neural` | sklearn-style MLP / RBM. (Duplicates ferrotorch's MLP capability — skip integrating.) |
| `ferrolearn-covariance` | EmpiricalCovariance, GraphicalLasso, LedoitWolf. Useful for KFAC natural gradient (already in `ferrotorch-optim::natural_gradient`). |
| `ferrolearn-sparse` | sparse matrix utilities. Parallel to `ferrotorch-core::sparse`. |
| `ferrolearn-numerical` | scipy-style special functions, decompositions. Parallel to `ferrotorch-core::special`. |

### Layering challenge

ferrolearn uses `ndarray::Array2`. ferrotorch uses `ferray::Array` /
`ferrotorch::Tensor`. **They cannot exchange tensors directly.**

Options:

#### Option A: Add adapter layer in a new `ferrotorch-ml` crate

Create a new workspace crate that bridges:

```rust
ferrotorch::Tensor<T>  ←→  ndarray::Array2<T>  ←→  ferrolearn estimators
```

The adapter is thin (`Tensor::to_ndarray()` / `Tensor::from_ndarray()`
helpers) and the new crate re-exports curated ferrolearn functionality
shaped for ferrotorch users.

What it would expose:
- `ferrotorch_ml::metrics::*` — full sklearn metrics, takes `Tensor`s
- `ferrotorch_ml::preprocess::*` — scalers etc. that fit/transform Tensors
- `ferrotorch_ml::model_selection::*` — `KFold`, `GridSearchCV`-style
  helpers wrapping ferrolearn-model-sel
- `ferrotorch_ml::decomposition::*` — PCA / ICA on activation tensors
- `ferrotorch_ml::cluster::*` — KMeans / DBSCAN on embeddings
- `ferrotorch_ml::datasets::*` — tabular dataset generators

**Cost**: 1 new crate, ~thin adapter functions (probably 500-1500 LOC).

**Benefit**:
- Closes the metrics gap from audit #5 with sklearn-compat surface
- Closes the preprocessing gap (no ferrotorch counterpart today)
- Adds tabular datasets, CV machinery, classical ML baselines

#### Option B: ferrolearn adopts ferray as its array backend

ferrolearn switches from `ndarray` to `ferray::Array`. This would be a
massive ferrolearn refactor (~21 crates) and is **not ferrotorch's call to
make**. If/when it happens, ferrotorch + ferrolearn become natively
compatible.

**This is the right long-term outcome but the wrong short-term
expectation.**

#### Option C: Do nothing

Treat ferrolearn as an external tool. Users do
`tensor.to_ndarray()` themselves to call ferrolearn. Documented but
unergonomic.

### Recommendation

**Adopt Option A.** Create `ferrotorch-ml` (or extend `ferrotorch-train`
with a `metric` module re-export of sklearn metrics).

The two highest-impact functional surfaces in ferrolearn for ferrotorch
users are:

1. **`ferrolearn-metrics`** — directly fills audit #5's metric gap.
   Wraps every fitted-tensor → score function: `accuracy_score`,
   `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`,
   `confusion_matrix`, `r2_score`, `mean_squared_error`, etc. ferrotorch
   users currently must hand-roll these.

2. **`ferrolearn-model-sel`** — `KFold`, `cross_val_score`, `GridSearchCV`.
   ferrotorch has no CV machinery. Adding this turns
   `ferrotorch-train::Learner` into a hyperparameter-search-capable
   trainer.

Lower priority: `ferrolearn-preprocess`, `ferrolearn-datasets`,
`ferrolearn-decomp` (PCA on activations is the killer use case here).

## Concrete actions

### Phase 1 (small, immediate)
1. Add `ferray-fft` + `ferray-linalg` (more APIs) as direct deps in
   `ferrotorch-core/Cargo.toml`.
2. Wire `ferray-fft::*` into `ferrotorch-core::fft` (closes audit #1 fft
   gap).
3. Wire `ferray-linalg::eig/eigh/eigvals/eigvalsh/lstsq/lu/...` into
   `ferrotorch-core::linalg` (closes audit #1 linalg gap).
4. Add `ferray-window` → new `ferrotorch-core::signal::windows` module.
5. Add `ferray-polynomial` → expand `ferrotorch-core::special` with
   Chebyshev / Hermite / Laguerre / Legendre polynomial families.
6. Add `ferray-stride-tricks` → `ferrotorch-core::Tensor::as_strided`.

### Phase 2 (medium, deliberate)
7. Add `ferray-ma` → new `ferrotorch-core::masked` module
   (`torch.masked` parity).
8. Add `ferray-numpy-interop` (feature-gated) → enables Arrow / Polars
   data pipelines in `ferrotorch-data`.

### Phase 3 (new crate)
9. Create `ferrotorch-ml` workspace crate. Bridge to ferrolearn-metrics,
   ferrolearn-preprocess, ferrolearn-model-sel, ferrolearn-decomp via
   `Tensor ↔ Array2` adapters.

### What NOT to integrate
- **`ferray-autodiff`** — duplicates `DualTensor` in ferrotorch-core.
  Skip.
- **`ferrolearn-neural`** — duplicates ferrotorch-nn MLP. Skip.
- **`ferrolearn-linear/tree/neighbors/cluster/bayes/kernel`** as direct
  ferrotorch-deps — these are sklearn ML, parallel to ferrotorch's deep
  learning. Users compose them through `ferrotorch-ml` if Phase 3 lands;
  otherwise they're external tools.

## Status

ferrotorch is **under-integrated with ferray**. The 4-of-17 ratio
leaves a lot of recently-shipped numpy-grade infrastructure on the table.
Phase 1 + 2 of the recommendations above would close roughly 60% of the
gaps identified in the original 15-crate audit, with a few weeks of
plumbing work and zero new algorithmic implementations.

ferrolearn is **not currently integrated at all**. The layering mismatch
(ndarray vs ferray) means a `ferrotorch-ml` adapter crate is the
shortest path to making sklearn metrics, preprocessing, CV, and
PCA available inside ferrotorch pipelines.

## Related issues
- (this audit doc) #563 — Audit ferray + ferrolearn integration with ferrotorch
- existing #515 (NTK RoPE), #459 (distributed backends), #451 (MPS) are
  unrelated to this integration question
