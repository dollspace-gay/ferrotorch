//! Sklearn-compatible adapter for ferrotorch.
//!
//! Bridges [`ferrotorch_core::Tensor`] to `ndarray::{Array1, Array2}` so
//! ferrolearn (a scikit-learn equivalent in Rust) can be driven from a
//! tensor-shaped pipeline.
//!
//! See the crate-level [`README`](https://crates.io/crates/ferrotorch-ml)
//! for the full design rationale. The short version:
//!
//! - [`adapter`] — `Tensor ↔ ndarray` round-trip primitives.
//! - [`metrics`] — sklearn classification + regression metrics shaped
//!   for `&Tensor<T>` inputs.
//! - [`datasets`] — built-in toy + synthetic generators returning tensor
//!   pairs.
//! - [`preprocess`], [`decomposition`], [`model_selection`] — direct
//!   re-exports from ferrolearn so the full sklearn surface is one
//!   import away.
//!
//! # CPU-only by design, GPU input transparently materialised
//!
//! ferrolearn is a CPU-only library (built on `ndarray` + `faer`).
//! `ferrotorch-ml` accepts tensors on **any** device — if the input
//! lives on CUDA / XPU, the data is moved to host memory before
//! conversion. This matches the torch idiom of `loss.cpu().item()`
//! where a single materialisation step crosses the device boundary.
//!
//! Compute crates (`ferrotorch-core`, `-nn`, `-gpu`) continue to enforce
//! the strict `/rust-gpu-discipline` no-silent-fallback rule. The
//! relaxation here applies **only** to this dedicated bridge crate —
//! the function names (`tensor_to_array2`, etc.) make the device
//! crossing self-evident at the call site.
//!
//! If you want fail-fast strictness in a hot path, add an explicit
//! `assert!(t.device().is_cpu())` before calling the adapter.

pub mod adapter;
pub mod datasets;
pub mod metrics;

// Direct re-exports of ferrolearn modules. These are CPU-only and
// operate on `ndarray::{Array1, Array2}` — pair them with `adapter::*`
// to round-trip from `Tensor`.
pub mod preprocess {
    //! Re-export of [`ferrolearn_preprocess`] — sklearn-style fit/transform
    //! preprocessors (`StandardScaler`, `MinMaxScaler`, `OneHotEncoder`,
    //! `PolynomialFeatures`, `KBinsDiscretizer`, `SimpleImputer`, etc.).
    //! Drive these from `Tensor` inputs by routing through
    //! [`super::adapter::tensor_to_array2`] and back via
    //! [`super::adapter::array2_to_tensor`].
    pub use ferrolearn_preprocess::*;
}

pub mod decomposition {
    //! Re-export of [`ferrolearn_decomp`] — dimensionality reduction
    //! (PCA, IncrementalPCA, FastICA, NMF, KernelPCA, t-SNE, UMAP,
    //! Isomap, LLE, MDS, FactorAnalysis, TruncatedSVD, SparsePCA).
    pub use ferrolearn_decomp::*;
}

pub mod model_selection {
    //! Re-export of [`ferrolearn_model_sel`] — cross-validation splitters
    //! (`KFold`, `StratifiedKFold`, `GroupKFold`, `ShuffleSplit`,
    //! `train_test_split`), pipelines (`make_pipeline`, `Pipeline`),
    //! grid search (`GridSearchCV`, `RandomizedSearchCV`), and the
    //! dummy / multiclass / multioutput meta-estimators.
    pub use ferrolearn_model_sel::*;
}
