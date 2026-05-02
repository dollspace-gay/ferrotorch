# ferrotorch-ml

Sklearn-compatible adapter for ferrotorch. Bridges
`ferrotorch_core::Tensor` to `ndarray::{Array1, Array2}` so you can drive
[ferrolearn](https://crates.io/crates/ferrolearn) (a scikit-learn
equivalent for Rust) from a tensor-shaped pipeline.

## Quick start

```rust,no_run
use ferrotorch_core::tensor;
use ferrotorch_ml::{adapter, metrics};

let y_true = tensor(&[1.0_f64, 0.0, 1.0, 1.0]).unwrap();
let y_pred = tensor(&[0.9_f64, 0.1, 0.8, 0.7]).unwrap();
let r2 = metrics::r2_score(&y_true, &y_pred).unwrap();
```

## Scope

- **Adapter primitives** (`adapter::*`) — `tensor_to_array1`,
  `tensor_to_array2`, `array1_to_tensor`, `array2_to_tensor`.
- **Curated metric wrappers** (`metrics::*`) — sklearn classification +
  regression metrics shaped for `&Tensor<T>` inputs.
- **Curated dataset helpers** (`datasets::*`) — `make_classification`,
  `make_regression`, `make_blobs`, `make_moons`, `make_circles`,
  `load_iris`, `load_wine`, `load_breast_cancer` — all returning
  `(Tensor<F>, Tensor<F>)` instead of `(Array2, Array1)`.
- **Re-exports** of `ferrolearn-preprocess`, `ferrolearn-decomposition`,
  `ferrolearn-model-sel` so users have the full sklearn surface
  one-import away.

## CPU-only by design, GPU input transparently materialised

ferrolearn is a CPU-only library (built on `ndarray` + `faer`).
`ferrotorch-ml` accepts tensors on any device — if the input lives on
CUDA / XPU, the data is moved to host memory automatically before
conversion. This matches torch's `loss.cpu().item()` idiom: a single
materialisation step crosses the device boundary, and the function
name (`tensor_to_array2`, etc.) makes the crossing obvious at the
call site.

```rust,ignore
// Works on either device — same call.
let arr = ferrotorch_ml::adapter::tensor_to_array2(&my_tensor)?;
```

The relaxation applies **only** to this dedicated bridge crate. Compute
crates (`ferrotorch-core`, `-nn`, `-gpu`) continue to enforce the
strict `/rust-gpu-discipline` no-silent-fallback rule, where hidden
device transfers would be a real hazard.

If you want fail-fast strictness in a hot path, add an explicit
`assert!(t.device().is_cpu())` before the adapter call.
