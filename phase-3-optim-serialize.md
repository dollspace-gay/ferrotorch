---
title: "Phase 3 — Optimizers + Serialization (ferrotorch-optim, ferrotorch-serialize)"
tags: [design-doc]
sources: []
contributors: [unknown]
created: 2026-03-15
updated: 2026-03-15
---


## Design Specification

### Summary

Two crates that complete the training loop. ferrotorch-optim provides first-order optimizers (SGD, Adam, AdamW, RMSprop, Adagrad) and a quasi-Newton optimizer (L-BFGS), plus learning rate schedulers for common training schedules (step decay, cosine annealing, warmup, plateau-based reduction). ferrotorch-serialize provides model persistence in three formats — msgpack state dicts for fast Rust-native save/load, SafeTensors for HuggingFace ecosystem interop with memory-mapped loading, and ONNX for cross-framework inference — plus full training checkpoints that bundle model weights, optimizer state, epoch counter, and RNG state into a single resumable artifact. Together they enable: `model.forward() -> loss.backward() -> optimizer.step() -> save checkpoint -> resume training`.

### Requirements

- REQ-1: An `Optimizer<T: Float>` trait must define the contract for all optimizers: `step()` to update parameters from accumulated gradients, `zero_grad()` to clear gradients, `lr()`/`set_lr()` for learning rate access, and `state_dict()`/`load_state_dict()` for checkpoint compatibility. All parameter updates must execute inside `no_grad()` to prevent the autograd engine from tracking optimizer arithmetic.
- REQ-2: Six optimizer implementations must be provided — SGD (with optional momentum, Nesterov acceleration, and weight decay), Adam (with bias correction and epsilon), AdamW (decoupled weight decay), RMSprop (with optional momentum and centered mode), Adagrad (with initial accumulator value), and L-BFGS (limited-memory quasi-Newton with line search). Each optimizer must store per-parameter state (momentum buffers, first/second moment estimates, step counts) keyed by a stable parameter identity derived from `Arc` pointer identity.
- REQ-3: Optimizers must accept parameter groups with per-group hyperparameters (learning rate, weight decay, momentum). A single optimizer instance must be able to apply different learning rates to different layers (e.g., lower learning rate for pretrained backbone, higher for new head).
- REQ-4: Six LR scheduler implementations must be provided — StepLR (multiplicative decay every N epochs), CosineAnnealingLR (cosine decay to a minimum LR), OneCycleLR (warmup to peak then anneal to minimum over a fixed step budget), ReduceLROnPlateau (reduce LR when a monitored metric stalls for a patience window), LinearWarmup (linear ramp from zero to base LR over N steps), and CosineWarmupScheduler (linear warmup followed by cosine decay). Schedulers must mutate the optimizer's learning rate via `set_lr()` and must be composable (warmup wrapping a base scheduler).
- REQ-5: ferrotorch-serialize must support saving and loading model state dicts (ordered map of parameter name to tensor data) in msgpack format via rmp-serde. Round-tripping a state dict through save/load must produce bit-identical tensor data.
- REQ-6: ferrotorch-serialize must support the SafeTensors format (HuggingFace specification) for both reading and writing. Loading must support memory-mapped I/O via `mmap` so that models larger than available RAM can be loaded lazily. Files produced by ferrotorch must be loadable by Python's `safetensors` library and vice versa.
- REQ-7: ferrotorch-serialize must support ONNX export by tracing a model's forward pass on example inputs and emitting a valid ONNX protobuf graph. The exported graph must include operator version information and be loadable by the ONNX runtime Python package for inference verification.
- REQ-8: ferrotorch-serialize must support full training checkpoints that bundle: model state dict, optimizer state dict, current epoch number, current global step count, and RNG state (both the Rust thread-local RNG seed and, if applicable, the CUDA RNG state). Resuming from a checkpoint must produce a training trajectory identical to uninterrupted training for deterministic workloads.
- REQ-9: All public functions in both crates must return `Result<T, FerrotorchError>`. Deserialization of corrupted files, shape mismatches during `load_state_dict()`, and missing keys must produce descriptive errors — never panics.

### Acceptance Criteria

- [ ] AC-1: `Sgd::new(model.parameters(), SgdConfig { lr: 0.01, momentum: 0.9, nesterov: true, weight_decay: 1e-4 })` constructs an optimizer. Calling `optimizer.step()` after `loss.backward()` updates parameter values. Verified by training a 2-layer MLP on XOR for 500 steps and reaching loss < 0.01 with SGD (momentum=0.9).
- [ ] AC-2: `Adam`, `AdamW`, `RMSprop`, `Adagrad`, and `Lbfgs` each converge on Rosenbrock's function `f(x,y) = (1-x)^2 + 100(y-x^2)^2` starting from `(-1.0, 1.0)` to within `||[x,y] - [1,1]|| < 0.1` in at most 10000 steps (Adam: lr=0.001; AdamW: lr=0.001, weight_decay=0.01; RMSprop: lr=0.001; Adagrad: lr=0.1; L-BFGS: lr=1.0, max line search steps=20).
- [ ] AC-3: Parameter groups work: an optimizer constructed with two groups (`[{params: backbone, lr: 1e-4}, {params: head, lr: 1e-2}]`) applies different learning rates to each group. After one `step()`, backbone parameters move by ~100x less than head parameters (for identical gradient magnitudes).
- [ ] AC-4: `StepLR`, `CosineAnnealingLR`, `OneCycleLR`, `ReduceLROnPlateau`, `LinearWarmup`, and `CosineWarmupScheduler` each produce the correct LR schedule. Verified by stepping each scheduler 100 times and asserting the LR at steps 0, 25, 50, 75, 99 matches the analytically computed values within `atol=1e-7`.
- [ ] AC-5: `save_state_dict(&state_dict, path)` followed by `load_state_dict(path)` produces a state dict with bit-identical tensor data for all parameters. Tested with a model containing f32 and f64 parameters, including tensors with shapes `[]` (scalar), `[1]`, `[768, 3072]`, and `[3, 224, 224]`.
- [ ] AC-6: `save_safetensors(&state_dict, path)` produces a file loadable by Python `safetensors.safe_open()`. Conversely, a SafeTensors file written by Python `safetensors.save_file()` is loadable by `load_safetensors(path)`. Both directions produce matching tensor data (verified via an integration test that shells out to Python).
- [ ] AC-7: `export_onnx(model, example_inputs, path)` produces a valid `.onnx` file. The file loads in `onnxruntime` and produces outputs matching the Rust model's forward pass within `rtol=1e-5` for f32. Tested with a model containing Linear, ReLU, and Softmax layers.
- [ ] AC-8: `save_checkpoint(path, model, optimizer, epoch, step, rng_state)` followed by `load_checkpoint(path)` restores all components. A training run that saves a checkpoint at epoch 5 and resumes from it produces the same parameter values at epoch 10 as an uninterrupted 10-epoch run (for a deterministic model with fixed seed, no dropout).
- [ ] AC-9: `load_state_dict` with a missing key returns `Err(FerrotorchError::MissingKey { key })`. `load_state_dict` with a shape mismatch returns `Err(FerrotorchError::ShapeMismatch { .. })`. Deserializing a truncated msgpack file returns `Err(FerrotorchError::Deserialization { .. })`. No panics in any error case.
- [ ] AC-10: `cargo test -p ferrotorch-optim -p ferrotorch-serialize` passes with 0 failures. Minimum 150 tests across both crates covering all optimizers, all schedulers, all serialization formats, error paths, and round-trip correctness.

### Architecture

### Crate Layout

```
ferrotorch-optim/
├── Cargo.toml
├── src/
│   ├── lib.rs                    # Public re-exports
│   ├── optimizer.rs              # Optimizer<T> trait, ParamGroup<T>, OptimizerState
│   ├── sgd.rs                    # SGD with momentum, Nesterov, weight decay
│   ├── adam.rs                   # Adam with bias correction
│   ├── adamw.rs                  # AdamW (decoupled weight decay)
│   ├── rmsprop.rs                # RMSprop with optional centering
│   ├── adagrad.rs                # Adagrad
│   ├── lbfgs.rs                  # L-BFGS with strong Wolfe line search
│   └── scheduler/
│       ├── mod.rs                # LrScheduler<T> trait
│       ├── step.rs               # StepLR
│       ├── cosine.rs             # CosineAnnealingLR
│       ├── one_cycle.rs          # OneCycleLR
│       ├── plateau.rs            # ReduceLROnPlateau
│       ├── warmup.rs             # LinearWarmup
│       └── cosine_warmup.rs      # CosineWarmupScheduler (warmup + cosine decay)
└── tests/
    ├── test_sgd.rs               # Convergence, momentum, Nesterov, weight decay
    ├── test_adam.rs               # Convergence, bias correction, epsilon
    ├── test_adamw.rs              # Decoupled weight decay vs L2 reg
    ├── test_rmsprop.rs            # Convergence, centered mode
    ├── test_adagrad.rs            # Convergence, accumulator behavior
    ├── test_lbfgs.rs              # Rosenbrock, line search termination
    ├── test_param_groups.rs       # Per-group hyperparameters
    ├── test_schedulers.rs         # LR values at specific steps for all schedulers
    └── test_state_dict.rs         # Optimizer state round-trip

ferrotorch-serialize/
├── Cargo.toml
├── src/
│   ├── lib.rs                    # Public re-exports
│   ├── state_dict.rs             # StateDict<T> type, save/load via msgpack
│   ├── safetensors.rs            # SafeTensors read/write with mmap support
│   ├── onnx.rs                   # ONNX protobuf export via forward-pass tracing
│   ├── checkpoint.rs             # TrainingCheckpoint: model + optimizer + epoch + RNG
│   └── error.rs                  # Serialization-specific error variants
└── tests/
    ├── test_state_dict.rs        # Round-trip, missing keys, shape mismatch, corruption
    ├── test_safetensors.rs       # Format compliance, mmap, Python interop
    ├── test_onnx.rs              # Export validity, onnxruntime inference match
    └── test_checkpoint.rs        # Save/resume determinism
```

### Core Types — ferrotorch-optim

**Optimizer trait** (`optimizer.rs`):
```rust
use ferrotorch_core::{Float, FerrotorchError, FerrotorchResult, Tensor};
use ferrotorch_nn::Parameter;
use std::collections::HashMap;

/// Per-parameter optimizer state (momentum buffers, moment estimates, etc.).
/// Keyed by parameter Arc pointer address for stable identity across steps.
pub type ParamStateKey = usize;

/// Opaque per-parameter state stored as named tensors.
pub struct ParamState<T: Float> {
    pub step: u64,
    pub buffers: HashMap<String, Tensor<T>>,
}

/// A group of parameters sharing hyperparameters.
pub struct ParamGroup<T: Float> {
    pub params: Vec<Parameter<T>>,
    pub lr: T,
    pub weight_decay: T,
    /// Group-specific overrides stored as key-value pairs.
    pub options: HashMap<String, T>,
}

/// Serializable optimizer state for checkpointing.
pub struct OptimizerStateDict<T: Float> {
    /// Per-parameter state, keyed by parameter name from the model's named_parameters().
    pub param_states: HashMap<String, ParamState<T>>,
    /// Per-group hyperparameters at time of save.
    pub param_groups: Vec<HashMap<String, T>>,
}

pub trait Optimizer<T: Float>: Send {
    /// Update all parameters using their accumulated gradients.
    /// All arithmetic executes inside no_grad() to avoid autograd tracking.
    fn step(&mut self) -> FerrotorchResult<()>;

    /// Closure-based step for optimizers that need to re-evaluate the loss (L-BFGS).
    /// Default implementation calls step() and ignores the closure.
    fn step_with_closure<F>(&mut self, closure: F) -> FerrotorchResult<Tensor<T>>
    where
        F: FnMut() -> FerrotorchResult<Tensor<T>>,
    {
        let _ = closure;
        self.step()?;
        Err(FerrotorchError::InvalidArgument {
            message: "step_with_closure requires an optimizer that supports closures (e.g., L-BFGS)".into(),
        })
    }

    /// Zero the .grad field on all managed parameters.
    fn zero_grad(&mut self);

    /// Return the base learning rate (first param group).
    fn lr(&self) -> T;

    /// Set the learning rate on all param groups.
    fn set_lr(&mut self, lr: T);

    /// Export optimizer state for checkpointing.
    fn state_dict(&self) -> OptimizerStateDict<T>;

    /// Restore optimizer state from a checkpoint.
    fn load_state_dict(&mut self, state: &OptimizerStateDict<T>) -> FerrotorchResult<()>;

    /// Return a mutable slice of param groups (for schedulers that set per-group LR).
    fn param_groups_mut(&mut self) -> &mut [ParamGroup<T>];
}
```

**SGD** (`sgd.rs`):
```rust
pub struct SgdConfig<T: Float> {
    pub lr: T,
    pub momentum: T,          // 0.0 = no momentum
    pub dampening: T,          // usually 0.0
    pub weight_decay: T,       // L2 regularization coefficient
    pub nesterov: bool,
}

pub struct Sgd<T: Float> {
    param_groups: Vec<ParamGroup<T>>,
    config: SgdConfig<T>,
    state: HashMap<ParamStateKey, ParamState<T>>,
}

impl<T: Float> Sgd<T> {
    pub fn new(params: Vec<Parameter<T>>, config: SgdConfig<T>) -> Self;
    pub fn new_with_groups(groups: Vec<ParamGroup<T>>, config: SgdConfig<T>) -> Self;
}
```

**Adam / AdamW** (`adam.rs`, `adamw.rs`):
```rust
pub struct AdamConfig<T: Float> {
    pub lr: T,
    pub betas: (T, T),         // (beta1, beta2), default (0.9, 0.999)
    pub eps: T,                // default 1e-8
    pub weight_decay: T,       // L2 reg for Adam, decoupled for AdamW
    pub amsgrad: bool,          // use max of past second moments
}

pub struct Adam<T: Float> {
    param_groups: Vec<ParamGroup<T>>,
    config: AdamConfig<T>,
    state: HashMap<ParamStateKey, ParamState<T>>,
    // ParamState buffers: "exp_avg" (first moment), "exp_avg_sq" (second moment),
    // optionally "max_exp_avg_sq" (for amsgrad)
}

pub struct AdamW<T: Float> {
    param_groups: Vec<ParamGroup<T>>,
    config: AdamConfig<T>,
    state: HashMap<ParamStateKey, ParamState<T>>,
}
```

**L-BFGS** (`lbfgs.rs`):
```rust
pub struct LbfgsConfig<T: Float> {
    pub lr: T,
    pub max_iter: usize,               // default 20
    pub max_eval: usize,               // default 25 (max function evaluations per step)
    pub tolerance_grad: T,             // default 1e-7
    pub tolerance_change: T,           // default 1e-9
    pub history_size: usize,           // default 10 (number of (s,y) pairs)
    pub line_search_fn: LineSearchFn,  // StrongWolfe (default)
}

pub enum LineSearchFn {
    StrongWolfe,
}

pub struct Lbfgs<T: Float> {
    param_groups: Vec<ParamGroup<T>>,
    config: LbfgsConfig<T>,
    state: LbfgsState<T>,
}
```

**LR Scheduler trait** (`scheduler/mod.rs`):
```rust
pub trait LrScheduler<T: Float> {
    /// Advance the scheduler by one step and update the optimizer's LR.
    fn step(&mut self, optimizer: &mut dyn Optimizer<T>);

    /// Return the current learning rate.
    fn current_lr(&self) -> T;

    /// Return the LR that will be used at a given step (for logging/plotting).
    fn lr_at_step(&self, step: usize) -> T;
}

/// Variant for ReduceLROnPlateau which needs a metric value.
pub trait MetricScheduler<T: Float>: LrScheduler<T> {
    /// Step with a metric observation (e.g., validation loss).
    fn step_with_metric(&mut self, metric: T, optimizer: &mut dyn Optimizer<T>);
}
```

**StepLR** (`scheduler/step.rs`):
```rust
pub struct StepLr<T: Float> {
    base_lr: T,
    step_size: usize,     // decay every N steps
    gamma: T,             // multiplicative factor (default 0.1)
    current_step: usize,
}
```

**CosineWarmupScheduler** (`scheduler/cosine_warmup.rs`):
```rust
pub struct CosineWarmupScheduler<T: Float> {
    base_lr: T,
    warmup_steps: usize,
    total_steps: usize,
    min_lr: T,
    current_step: usize,
}
```

### Core Types — ferrotorch-serialize

**StateDict** (`state_dict.rs`):
```rust
use ferrotorch_core::{Float, FerrotorchResult, Tensor};
use std::collections::BTreeMap;
use std::path::Path;

/// Ordered mapping from parameter names to tensor data.
/// BTreeMap ensures deterministic serialization order.
pub struct StateDict<T: Float> {
    pub tensors: BTreeMap<String, Tensor<T>>,
}

/// Save a state dict to msgpack format.
pub fn save_state_dict<T: Float>(
    state_dict: &StateDict<T>,
    path: impl AsRef<Path>,
) -> FerrotorchResult<()>;

/// Load a state dict from msgpack format.
pub fn load_state_dict<T: Float>(
    path: impl AsRef<Path>,
) -> FerrotorchResult<StateDict<T>>;
```

**SafeTensors** (`safetensors.rs`):
```rust
use ferrotorch_core::{Float, FerrotorchResult, Tensor};
use std::collections::BTreeMap;
use std::path::Path;

/// Save tensors in SafeTensors format (HuggingFace specification).
pub fn save_safetensors<T: Float>(
    state_dict: &StateDict<T>,
    path: impl AsRef<Path>,
) -> FerrotorchResult<()>;

/// Load tensors from SafeTensors format.
/// If mmap is true, the file is memory-mapped and tensors are loaded lazily.
pub fn load_safetensors<T: Float>(
    path: impl AsRef<Path>,
    mmap: bool,
) -> FerrotorchResult<StateDict<T>>;

/// Load a single tensor by name from a SafeTensors file (memory-mapped).
/// Useful for loading individual layers without reading the full file.
pub fn load_safetensors_tensor<T: Float>(
    path: impl AsRef<Path>,
    name: &str,
) -> FerrotorchResult<Tensor<T>>;
```

**ONNX Export** (`onnx.rs`):
```rust
use ferrotorch_core::{Float, FerrotorchResult, Tensor};
use ferrotorch_nn::Module;
use std::path::Path;

/// Export a model to ONNX format by tracing its forward pass.
///
/// The model is run once on the example inputs to record the computation graph.
/// Dynamic axes can be specified for dimensions that vary at inference time
/// (e.g., batch size).
pub fn export_onnx<T, M>(
    model: &M,
    example_inputs: &[Tensor<T>],
    path: impl AsRef<Path>,
    config: OnnxExportConfig,
) -> FerrotorchResult<()>
where
    T: Float,
    M: Module<T>;

pub struct OnnxExportConfig {
    pub opset_version: u32,                        // default 17
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
    pub dynamic_axes: HashMap<String, Vec<usize>>, // tensor name -> axis indices
}
```

**Training Checkpoint** (`checkpoint.rs`):
```rust
use ferrotorch_core::{Float, FerrotorchResult};
use crate::StateDict;
use ferrotorch_optim::OptimizerStateDict;
use std::path::Path;

/// Everything needed to resume training from exactly where it stopped.
pub struct TrainingCheckpoint<T: Float> {
    pub model_state: StateDict<T>,
    pub optimizer_state: OptimizerStateDict<T>,
    pub epoch: usize,
    pub global_step: u64,
    pub rng_state: RngState,
}

/// Captured RNG state for deterministic resumption.
pub struct RngState {
    pub cpu_seed: u64,
    pub cuda_seeds: Vec<u64>,  // per-device CUDA RNG seeds (empty if CPU-only)
}

/// Save a training checkpoint to disk (msgpack container wrapping all components).
pub fn save_checkpoint<T: Float>(
    path: impl AsRef<Path>,
    model_state: &StateDict<T>,
    optimizer_state: &OptimizerStateDict<T>,
    epoch: usize,
    global_step: u64,
    rng_state: &RngState,
) -> FerrotorchResult<()>;

/// Load a training checkpoint from disk.
pub fn load_checkpoint<T: Float>(
    path: impl AsRef<Path>,
) -> FerrotorchResult<TrainingCheckpoint<T>>;
```

### Parameter Identity

Optimizers must track per-parameter state across `step()` calls. Parameters are identified by the `Arc` pointer address of their underlying `TensorStorage`. This avoids requiring users to name parameters when constructing optimizers — identity is intrinsic:

```rust
fn param_key<T: Float>(param: &Parameter<T>) -> ParamStateKey {
    Arc::as_ptr(&param.data.storage()) as usize
}
```

This is stable for the lifetime of the parameter (Arc is never reallocated). When saving optimizer state to a checkpoint, the pointer keys are mapped to string names via the model's `named_parameters()` method, ensuring checkpoint portability across program restarts where pointer addresses change.

### Optimizer Step Execution Model

Every optimizer's `step()` method follows this pattern:

1. Enter `no_grad()` scope (prevents autograd from tracking parameter updates)
2. For each param group, for each parameter in the group:
   a. Read `param.grad()` — skip if `None`
   b. Apply weight decay (L2 for SGD/Adam, decoupled for AdamW)
   c. Update per-parameter state (momentum buffer, moment estimates, etc.)
   d. Compute the parameter update delta
   e. Apply: `param.data -= lr * delta` (in-place mutation)
3. Exit `no_grad()` scope

The in-place mutation of `param.data` uses raw slice access on the underlying `TensorStorage`, bypassing tensor operations entirely. This is safe because `no_grad()` guarantees no graph is being built, and `step()` takes `&mut self` preventing concurrent access.

### Serialization Format Details

**Msgpack state dict**: Tensor data is serialized as a header (name, dtype tag, shape as `Vec<usize>`) followed by raw bytes in little-endian format. rmp-serde handles the framing; the tensor bytes are inlined as msgpack binary blobs. This avoids floating-point text encoding overhead.

**SafeTensors**: Follows the HuggingFace specification exactly — an 8-byte little-endian header size, a JSON header containing tensor metadata (name, dtype string, shape, byte offsets), followed by concatenated raw tensor data. The `safetensors` crate (version 0.4) is used as the implementation, with ferrotorch providing a thin wrapper that converts between `Tensor<T>` and the crate's `TensorView` type.

**ONNX**: The model's forward pass is traced by running it once with gradient tracking enabled but without calling backward. The resulting computation graph nodes are mapped to ONNX operator types. This reuses the existing autograd graph structure — each `GradFn` has a `name()` that maps to an ONNX operator (e.g., "MatmulForward" -> "MatMul", "ReluForward" -> "Relu"). Weights are embedded as ONNX initializers. The protobuf is constructed using the `prost` crate with ONNX proto definitions.

### Dependencies

**ferrotorch-optim:**

| Crate | Version | Purpose |
|-------|---------|---------|
| `ferrotorch-core` | workspace | Tensor, Float, FerrotorchError, no_grad |
| `ferrotorch-nn` | workspace | Parameter, Module (for parameter extraction) |

**ferrotorch-serialize:**

| Crate | Version | Purpose |
|-------|---------|---------|
| `ferrotorch-core` | workspace | Tensor, Float, FerrotorchError |
| `ferrotorch-nn` | workspace | Module (for ONNX tracing and state_dict) |
| `ferrotorch-optim` | workspace | OptimizerStateDict (for training checkpoints) |
| `serde` | 1.0 | Serialization framework |
| `rmp-serde` | 1.3 | Msgpack encoding/decoding for state dicts and checkpoints |
| `safetensors` | 0.4 | HuggingFace SafeTensors format |
| `prost` | 0.13 | Protobuf encoding for ONNX export |
| `memmap2` | 0.9 | Memory-mapped file I/O for lazy SafeTensors loading |

### Error Extensions

FerrotorchError gains new variants for serialization failures:

```rust
// Added to ferrotorch-core's FerrotorchError enum
#[error("missing key in state dict: {key}")]
MissingKey { key: String },

#[error("unexpected key in state dict: {key}")]
UnexpectedKey { key: String },

#[error("serialization failed: {message}")]
Serialization { message: String },

#[error("deserialization failed: {message}")]
Deserialization { message: String },

#[error("ONNX export failed: unsupported operation {op_name}")]
OnnxUnsupportedOp { op_name: String },

#[error("I/O error: {0}")]
Io(#[from] std::io::Error),
```

### Test Strategy

1. **Optimizer convergence**: Each optimizer is tested on at least two functions — a simple quadratic `f(x) = x^2` (verifies basic gradient descent) and Rosenbrock (verifies handling of ill-conditioned curvature). Convergence is asserted numerically, not just "loss decreased."
2. **PyTorch reference**: For SGD and Adam, run identical training loops in PyTorch and ferrotorch on a 2-layer MLP with fixed seeds. Assert parameter values match within `atol=1e-5` after 100 steps. This catches subtle bugs in bias correction, weight decay application order, and momentum update formulas.
3. **Scheduler correctness**: Each scheduler is tested by computing LR at every step for a known configuration and comparing against analytically derived values. No numerical optimization needed — these are closed-form formulas.
4. **Serialization round-trip**: Save then load, assert bit-identical for msgpack and SafeTensors. For ONNX, assert inference output matches within floating-point tolerance.
5. **Cross-format interop**: SafeTensors files are validated against Python's safetensors library. ONNX files are validated against onnxruntime. These are integration tests gated behind a `python-interop` feature flag.
6. **Checkpoint resume**: Train for N epochs, save checkpoint, resume, train for M more epochs. Compare final parameters against an uninterrupted N+M epoch run. Must be bit-identical for deterministic models.
7. **Error paths**: Corrupt files, missing keys, shape mismatches, and type mismatches must all produce `Err`, never panic.

### Out of Scope

- Gradient clipping — that is a utility on `Tensor`/`Module`, not an optimizer concern; it belongs in ferrotorch-nn or ferrotorch-core
- Mixed-precision optimizer state (storing moments in fp16) — future optimization after GPU backend
- Distributed optimizer (ZeroRedundancyOptimizer) — that is Phase 7 (ferrotorch-distributed)
- Model compilation/optimization before ONNX export — that is Phase 8 (ferrotorch-jit)
- Custom serialization formats beyond msgpack/SafeTensors/ONNX
- Python bindings for save/load — that is a late phase (ferrotorch-python)
- Automatic mixed precision (AMP) / loss scaling — depends on GPU backend
- Optimizer fusion (fusing parameter updates into a single kernel) — GPU optimization concern

### resolved questions

### Q1: Parameter identity — Arc pointer vs explicit naming
**Decision**: Arc pointer address for runtime identity, mapped to string names only during serialization.

Requiring users to name parameters when constructing optimizers (like `optimizer.add_param("layer1.weight", param)`) is verbose and error-prone. Instead, optimizers key their internal state by `Arc::as_ptr()` on the parameter's storage. When checkpointing, the optimizer traverses the model's `named_parameters()` to build a pointer-to-name mapping, then serializes state with string keys. On load, the reverse mapping restores state to the correct parameters. This matches PyTorch's behavior where `optimizer.state` is keyed by parameter object identity.

### Q2: AdamW as separate struct vs Adam with a flag
**Decision**: Separate struct (`AdamW`).

Decoupled weight decay (AdamW) applies weight decay directly to parameters *before* the Adam update, while L2 regularization (Adam with weight_decay > 0) adds the regularization gradient to the parameter gradient *before* computing moments. These are mathematically different and produce different training dynamics. Conflating them behind a flag invites misuse. Separate structs with shared `AdamConfig` make the distinction explicit at the type level.

### Q3: ONNX export — graph tracing vs manual construction
**Decision**: Graph tracing via forward pass.

Running the model's forward pass once with example inputs produces a computation graph in the autograd engine. This graph is walked to extract the operator sequence and tensor shapes. Each `GradFn::name()` maps to an ONNX operator via a hardcoded lookup table. This approach reuses existing infrastructure and automatically handles complex models without requiring users to manually specify the graph. Operations not in the lookup table cause `export_onnx` to return `Err(FerrotorchError::OnnxUnsupportedOp { .. })` with the unrecognized operation name.

### Q4: Checkpoint format — single file vs directory
**Decision**: Single file (msgpack container).

A single `.ckpt` file containing all checkpoint components (model state, optimizer state, metadata) is simpler to manage than a directory of files. The msgpack container wraps each component as a named top-level entry. File sizes are dominated by tensor data regardless of format overhead, so the msgpack framing cost is negligible. For very large models where memory-mapped loading matters, users should save model weights separately in SafeTensors format and use the checkpoint file only for optimizer state and metadata.

### Q5: LR scheduler composability — wrapping vs chaining
**Decision**: Wrapping (warmup schedulers wrap a base scheduler).

`CosineWarmupScheduler` internally contains the warmup logic and the cosine decay logic in a single struct, selecting behavior based on the current step count vs the warmup horizon. This avoids the complexity of a generic scheduler composition framework. Users who need custom schedules can implement `LrScheduler<T>` directly — the trait has only three methods.

