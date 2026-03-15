# Feature: Phase 3 — Optimizers + Serialization (ferrotorch-optim, ferrotorch-serialize)

## Summary
Two crates that complete the training loop. ferrotorch-optim provides first-order optimizers (SGD, Adam, AdamW, RMSprop, Adagrad) and a quasi-Newton optimizer (L-BFGS) with full parameter group support, composable learning rate schedulers (step decay, cosine annealing, warmup, plateau-based reduction, and arbitrary sequential composition), and a `GradScaler` for mixed-precision training with dynamic loss scaling. ferrotorch-serialize provides model persistence with SafeTensors as the primary format (HuggingFace ecosystem standard, memory-mapped loading), msgpack as a secondary internal format for optimizer state, ONNX export (opset 17+) for cross-framework inference, and PyTorch `.pt`/`.pth` checkpoint import for pretrained weight loading — plus full training checkpoints that bundle model weights, optimizer state (momentum buffers, moment estimates, step counts), GradScaler state, epoch counter, and RNG state into a single resumable artifact. Together they enable: `model.forward() -> scaler.scale(loss).backward() -> scaler.step(optimizer) -> scaler.update() -> save checkpoint -> resume training`.

## Requirements
- REQ-1: An `Optimizer<T: Float>` trait must define the contract for all optimizers: `step()` to update parameters from accumulated gradients, `zero_grad()` to clear gradients, `lr()`/`set_lr()` for learning rate access, and `state_dict()`/`load_state_dict()` for checkpoint compatibility. All parameter updates must execute inside `no_grad()` to prevent the autograd engine from tracking optimizer arithmetic.
- REQ-2: Six optimizer implementations must be provided — SGD (with optional momentum, Nesterov acceleration, and weight decay), Adam (with bias correction and epsilon), AdamW (decoupled weight decay), RMSprop (with optional momentum and centered mode), Adagrad (with initial accumulator value), and L-BFGS (limited-memory quasi-Newton with line search). Each optimizer must store per-parameter state (momentum buffers, first/second moment estimates, step counts) keyed by a stable parameter identity derived from `Arc` pointer identity.
- REQ-3: Optimizers must accept parameter groups with per-group hyperparameters (learning rate, weight decay, momentum, and any optimizer-specific hyperparameters such as betas and epsilon for Adam). A single optimizer instance must be able to apply different learning rates to different layers (e.g., lower learning rate for pretrained backbone, higher for new head). An `add_param_group(&mut self, group: ParamGroup<T>)` method on the `Optimizer` trait must allow adding new parameter groups after construction. Per-group hyperparameters override the optimizer's defaults -- any hyperparameter not explicitly set in the group falls back to the default provided at construction time.
- REQ-4: LR schedulers must compose via a `SequentialLr` combinator that chains multiple schedulers with milestone step counts, following PyTorch's `SequentialLR` pattern. Five atomic schedulers must be provided -- StepLR (multiplicative decay every N epochs), CosineAnnealingLR (cosine decay to a minimum LR), OneCycleLR (warmup to peak then anneal to minimum over a fixed step budget), ReduceLROnPlateau (reduce LR when a monitored metric stalls for a patience window), and LinearWarmup (linear ramp from zero to base LR over N steps). `CosineWarmupScheduler` must NOT be a monolithic struct; it must be a convenience constructor that returns `SequentialLr` combining `LinearWarmup` + `CosineAnnealingLR`. Users must be able to compose arbitrary scheduler sequences the same way (e.g., `LinearWarmup` for 1000 steps then `StepLR` for the rest). `SequentialLr` must accept `Vec<(Box<dyn LrScheduler<T>>, usize)>` where the `usize` is the milestone step at which to switch to the next scheduler. All schedulers must mutate the optimizer's learning rate via `set_lr()` and must operate on per-group learning rates when the optimizer has multiple parameter groups.
- REQ-5: A `GradScaler<T: Float>` must be provided for mixed-precision training (fp16/bf16). The scaler multiplies the loss by a scale factor before `backward()` to prevent gradient underflow, then unscales gradients (divides by the same factor) before `optimizer.step()`. If any gradient contains inf/NaN after unscaling, the optimizer step is skipped and the scale factor is reduced. The scaler must maintain dynamic loss scaling: it increases the scale factor when gradients are healthy for a configurable number of consecutive steps (`growth_interval`, default 2000) and decreases it on inf/NaN detection. `GradScaler` must provide `scale(loss) -> Tensor<T>`, `unscale_(&self, optimizer: &mut dyn Optimizer<T>)`, `step(optimizer: &mut dyn Optimizer<T>)` (which calls unscale then conditionally steps), and `update()` (adjusts the scale factor). Scaler state must be included in training checkpoints for exact resume.
- REQ-6: SafeTensors must be the primary serialization format for model state dicts. `save_state_dict` and `load_state_dict` must default to SafeTensors. The format must follow the HuggingFace specification exactly. Loading must support memory-mapped I/O via `mmap` so that models larger than available RAM can be loaded lazily. Files produced by ferrotorch must be loadable by Python's `safetensors` library and vice versa. Msgpack is available as a secondary/internal format via explicit `save_state_dict_msgpack` / `load_state_dict_msgpack` functions, used primarily for optimizer state and checkpoint metadata where SafeTensors' tensor-only format is insufficient.
- REQ-7: `optimizer.state_dict()` and `optimizer.load_state_dict()` must preserve all internal optimizer state required for exact training resume: momentum buffers, first and second moment estimates, step counts per parameter, adaptive learning rate accumulators (Adagrad), max second moment trackers (AMSGrad), L-BFGS history vectors, and per-group hyperparameters. After `save_checkpoint` followed by `load_checkpoint`, continuing training for N steps must produce bit-identical parameter values compared to an uninterrupted run (for deterministic workloads). The serialized format must use string-keyed parameter names (resolved via `named_parameters()`) so that checkpoints are portable across program restarts where pointer addresses change.
- REQ-8: ferrotorch-serialize must support loading PyTorch `.pt` / `.pth` checkpoint files, at minimum the `state_dict` portion. This requires reading Python pickle format (the `pickle-rs` or `repugnant-pickle` crate) and converting PyTorch's tensor storage layout (storage object + offset + stride + shape) into ferrotorch `Tensor<T>`. Supported dtypes: float32, float64, float16, bfloat16, int32, int64, int8, uint8. `torch.save(model.state_dict(), path)` files must be loadable via `load_pytorch_state_dict(path) -> FerrotorchResult<StateDict<T>>`. This is critical for adoption -- users must be able to load pretrained PyTorch weights without round-tripping through Python.
- REQ-9: ferrotorch-serialize must support ONNX export by tracing a model's forward pass on example inputs and emitting a valid ONNX protobuf graph. The exported graph must target ONNX opset version 17 or higher to ensure coverage of modern operators (LayerNormalization, GroupNormalization, etc.). The opset version must be configurable via `OnnxExportConfig` with a minimum floor of opset 17. The exported graph must be loadable by the ONNX runtime Python package for inference verification.
- REQ-10: ferrotorch-serialize must support full training checkpoints that bundle: model state dict, optimizer state dict (including all internal state per REQ-7), GradScaler state (current scale factor, growth tracker, growth interval), current epoch number, current global step count, and RNG state (both the Rust thread-local RNG seed and, if applicable, the CUDA RNG state). Resuming from a checkpoint must produce a training trajectory identical to uninterrupted training for deterministic workloads.
- REQ-11: All public functions in both crates must return `Result<T, FerrotorchError>`. Deserialization of corrupted files, shape mismatches during `load_state_dict()`, missing keys, unsupported pickle opcodes in PyTorch files, and ONNX opset version violations must produce descriptive errors -- never panics.

## Acceptance Criteria
- [ ] AC-1: `Sgd::new(model.parameters(), SgdConfig { lr: 0.01, momentum: 0.9, nesterov: true, weight_decay: 1e-4 })` constructs an optimizer. Calling `optimizer.step()` after `loss.backward()` updates parameter values. Verified by training a 2-layer MLP on XOR for 500 steps and reaching loss < 0.01 with SGD (momentum=0.9).
- [ ] AC-2: `Adam`, `AdamW`, `RMSprop`, `Adagrad`, and `Lbfgs` each converge on Rosenbrock's function `f(x,y) = (1-x)^2 + 100(y-x^2)^2` starting from `(-1.0, 1.0)` to within `||[x,y] - [1,1]|| < 0.1` in at most 10000 steps (Adam: lr=0.001; AdamW: lr=0.001, weight_decay=0.01; RMSprop: lr=0.001; Adagrad: lr=0.1; L-BFGS: lr=1.0, max line search steps=20).
- [ ] AC-3: Parameter groups work: an optimizer constructed with two groups (`[{params: backbone, lr: 1e-4}, {params: head, lr: 1e-2}]`) applies different learning rates to each group. After one `step()`, backbone parameters move by ~100x less than head parameters (for identical gradient magnitudes). Additionally, `optimizer.add_param_group(ParamGroup { params: new_params, lr: 5e-3, .. })` appends a third group mid-training, and subsequent `step()` calls update the new group's parameters at the specified learning rate.
- [ ] AC-4: `StepLR`, `CosineAnnealingLR`, `OneCycleLR`, `ReduceLROnPlateau`, and `LinearWarmup` each produce the correct LR schedule. Verified by stepping each scheduler 100 times and asserting the LR at steps 0, 25, 50, 75, 99 matches the analytically computed values within `atol=1e-7`.
- [ ] AC-5: `SequentialLr` correctly composes schedulers: `cosine_warmup_scheduler(base_lr, warmup_steps, total_steps, min_lr)` returns a `SequentialLr` combining `LinearWarmup` + `CosineAnnealingLR`. The LR curve matches: linear ramp from 0 to `base_lr` over `warmup_steps`, then cosine decay to `min_lr` over the remaining steps. A custom `SequentialLr` combining `LinearWarmup(500 steps)` + `StepLR(gamma=0.5, step_size=100)` produces the expected piecewise schedule.
- [ ] AC-6: `GradScaler` works end-to-end: `scaler.scale(loss)` returns `loss * scale_factor`. After `backward()`, `scaler.step(&mut optimizer)` unscales gradients and calls `optimizer.step()`. When gradients contain inf (simulated by manually setting a gradient to `f32::INFINITY`), the optimizer step is skipped, and after `scaler.update()` the scale factor is halved. After `growth_interval` consecutive healthy steps, the scale factor doubles.
- [ ] AC-7: `save_state_dict(&state_dict, path)` followed by `load_state_dict(path)` uses SafeTensors format by default and produces bit-identical tensor data for all parameters. Tested with a model containing f32 and f64 parameters, including tensors with shapes `[]` (scalar), `[1]`, `[768, 3072]`, and `[3, 224, 224]`. The file is also loadable by Python `safetensors.safe_open()`, and files written by Python `safetensors.save_file()` are loadable by `load_state_dict(path)`. Both directions produce matching tensor data (verified via an integration test that shells out to Python).
- [ ] AC-8: Optimizer state round-trip: train Adam for 50 steps, call `optimizer.state_dict()`, serialize to checkpoint, deserialize, call `optimizer.load_state_dict()`, then continue training for 50 more steps. Final parameters must be bit-identical to an uninterrupted 100-step run. The restored state must contain: step count per parameter, exp_avg (first moment), exp_avg_sq (second moment), and max_exp_avg_sq (if amsgrad=true). Same test for SGD (momentum buffer) and Adagrad (sum accumulator).
- [ ] AC-9: `load_pytorch_state_dict("model.pt")` successfully loads a PyTorch state dict saved via `torch.save(model.state_dict(), "model.pt")`. Tensor values match within `atol=0` (bit-identical after endianness normalization). Tested with a ResNet-18 state dict containing Conv2d, BatchNorm, and Linear layer weights in float32. Unsupported pickle opcodes produce `Err(FerrotorchError::Deserialization { .. })`, not panics.
- [ ] AC-10: `export_onnx(model, example_inputs, path, config)` with `config.opset_version = 17` produces a valid `.onnx` file whose `opset_import` declares version 17. The file loads in `onnxruntime` and produces outputs matching the Rust model's forward pass within `rtol=1e-5` for f32. Tested with a model containing Linear, ReLU, LayerNorm, and Softmax layers. Attempting `opset_version = 12` returns `Err(FerrotorchError::InvalidArgument { .. })` because it is below the minimum floor.
- [ ] AC-11: `save_checkpoint(path, model, optimizer, scaler, epoch, step, rng_state)` followed by `load_checkpoint(path)` restores all components including GradScaler state. A training run that saves a checkpoint at epoch 5 and resumes from it produces the same parameter values at epoch 10 as an uninterrupted 10-epoch run (for a deterministic model with fixed seed, no dropout). The GradScaler's scale factor, growth tracker, and growth interval are restored exactly.
- [ ] AC-12: `load_state_dict` with a missing key returns `Err(FerrotorchError::MissingKey { key })`. `load_state_dict` with a shape mismatch returns `Err(FerrotorchError::ShapeMismatch { .. })`. Deserializing a truncated file returns `Err(FerrotorchError::Deserialization { .. })`. `load_pytorch_state_dict` on a non-pickle file returns `Err(FerrotorchError::Deserialization { .. })`. No panics in any error case.
- [ ] AC-13: `cargo test -p ferrotorch-optim -p ferrotorch-serialize` passes with 0 failures. Minimum 200 tests across both crates covering all optimizers, all schedulers, scheduler composition, GradScaler, all serialization formats (SafeTensors, msgpack, PyTorch import, ONNX), optimizer state round-trip, error paths, and checkpoint resume correctness.

## Architecture

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
│   ├── grad_scaler.rs            # GradScaler for mixed-precision training
│   └── scheduler/
│       ├── mod.rs                # LrScheduler<T> trait, SequentialLr combinator
│       ├── step.rs               # StepLR
│       ├── cosine.rs             # CosineAnnealingLR
│       ├── one_cycle.rs          # OneCycleLR
│       ├── plateau.rs            # ReduceLROnPlateau
│       └── warmup.rs             # LinearWarmup
└── tests/
    ├── test_sgd.rs               # Convergence, momentum, Nesterov, weight decay
    ├── test_adam.rs               # Convergence, bias correction, epsilon
    ├── test_adamw.rs              # Decoupled weight decay vs L2 reg
    ├── test_rmsprop.rs            # Convergence, centered mode
    ├── test_adagrad.rs            # Convergence, accumulator behavior
    ├── test_lbfgs.rs              # Rosenbrock, line search termination
    ├── test_param_groups.rs       # Per-group hyperparameters, add_param_group
    ├── test_schedulers.rs         # LR values at specific steps for all schedulers
    ├── test_sequential_lr.rs      # Scheduler composition, cosine_warmup_scheduler
    ├── test_grad_scaler.rs        # Scale/unscale, inf skip, dynamic scaling
    └── test_state_dict.rs         # Optimizer state round-trip (all internal state)

ferrotorch-serialize/
├── Cargo.toml
├── src/
│   ├── lib.rs                    # Public re-exports
│   ├── state_dict.rs             # StateDict<T> type, save/load (SafeTensors default)
│   ├── safetensors.rs            # SafeTensors read/write with mmap support (primary)
│   ├── msgpack.rs                # Msgpack format for optimizer state/metadata (secondary)
│   ├── pytorch.rs                # PyTorch .pt/.pth pickle-based checkpoint import
│   ├── onnx.rs                   # ONNX protobuf export via forward-pass tracing
│   ├── checkpoint.rs             # TrainingCheckpoint: model + optimizer + scaler + epoch + RNG
│   └── error.rs                  # Serialization-specific error variants
└── tests/
    ├── test_state_dict.rs        # Round-trip, missing keys, shape mismatch, corruption
    ├── test_safetensors.rs       # Format compliance, mmap, Python interop
    ├── test_msgpack.rs           # Msgpack round-trip for optimizer state
    ├── test_pytorch.rs           # PyTorch .pt/.pth import, dtype coverage, error paths
    ├── test_onnx.rs              # Export validity, opset 17+, onnxruntime inference match
    └── test_checkpoint.rs        # Save/resume determinism, GradScaler state
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

    /// Add a new parameter group with its own hyperparameters.
    /// Hyperparameters not set in the group's `options` map fall back to the
    /// optimizer's defaults provided at construction time.
    fn add_param_group(&mut self, group: ParamGroup<T>);

    /// Export optimizer state for checkpointing. Must capture ALL internal state
    /// needed for exact training resume: momentum buffers, moment estimates,
    /// step counts, adaptive accumulators, and per-group hyperparameters.
    fn state_dict(&self) -> OptimizerStateDict<T>;

    /// Restore optimizer state from a checkpoint. After load_state_dict(),
    /// continuing training must produce bit-identical results to an uninterrupted
    /// run (for deterministic workloads).
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

**SequentialLr** (`scheduler/mod.rs`):
```rust
/// Composes multiple schedulers that activate at successive milestone steps.
/// Follows PyTorch's SequentialLR pattern.
pub struct SequentialLr<T: Float> {
    /// (scheduler, milestone_step) pairs. The scheduler at index i is active for
    /// steps in [milestones[i-1], milestones[i]). The first scheduler starts at step 0.
    schedulers: Vec<(Box<dyn LrScheduler<T>>, usize)>,
    current_step: usize,
}

impl<T: Float> SequentialLr<T> {
    pub fn new(schedulers: Vec<(Box<dyn LrScheduler<T>>, usize)>) -> Self;
}

/// Convenience constructor: LinearWarmup for warmup_steps, then CosineAnnealingLR
/// for the remaining steps. Returns a SequentialLr, NOT a monolithic struct.
pub fn cosine_warmup_scheduler<T: Float>(
    base_lr: T,
    warmup_steps: usize,
    total_steps: usize,
    min_lr: T,
) -> SequentialLr<T>;
```

**GradScaler** (`grad_scaler.rs`):
```rust
/// Dynamic loss scaler for mixed-precision (fp16/bf16) training.
/// Scales loss before backward to prevent gradient underflow, unscales
/// gradients before optimizer step, and skips updates on inf/NaN.
pub struct GradScaler<T: Float> {
    scale_factor: T,
    growth_factor: T,          // default 2.0
    backoff_factor: T,         // default 0.5
    growth_interval: usize,    // default 2000 (consecutive healthy steps before scale-up)
    growth_tracker: usize,     // consecutive steps without inf/NaN
    enabled: bool,             // allows disabling without changing call sites
}

pub struct GradScalerConfig<T: Float> {
    pub init_scale: T,         // default 2^16 = 65536.0
    pub growth_factor: T,      // default 2.0
    pub backoff_factor: T,     // default 0.5
    pub growth_interval: usize, // default 2000
    pub enabled: bool,          // default true
}

/// Serializable scaler state for inclusion in training checkpoints.
pub struct GradScalerState<T: Float> {
    pub scale_factor: T,
    pub growth_tracker: usize,
    pub growth_interval: usize,
}

impl<T: Float> GradScaler<T> {
    pub fn new(config: GradScalerConfig<T>) -> Self;

    /// Multiply loss by the current scale factor.
    pub fn scale(&self, loss: &Tensor<T>) -> Tensor<T>;

    /// Divide gradients on all optimizer parameters by the scale factor.
    /// Marks whether any inf/NaN was found (checked by step()).
    pub fn unscale_(&mut self, optimizer: &mut dyn Optimizer<T>);

    /// Unscale gradients (if not already done), then call optimizer.step()
    /// only if no inf/NaN was found. Returns whether the step was performed.
    pub fn step(&mut self, optimizer: &mut dyn Optimizer<T>) -> FerrotorchResult<bool>;

    /// Adjust the scale factor: increase if healthy for growth_interval steps,
    /// decrease if inf/NaN was detected this round.
    pub fn update(&mut self);

    /// Export state for checkpointing.
    pub fn state_dict(&self) -> GradScalerState<T>;

    /// Restore state from a checkpoint.
    pub fn load_state_dict(&mut self, state: &GradScalerState<T>);
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

/// Save a state dict in SafeTensors format (primary, HuggingFace-compatible).
pub fn save_state_dict<T: Float>(
    state_dict: &StateDict<T>,
    path: impl AsRef<Path>,
) -> FerrotorchResult<()>;

/// Load a state dict from SafeTensors format (primary).
/// Memory-maps the file by default for efficient loading of large models.
pub fn load_state_dict<T: Float>(
    path: impl AsRef<Path>,
) -> FerrotorchResult<StateDict<T>>;

/// Save a state dict in msgpack format (secondary/internal, used for
/// optimizer state and checkpoint metadata where SafeTensors' tensor-only
/// format is insufficient).
pub fn save_state_dict_msgpack<T: Float>(
    state_dict: &StateDict<T>,
    path: impl AsRef<Path>,
) -> FerrotorchResult<()>;

/// Load a state dict from msgpack format (secondary/internal).
pub fn load_state_dict_msgpack<T: Float>(
    path: impl AsRef<Path>,
) -> FerrotorchResult<StateDict<T>>;
```

**SafeTensors** (`safetensors.rs`):
```rust
use ferrotorch_core::{Float, FerrotorchResult, Tensor};
use std::collections::BTreeMap;
use std::path::Path;

/// Save tensors in SafeTensors format (HuggingFace specification).
/// This is the primary serialization format — used by save_state_dict().
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

**PyTorch Import** (`pytorch.rs`):
```rust
use ferrotorch_core::{Float, FerrotorchResult};
use crate::StateDict;
use std::path::Path;

/// Load a PyTorch state dict from a .pt or .pth file.
///
/// Parses the Python pickle format used by torch.save() and converts
/// PyTorch tensor storage objects (storage + offset + stride + shape)
/// into ferrotorch Tensor<T>. Supports float32, float64, float16,
/// bfloat16, int32, int64, int8, and uint8 dtypes.
///
/// Only the state_dict portion is loaded — optimizer state, Python
/// objects, and custom pickled classes are not supported and will be
/// skipped or produce an error depending on context.
pub fn load_pytorch_state_dict<T: Float>(
    path: impl AsRef<Path>,
) -> FerrotorchResult<StateDict<T>>;

/// Configuration for PyTorch import behavior.
pub struct PyTorchLoadConfig {
    /// If true, skip keys with unsupported dtypes instead of erroring.
    pub skip_unsupported_dtypes: bool,   // default false
    /// If true, allow loading from zip-archived .pt files (torch.save format).
    pub allow_zip: bool,                  // default true
}

pub fn load_pytorch_state_dict_with_config<T: Float>(
    path: impl AsRef<Path>,
    config: &PyTorchLoadConfig,
) -> FerrotorchResult<StateDict<T>>;
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
    /// ONNX opset version. Default 17. Minimum 17 (enforced — values below 17
    /// cause export_onnx to return Err(InvalidArgument)). Opset 17 provides
    /// LayerNormalization, GroupNormalization, and other modern ops required
    /// for transformer architectures.
    pub opset_version: u32,                        // default 17, minimum 17
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
    pub scaler_state: Option<GradScalerState<T>>,
    pub epoch: usize,
    pub global_step: u64,
    pub rng_state: RngState,
}

/// Captured RNG state for deterministic resumption.
pub struct RngState {
    pub cpu_seed: u64,
    pub cuda_seeds: Vec<u64>,  // per-device CUDA RNG seeds (empty if CPU-only)
}

/// Save a training checkpoint to disk. Model weights are stored in SafeTensors
/// format; optimizer state, scaler state, and metadata use msgpack framing.
pub fn save_checkpoint<T: Float>(
    path: impl AsRef<Path>,
    model_state: &StateDict<T>,
    optimizer_state: &OptimizerStateDict<T>,
    scaler_state: Option<&GradScalerState<T>>,
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

**SafeTensors (primary format for model weights)**: Follows the HuggingFace specification exactly -- an 8-byte little-endian header size, a JSON header containing tensor metadata (name, dtype string, shape, byte offsets), followed by concatenated raw tensor data. The `safetensors` crate (version 0.4) is used as the implementation, with ferrotorch providing a thin wrapper that converts between `Tensor<T>` and the crate's `TensorView` type. This is the default format for `save_state_dict` / `load_state_dict` because it is the HuggingFace ecosystem standard -- users expect `.safetensors` files.

**Msgpack (secondary format for optimizer state and metadata)**: Tensor data is serialized as a header (name, dtype tag, shape as `Vec<usize>`) followed by raw bytes in little-endian format. rmp-serde handles the framing; the tensor bytes are inlined as msgpack binary blobs. Used for optimizer state dicts (which contain non-tensor data like step counts and hyperparameters that SafeTensors cannot represent), GradScaler state, and checkpoint metadata. Available via explicit `save_state_dict_msgpack` / `load_state_dict_msgpack` functions.

**PyTorch import**: PyTorch `.pt` / `.pth` files use Python's pickle protocol with a zip archive wrapper. The `repugnant-pickle` crate (or equivalent) parses the pickle bytecode to extract tensor storage references, which are then read from the zip archive's `data/` entries. Each PyTorch tensor is represented as (storage_type, storage_key, device, num_elements) + (storage_offset, size, stride). Ferrotorch reads the raw bytes from the storage entry, applies the offset and stride to produce a contiguous tensor, and wraps it in a `Tensor<T>`. Only `state_dict` loading is supported -- arbitrary Python objects in pickle are rejected with a descriptive error.

**ONNX**: The model's forward pass is traced by running it once with gradient tracking enabled but without calling backward. The resulting computation graph nodes are mapped to ONNX operator types. This reuses the existing autograd graph structure -- each `GradFn` has a `name()` that maps to an ONNX operator (e.g., "MatmulForward" -> "MatMul", "ReluForward" -> "Relu"). Weights are embedded as ONNX initializers. The protobuf is constructed using the `prost` crate with ONNX proto definitions. The exported graph targets opset version 17+ by default, which provides native support for LayerNormalization (op 17), GroupNormalization (op 18 if requested), and other modern operators needed for transformer architectures.

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
| `ferrotorch-optim` | workspace | OptimizerStateDict, GradScalerState (for training checkpoints) |
| `serde` | 1.0 | Serialization framework |
| `safetensors` | 0.4 | HuggingFace SafeTensors format (primary model format) |
| `rmp-serde` | 1.3 | Msgpack encoding/decoding for optimizer state and checkpoint metadata |
| `repugnant-pickle` | 0.1 | Python pickle parsing for PyTorch .pt/.pth import |
| `zip` | 2.0 | Zip archive reading for PyTorch checkpoint files |
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

#[error("ONNX opset version {requested} is below the minimum supported version {minimum}")]
OnnxOpsetTooLow { requested: u32, minimum: u32 },

#[error("PyTorch import failed: unsupported pickle opcode {opcode:#x}")]
UnsupportedPickleOpcode { opcode: u8 },

#[error("PyTorch import failed: unsupported dtype {dtype}")]
UnsupportedPyTorchDtype { dtype: String },

#[error("I/O error: {0}")]
Io(#[from] std::io::Error),
```

### Test Strategy

1. **Optimizer convergence**: Each optimizer is tested on at least two functions -- a simple quadratic `f(x) = x^2` (verifies basic gradient descent) and Rosenbrock (verifies handling of ill-conditioned curvature). Convergence is asserted numerically, not just "loss decreased."
2. **PyTorch reference**: For SGD and Adam, run identical training loops in PyTorch and ferrotorch on a 2-layer MLP with fixed seeds. Assert parameter values match within `atol=1e-5` after 100 steps. This catches subtle bugs in bias correction, weight decay application order, and momentum update formulas.
3. **Optimizer state round-trip**: For each optimizer, train for N steps, save `state_dict()`, restore via `load_state_dict()`, continue for M steps. Assert final parameters are bit-identical to an uninterrupted N+M step run. Verify all internal buffers are present: momentum buffers (SGD), exp_avg/exp_avg_sq/max_exp_avg_sq (Adam/AdamW), sum accumulators (Adagrad), centered mean square (RMSprop), history vectors (L-BFGS).
4. **Parameter group dynamics**: Test `add_param_group()` mid-training. Verify per-group hyperparameter overrides (lr, weight_decay, betas, eps). Verify that groups added after construction participate in subsequent `step()` calls with their own hyperparameters.
5. **Scheduler composition**: Test `SequentialLr` with two and three sub-schedulers. Verify `cosine_warmup_scheduler` convenience constructor produces identical LR curves to a manually constructed `SequentialLr(LinearWarmup, CosineAnnealingLR)`. Test milestone transitions produce no LR discontinuities beyond what the schedule defines.
6. **GradScaler**: Test scale/unscale/step/update cycle. Test inf detection skips optimizer step. Test scale factor growth after `growth_interval` healthy steps. Test scale factor reduction on inf/NaN. Test disabled scaler is a no-op passthrough. Test scaler state round-trip through checkpoint.
7. **Scheduler correctness**: Each atomic scheduler is tested by computing LR at every step for a known configuration and comparing against analytically derived values. No numerical optimization needed -- these are closed-form formulas.
8. **Serialization round-trip**: Save then load, assert bit-identical for SafeTensors (primary) and msgpack (secondary). For ONNX, assert inference output matches within floating-point tolerance.
9. **Cross-format interop**: SafeTensors files are validated against Python's safetensors library. ONNX files are validated against onnxruntime. PyTorch `.pt` files generated by Python are loaded and verified against the original tensor values. These are integration tests gated behind a `python-interop` feature flag.
10. **PyTorch import**: Load real-world PyTorch checkpoints (ResNet-18, BERT-base state dicts). Verify tensor shapes, dtypes, and values match. Test both zip-archived and legacy format. Test graceful errors on unsupported pickle opcodes and dtypes.
11. **Checkpoint resume**: Train for N epochs, save checkpoint (including optimizer state and GradScaler state), resume, train for M more epochs. Compare final parameters against an uninterrupted N+M epoch run. Must be bit-identical for deterministic models.
12. **Error paths**: Corrupt files, missing keys, shape mismatches, type mismatches, non-pickle files passed to `load_pytorch_state_dict`, opset versions below 17, and truncated SafeTensors files must all produce `Err`, never panic.

## Resolved Questions

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

### Q5: LR scheduler composability -- wrapping vs chaining vs sequential
**Decision**: Sequential composition via `SequentialLr`, following PyTorch's `SequentialLR` pattern.

A `SequentialLr` combinator chains multiple schedulers with milestone step counts. Each sub-scheduler is active for a range of steps defined by the milestone boundaries. `CosineWarmupScheduler` is NOT a monolithic struct -- it is a convenience constructor that returns `SequentialLr` combining `LinearWarmup` + `CosineAnnealingLR`. This avoids duplicating schedule logic and allows users to compose arbitrary scheduler sequences the same way (e.g., `LinearWarmup` then `StepLR`, or `LinearWarmup` then `CosineAnnealingLR` then constant). Users who need fully custom schedules can also implement `LrScheduler<T>` directly -- the trait has only three methods.

## Out of Scope
- Gradient clipping -- that is a utility on `Tensor`/`Module`, not an optimizer concern; it belongs in ferrotorch-nn or ferrotorch-core
- Mixed-precision optimizer state (storing moments in fp16) -- future optimization after GPU backend
- Distributed optimizer (ZeroRedundancyOptimizer) -- that is Phase 7 (ferrotorch-distributed)
- Model compilation/optimization before ONNX export -- that is Phase 8 (ferrotorch-jit)
- Custom serialization formats beyond SafeTensors/msgpack/ONNX/PyTorch-import
- Python bindings for save/load -- that is a late phase (ferrotorch-python)
- Full PyTorch checkpoint loading (optimizer state, arbitrary Python objects) -- only state_dict tensor loading is supported
- Optimizer fusion (fusing parameter updates into a single kernel) -- GPU optimization concern
- ONNX import / loading ONNX models for inference -- only export is in scope
