# Feature: Phase 2 — Neural Network Modules (ferrotorch-nn)

## Summary
The neural network module crate for ferrotorch: a `Module<T>` trait with `Parameter<T>` registration, pre-built layers (Linear, Conv2d, normalization, attention, RNN), loss functions, weight initialization, and the layer-level `GradFn<T>` implementations deferred from Phase 1. Every layer composes over `Tensor<T>` and `GradFn<T>` from ferrotorch-core. Downstream crates (optim, data, vision) depend on this for model construction and training.

## Requirements
- REQ-1: A `Module<T: Float>` trait must define the contract for all neural network layers, with methods for `forward()`, `parameters()`, `parameters_mut()`, `train()`, `eval()`, `named_parameters()`, `state_dict()`, and `load_state_dict()`. The trait must be object-safe enough for `Sequential` to hold `Vec<Box<dyn Module<T>>>`. The trait must require `Send + Sync` to match `Tensor<T>`'s thread-safety guarantees. A `#[derive(Module)]` proc macro (in a companion crate `ferrotorch-nn-derive`) must auto-generate `parameters()`, `parameters_mut()`, `named_parameters()`, `train()`, `eval()`, `is_training()`, `state_dict()`, and `load_state_dict()` from struct fields. Any field of type `Parameter<T>` is registered as a parameter, any field implementing `Module<T>` is recursed into as a sub-module, and a `training: bool` field is managed automatically. The derive macro uses the field name as the parameter/sub-module name in `named_parameters()` and `state_dict()` (e.g., field `weight` becomes key `"weight"`, sub-module field `layer1` prefixes its children as `"layer1.weight"`, `"layer1.bias"`). Fields annotated with `#[module(skip)]` are excluded from parameter collection and state dict. The user must still implement `forward()` manually — the derive only handles boilerplate. Manual `impl Module<T>` remains supported for cases the derive cannot handle (e.g., conditional parameters, dynamic sub-module counts).
- REQ-2: A `Parameter<T: Float>` type must wrap `Tensor<T>` with `requires_grad` always set to `true`. Parameter creation must enforce this invariant. Parameters must be the unit of registration for optimizer consumption — `Module::parameters()` returns references to `Parameter<T>`, not raw `Tensor<T>`.
- REQ-3: Layer-level `GradFn<T>` implementations must exist for all operations deferred from Phase 1: conv (conv1d, conv2d, conv_transpose2d), pool (max_pool2d, avg_pool2d, adaptive_avg_pool2d), norm (batch_norm, layer_norm, group_norm, rms_norm), dropout, embedding lookup, loss backward, and scaled_dot_product_attention. Each must implement the `GradFn<T>` trait from ferrotorch-core (`backward()`, `inputs()`, `name()`) and produce correct vector-Jacobian products.
- REQ-4: The following layers must be implemented as `Module<T>`: `Linear`, `Conv1d`, `Conv2d`, `ConvTranspose2d`, `BatchNorm1d`, `BatchNorm2d`, `LayerNorm`, `GroupNorm`, `RMSNorm`, `Dropout`, `Dropout2d`, `Embedding`, `MultiheadAttention`, `LSTM`, `MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d`. Each must register its learnable tensors as `Parameter<T>` and delegate forward computation to a functional operation that records the appropriate `GradFn<T>` on the output `Tensor<T>` via `Tensor::from_operation()`.
- REQ-5: Activation wrapper modules (`ReLU`, `GELU`, `SiLU`, `Sigmoid`, `Tanh`, `Softmax`, `LeakyReLU`, `ELU`, `Mish`) must implement `Module<T>` with zero parameters, delegating to the activation `GradFn<T>` implementations already in ferrotorch-core's `grad_fns::activation`.
- REQ-6: Container modules `Sequential`, `ModuleList`, and `ModuleDict` must implement `Module<T>` and propagate `parameters()`, `train()`, `eval()`, `named_parameters()`, `state_dict()`, and `load_state_dict()` to all contained sub-modules. `Sequential::forward()` must chain sub-module forward calls in order. `ModuleList` must support indexed access but not implement `forward()` itself. `ModuleDict` must store sub-modules by string key (backed by `IndexMap<String, Box<dyn Module<T>>>` to preserve insertion order), support keyed access via `get()` and `get_mut()`, and prefix `named_parameters()` / `state_dict()` keys with the sub-module's string key (e.g., key `"encoder"` with a `Linear` produces `"encoder.weight"`, `"encoder.bias"`). `ModuleDict` must not implement `forward()` — iteration order is user-controlled.
- REQ-7: Loss functions must be implemented as structs (not modules): `CrossEntropyLoss`, `MSELoss`, `BCEWithLogitsLoss`, `HuberLoss`, `CTCLoss`, `TripletMarginLoss`. Each must accept `Tensor<T>` inputs (predictions and targets), return a scalar `Tensor<T>` with a `GradFn<T>` attached, and support configurable reduction (`Mean`, `Sum`, `None`). `CrossEntropyLoss` must support label smoothing. All loss computations must be numerically stable (log-sum-exp for cross-entropy, clamping for BCE).
- REQ-8: Weight initialization functions must operate on `Parameter<T>` in-place: `xavier_uniform`, `xavier_normal`, `kaiming_uniform`, `kaiming_normal`, `uniform`, `normal`, `zeros`, `ones`, `constant`. Kaiming variants must accept a `NonLinearity` enum to select the correct gain. Each layer's `new()` constructor must apply the appropriate default initialization (Kaiming uniform for Linear weight, zeros for bias, etc.) matching PyTorch defaults.
- REQ-9: Layers with train/eval behavioral differences (`BatchNorm1d`, `BatchNorm2d`, `Dropout`, `Dropout2d`) must respect the `training` flag set by `Module::train()` / `Module::eval()`. BatchNorm must use batch statistics during training and running statistics during eval. Dropout must apply the random mask during training and be a no-op during eval.
- REQ-10: All public functions must return `FerrotorchResult<T>`. Invalid configurations (zero `in_features`, negative dropout probability, kernel size larger than input, embedding index out of vocabulary, incompatible state dict keys) must produce descriptive `FerrotorchError` variants, never panics.
- REQ-11: `Conv2d` forward must use the im2col + matmul approach, and `ConvTranspose2d` must use col2im. Convolution backward (`ConvBackward`) must compute correct gradients for weight, bias, and input using the transposed convolution relationship. Padding, stride, dilation, and groups must all be supported.
- REQ-12: `MultiheadAttention` must implement scaled dot-product attention with Q/K/V linear projections and an output projection. It must support an optional attention mask (causal or arbitrary boolean) and dropout on attention weights. The `GradFn<T>` must propagate gradients through the softmax-weighted value computation and all four projections.
- REQ-13: A `ferrotorch_nn::functional` module must provide stateless versions of all layer operations: `functional::linear()`, `functional::conv1d()`, `functional::conv2d()`, `functional::conv_transpose2d()`, `functional::batch_norm()`, `functional::layer_norm()`, `functional::group_norm()`, `functional::rms_norm()`, `functional::dropout()`, `functional::dropout2d()`, `functional::embedding()`, `functional::max_pool2d()`, `functional::avg_pool2d()`, `functional::adaptive_avg_pool2d()`, `functional::scaled_dot_product_attention()`, and all activation functions (`functional::relu()`, `functional::gelu()`, `functional::silu()`, `functional::sigmoid()`, `functional::tanh()`, `functional::softmax()`, `functional::leaky_relu()`, `functional::elu()`, `functional::mish()`, `functional::log_softmax()`). Each functional call takes tensor inputs (including weight/bias tensors as explicit arguments), performs the computation, and attaches the appropriate `GradFn<T>` to the output. Module structs are thin wrappers: they hold `Parameter<T>` fields and delegate to the corresponding functional call in their `forward()`. This separation allows users to write custom modules that compose functional operations without needing to define new Module structs.
- REQ-14: Pre-trained weight loading must be supported via the SafeTensors format. `ferrotorch_nn::load_safetensors<T: Float>(path: &Path) -> FerrotorchResult<StateDict<T>>` must parse a SafeTensors file and return a `StateDict<T>` that can be passed directly to `Module::load_state_dict()`. Key name mapping must support PyTorch-style state dict keys as written by HuggingFace `safetensors.torch.save_file()`. A `strict: bool` parameter on `load_state_dict()` controls whether unexpected keys are an error (strict=true, the default) or silently ignored (strict=false), matching PyTorch's behavior. Dtype conversion must be handled: if the file contains f16/bf16 weights and the module is `Module<f32>`, the tensors must be upcast automatically. This is the critical adoption path — users must be able to load HuggingFace model weights.
- REQ-15: All modules must work with `bf16` tensors (via the `Float` bound, which includes `bf16`). `BatchNorm1d`, `BatchNorm2d`, `LayerNorm`, `GroupNorm`, and `RMSNorm` must maintain their `running_mean`, `running_var`, and accumulation buffers in `f32` regardless of the input dtype, matching PyTorch's default mixed-precision behavior. When the input is `bf16`, normalization layers must upcast to `f32` for the statistical computation, apply the affine transform, and downcast the output back to `bf16`. The `Parameter<T>` for weight and bias in normalization layers follows the module's `T`, but running stats are always `f32` (stored as a separate `Tensor<f32>` field, not as `Parameter<T>`).
- REQ-16: Gradient clipping utilities must be provided: `clip_grad_norm_(parameters: &[&Parameter<T>], max_norm: f64, norm_type: f64) -> FerrotorchResult<f64>` computes the total norm of all parameter gradients, scales them in-place if the total norm exceeds `max_norm`, and returns the total norm. `clip_grad_value_(parameters: &[&Parameter<T>], clip_value: f64) -> FerrotorchResult<()>` clamps all parameter gradients element-wise to `[-clip_value, clip_value]`. Both must handle parameters with `None` gradients (skip them). These live in `ferrotorch_nn::utils`.

## Acceptance Criteria
- [ ] AC-1: `Module::parameters()` on a `Linear` with `in_features=128, out_features=64, bias=true` returns exactly 2 `Parameter<T>` references — one with shape `[64, 128]` (weight) and one with shape `[64]` (bias). Both have `requires_grad() == true`.
- [ ] AC-2: `Linear::forward()` on a `Tensor<f32>` with shape `[32, 128]` produces a `Tensor<f32>` with shape `[32, 64]` that has a `GradFn` attached. Calling `backward()` on a scalar derived from this output populates gradients on the weight and bias parameters, matching PyTorch `nn.Linear` within `rtol=1e-4, atol=1e-6`.
- [ ] AC-3: `Conv2d` with `in_channels=3, out_channels=16, kernel_size=3, padding=1, stride=1` applied to a `[1, 3, 32, 32]` input produces a `[1, 16, 32, 32]` output. The backward pass computes correct gradients for weight, bias, and input, verified against PyTorch within `rtol=1e-4, atol=1e-6` for f32.
- [ ] AC-4: `BatchNorm2d` in training mode (`Module::train()`) normalizes using batch statistics and updates running mean/variance. In eval mode (`Module::eval()`) it normalizes using the stored running statistics. A test trains for 10 batches, switches to eval, and verifies the running statistics match PyTorch.
- [ ] AC-5: `Sequential` containing `[Linear(784, 256), ReLU, Dropout(0.5), Linear(256, 10)]` chains forward calls correctly. `parameters()` returns all 4 parameters (2 weights + 2 biases). `train()` / `eval()` propagates to the Dropout sub-module. `state_dict()` returns a map with keys `"0.weight"`, `"0.bias"`, `"3.weight"`, `"3.bias"`. `ModuleDict` with keys `"encoder"` -> `Linear(784, 256)` and `"decoder"` -> `Linear(256, 784)` returns `named_parameters()` entries prefixed with `"encoder."` and `"decoder."`, preserves insertion order during iteration, and round-trips through `state_dict()` / `load_state_dict()`.
- [ ] AC-6: `CrossEntropyLoss` with label smoothing=0.1 applied to logits of shape `[32, 10]` and integer targets of shape `[32]` returns a scalar `Tensor<T>`. Backward produces gradients on the logits matching PyTorch `nn.CrossEntropyLoss(label_smoothing=0.1)` within tolerance. Numerical stability is verified: no NaN or Inf for logits in the range `[-100, 100]`.
- [ ] AC-7: Every layer-level `GradFn<T>` (ConvBackward, MaxPoolBackward, AvgPoolBackward, AdaptiveAvgPoolBackward, BatchNormBackward, LayerNormBackward, GroupNormBackward, RMSNormBackward, DropoutBackward, EmbeddingBackward, CrossEntropyBackward, MSEBackward, BCEWithLogitsBackward, HuberBackward, CTCBackward, TripletMarginBackward, AttentionBackward) passes a numerical gradient check with finite differences (`rtol=1e-4, atol=1e-6` for f32, `rtol=1e-7, atol=1e-10` for f64).
- [ ] AC-8: Weight initialization functions produce distributions matching their specifications: `xavier_uniform` fills values in `[-limit, limit]` where `limit = sqrt(6 / (fan_in + fan_out))`; `kaiming_normal` fills with `N(0, sqrt(2 / fan_in))` for ReLU. Verified statistically over 10,000 elements (mean and variance within 5% of theoretical values).
- [ ] AC-9: `MultiheadAttention` with `embed_dim=512, num_heads=8` applied to query/key/value tensors of shape `[32, 10, 512]` produces output of shape `[32, 10, 512]`. With a causal mask, attention weights are zero above the diagonal. Backward populates gradients on all 4 projection parameters (Q, K, V, output), verified against PyTorch.
- [ ] AC-10: `LSTM` with `input_size=128, hidden_size=256, num_layers=2, bidirectional=true` applied to input `[32, 20, 128]` (batch, seq, feature) produces output `[32, 20, 512]` and hidden state tuple `(h_n, c_n)` with shapes `[4, 32, 256]`. Backward through the output computes correct gradients on all weight matrices.
- [ ] AC-11: `Dropout(p=0.3)` in training mode zeros approximately 30% of elements (verified over 100,000 elements, within 2% of target rate) and scales surviving elements by `1/(1-p)`. In eval mode, the output equals the input exactly. The dropout mask `GradFn<T>` correctly routes gradients only through surviving elements.
- [ ] AC-12: `cargo test -p ferrotorch-nn` and `cargo test -p ferrotorch-nn-derive` pass with 0 failures. Minimum 300 tests across both crates, covering: all modules, all loss functions, all grad_fns, all init functions, all functional API functions, derive macro code generation, SafeTensors loading, gradient clipping, mixed-precision normalization, train/eval mode switching, state_dict round-trip (strict and non-strict), ModuleDict keyed access and iteration order, error paths (invalid shapes, out-of-bounds embedding indices, mismatched state dict keys, corrupted SafeTensors files), and edge cases (batch size 1, sequence length 1, single-channel input, zero-padding convolution, bf16 inputs).
- [ ] AC-13: `load_state_dict(state, strict=true)` on a `Linear` with a state dict containing the wrong key names returns `Err(FerrotorchError::InvalidArgument { .. })`. A state dict with correct keys but wrong tensor shapes returns `Err(FerrotorchError::ShapeMismatch { .. })`. With `strict=false`, unexpected keys are silently ignored and missing keys leave existing parameter values unchanged.
- [ ] AC-14: A `Sequential` model can be constructed, trained for one step (forward + backward), and its parameters collected — all from one thread, then sent to another thread for inference. This verifies `Module<T>: Send + Sync`.
- [ ] AC-15: `#[derive(Module)]` on a struct with two `Parameter<T>` fields (`weight`, `bias`) and one `Linear` sub-module field (`projection`) auto-generates correct `parameters()` (returns 4 params: 2 from struct + 2 from Linear), `named_parameters()` (keys `"weight"`, `"bias"`, `"projection.weight"`, `"projection.bias"`), and `train()`/`eval()` (propagates to the `projection` sub-module). A field annotated `#[module(skip)]` is excluded from all generated methods.
- [ ] AC-16: `functional::linear(input, weight, Some(bias))` produces the same output and gradients as `Linear::forward()` when given the same weight and bias tensors. The same correspondence holds for `functional::conv2d()` / `Conv2d::forward()`, `functional::batch_norm()` / `BatchNorm2d::forward()`, and all other module/functional pairs.
- [ ] AC-17: `load_safetensors::<f32>("model.safetensors")` parses a SafeTensors file containing a 2-layer MLP's weights (saved from PyTorch) and returns a `StateDict<f32>` with the correct keys and tensor shapes. Passing this state dict to a matching `Sequential`'s `load_state_dict()` loads all weights and produces forward outputs matching the original PyTorch model within tolerance. When the SafeTensors file contains `bf16` tensors and the module is `Module<f32>`, the tensors are automatically upcast to `f32`.
- [ ] AC-18: `BatchNorm2d` with `bf16` input maintains `f32` running statistics. After 10 training batches with `bf16` input, the running mean and running variance are `f32` tensors. The forward output is `bf16`. Switching to eval mode and running a `bf16` input through produces the same output (within `bf16` tolerance) as PyTorch's `BatchNorm2d` in mixed-precision mode.
- [ ] AC-19: `clip_grad_norm_` applied to a model's parameters with `max_norm=1.0` returns the original total norm and, when that norm exceeds 1.0, all parameter gradients are scaled such that their combined norm equals exactly 1.0 (within `f32` tolerance). `clip_grad_value_` with `clip_value=0.5` clamps all gradient elements to `[-0.5, 0.5]`. Both functions skip parameters with `None` gradients without error.

## Architecture

### Crate Layout

```
ferrotorch-nn/
├── Cargo.toml
├── src/
│   ├── lib.rs                    # Public re-exports
│   ├── module.rs                 # Module<T> trait, training flag
│   ├── parameter.rs              # Parameter<T> type
│   ├── linear.rs                 # Linear (fully connected)
│   ├── conv.rs                   # Conv1d, Conv2d, ConvTranspose2d
│   ├── norm.rs                   # BatchNorm1d, BatchNorm2d, LayerNorm, GroupNorm, RMSNorm
│   ├── activation.rs             # ReLU, GELU, SiLU, Sigmoid, Tanh, Softmax, LeakyReLU, ELU, Mish
│   ├── dropout.rs                # Dropout, Dropout2d
│   ├── pooling.rs                # MaxPool2d, AvgPool2d, AdaptiveAvgPool2d
│   ├── rnn.rs                    # LSTM
│   ├── attention.rs              # MultiheadAttention
│   ├── embedding.rs              # Embedding
│   ├── container.rs              # Sequential, ModuleList, ModuleDict
│   ├── init.rs                   # Weight initialization functions
│   ├── loss.rs                   # CrossEntropyLoss, MSELoss, BCEWithLogitsLoss, HuberLoss, CTCLoss, TripletMarginLoss
│   ├── functional.rs             # Stateless functional API (linear, conv2d, batch_norm, dropout, all activations, etc.)
│   ├── safetensors.rs            # SafeTensors file loading into StateDict<T>
│   ├── utils.rs                  # clip_grad_norm_, clip_grad_value_
│   └── grad_fns/                 # Layer-level GradFn<T> implementations (deferred from core)
│       ├── mod.rs
│       ├── conv.rs               # ConvBackward, ConvTransposeBackward
│       ├── pool.rs               # MaxPoolBackward, AvgPoolBackward, AdaptiveAvgPoolBackward
│       ├── norm.rs               # BatchNormBackward, LayerNormBackward, GroupNormBackward, RMSNormBackward
│       ├── dropout.rs            # DropoutBackward
│       ├── embedding.rs          # EmbeddingBackward
│       ├── loss.rs               # CrossEntropyBackward, MSEBackward, BCEWithLogitsBackward, HuberBackward, CTCBackward, TripletMarginBackward
│       └── attention.rs          # AttentionBackward
├── tests/
│   ├── test_linear.rs            # Linear forward, backward, parameter registration
│   ├── test_conv.rs              # Conv1d, Conv2d, ConvTranspose2d with various padding/stride/dilation
│   ├── test_norm.rs              # BatchNorm train/eval, LayerNorm, GroupNorm, RMSNorm, bf16 mixed precision
│   ├── test_activation.rs        # All activation modules (zero-param wrappers)
│   ├── test_dropout.rs           # Dropout rate, train/eval behavior, gradient routing
│   ├── test_pooling.rs           # MaxPool2d, AvgPool2d, AdaptiveAvgPool2d forward + backward
│   ├── test_rnn.rs               # LSTM forward, backward, bidirectional, multi-layer
│   ├── test_attention.rs         # MultiheadAttention with and without causal mask
│   ├── test_embedding.rs         # Embedding lookup, out-of-bounds error, sparse gradient
│   ├── test_container.rs         # Sequential chaining, ModuleList indexing, ModuleDict keyed access, parameter propagation
│   ├── test_init.rs              # Statistical verification of all init functions
│   ├── test_loss.rs              # All loss functions: numerical correctness, stability, reduction modes
│   ├── test_state_dict.rs        # Round-trip save/load, error on key/shape mismatch
│   ├── test_safetensors.rs       # Loading PyTorch-saved SafeTensors files, dtype conversion, key mapping
│   ├── test_functional.rs        # All functional API functions match their module counterparts
│   ├── test_derive.rs            # #[derive(Module)] code generation, #[module(skip)], nested sub-modules
│   ├── test_grad_fns.rs          # Numerical gradient checks for all layer-level GradFn<T>
│   ├── test_grad_clip.rs         # clip_grad_norm_, clip_grad_value_, edge cases
│   ├── test_mixed_precision.rs   # bf16 forward/backward, normalization f32 running stats
│   └── test_thread_safety.rs     # Module<T>: Send + Sync across threads
│
ferrotorch-nn-derive/
├── Cargo.toml
└── src/
    └── lib.rs                    # #[derive(Module)] proc macro implementation
```

### Core Types

**Module<T>** (`module.rs`):

```rust
/// State dict: a map from parameter names to tensors.
pub type StateDict<T> = std::collections::HashMap<String, Tensor<T>>;

/// The trait that all neural network layers implement.
///
/// Requires `Send + Sync` to match `Tensor<T>`'s thread-safety guarantees.
/// Object-safe for the subset needed by `Sequential`: `forward()` and
/// `parameters()` use `&self`, making `dyn Module<T>` viable.
pub trait Module<T: Float>: Send + Sync {
    /// Forward pass. Takes input tensor(s), returns output tensor(s).
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>>;

    /// Iterate over all learnable parameters.
    fn parameters(&self) -> Vec<&Parameter<T>>;

    /// Iterate over all learnable parameters mutably (for in-place init or loading).
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>>;

    /// Set training mode. Affects layers like Dropout and BatchNorm.
    fn train(&mut self);

    /// Set evaluation mode.
    fn eval(&mut self);

    /// Whether the module is in training mode.
    fn is_training(&self) -> bool;

    /// Named parameters for state dict serialization.
    /// Returns (name, parameter) pairs with dot-separated hierarchical names.
    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)>;

    /// Export all parameters as a state dict.
    fn state_dict(&self) -> StateDict<T>;

    /// Load parameters from a state dict. If `strict` is true (default),
    /// missing or unexpected keys produce an error. If `strict` is false,
    /// unexpected keys are silently ignored and missing keys leave existing
    /// parameter values unchanged. Shape mismatches always produce an error.
    fn load_state_dict(&mut self, state: &StateDict<T>, strict: bool) -> FerrotorchResult<()>;
}
```

**Parameter<T>** (`parameter.rs`):

```rust
/// A tensor registered for gradient descent.
///
/// Wraps `Tensor<T>` with the invariant that `requires_grad` is always `true`.
/// Modules store their learnable weights as `Parameter<T>` values, and
/// `Module::parameters()` returns references to them for optimizer consumption.
pub struct Parameter<T: Float = f32> {
    tensor: Tensor<T>,
}

impl<T: Float> Parameter<T> {
    /// Create a new parameter. The tensor's `requires_grad` is forced to `true`.
    pub fn new(tensor: Tensor<T>) -> Self {
        let tensor = if tensor.requires_grad() {
            tensor
        } else {
            tensor.requires_grad_(true)
        };
        Self { tensor }
    }

    /// Borrow the underlying tensor.
    pub fn tensor(&self) -> &Tensor<T> {
        &self.tensor
    }

    /// Mutable access to the underlying tensor (for in-place initialization).
    pub fn tensor_mut(&mut self) -> &mut Tensor<T> {
        &mut self.tensor
    }

    /// Delegate to `Tensor::shape()`.
    pub fn shape(&self) -> &[usize] {
        self.tensor.shape()
    }

    /// Delegate to `Tensor::grad()`.
    pub fn grad(&self) -> FerrotorchResult<Option<Tensor<T>>> {
        self.tensor.grad()
    }

    /// Zero out the accumulated gradient.
    pub fn zero_grad(&self) -> FerrotorchResult<()> {
        self.tensor.set_grad(None)
    }
}
```

### Layer-Level GradFn Implementations

These structs implement `GradFn<T>` (the trait defined in `ferrotorch_core::tensor`) and live in `src/grad_fns/`. The backward engine in ferrotorch-core calls `grad_fn.backward()` via dynamic dispatch on `Arc<dyn GradFn<T>>` — it does not need to know the concrete type. This is the same pattern used by core's arithmetic, reduction, and activation grad_fns.

| File | Structs | VJP Strategy |
|------|---------|-------------|
| `grad_fns/conv.rs` | `ConvBackward`, `ConvTransposeBackward` | Input grad via transposed convolution; weight grad via correlation of input and grad_output; bias grad via reduction over spatial dims |
| `grad_fns/pool.rs` | `MaxPoolBackward`, `AvgPoolBackward`, `AdaptiveAvgPoolBackward` | Max: route grad to argmax indices; Avg: distribute grad equally over window; Adaptive: scale by input/output ratio |
| `grad_fns/norm.rs` | `BatchNormBackward`, `LayerNormBackward`, `GroupNormBackward`, `RMSNormBackward` | Standard normalization backward: grad w.r.t. input, weight (gamma), bias (beta) using saved mean/variance |
| `grad_fns/dropout.rs` | `DropoutBackward` | Multiply grad_output by the saved binary mask and scale by `1/(1-p)` |
| `grad_fns/embedding.rs` | `EmbeddingBackward` | Scatter-add grad_output rows to the corresponding embedding weight indices |
| `grad_fns/loss.rs` | `CrossEntropyBackward`, `MSEBackward`, `BCEWithLogitsBackward`, `HuberBackward`, `CTCBackward`, `TripletMarginBackward` | Each computes the analytic gradient of its loss formula; cross-entropy uses `softmax(logits) - one_hot(targets)` for numerical stability |
| `grad_fns/attention.rs` | `AttentionBackward` | Backprop through softmax-weighted matmul: grad flows to Q, K, V via transposed attention weight computations and through all four linear projections |

### Functional API (`functional.rs`)

Stateless functions that perform the computation and attach the appropriate `GradFn<T>`. Module structs are thin wrappers that hold parameters and call these functions. This mirrors `torch.nn.functional` — every module operation has a corresponding functional form, and users can compose functional calls directly in custom `forward()` methods without defining new module structs.

```rust
// --- Linear ---
pub fn linear<T: Float>(input: &Tensor<T>, weight: &Tensor<T>, bias: Option<&Tensor<T>>) -> FerrotorchResult<Tensor<T>>;

// --- Convolution ---
pub fn conv1d<T: Float>(input: &Tensor<T>, weight: &Tensor<T>, bias: Option<&Tensor<T>>, stride: usize, padding: usize, dilation: usize, groups: usize) -> FerrotorchResult<Tensor<T>>;
pub fn conv2d<T: Float>(input: &Tensor<T>, weight: &Tensor<T>, bias: Option<&Tensor<T>>, stride: [usize; 2], padding: [usize; 2], dilation: [usize; 2], groups: usize) -> FerrotorchResult<Tensor<T>>;
pub fn conv_transpose2d<T: Float>(input: &Tensor<T>, weight: &Tensor<T>, bias: Option<&Tensor<T>>, stride: [usize; 2], padding: [usize; 2], output_padding: [usize; 2], groups: usize, dilation: [usize; 2]) -> FerrotorchResult<Tensor<T>>;

// --- Normalization ---
pub fn batch_norm<T: Float>(input: &Tensor<T>, running_mean: Option<&Tensor<T>>, running_var: Option<&Tensor<T>>, weight: Option<&Tensor<T>>, bias: Option<&Tensor<T>>, training: bool, momentum: f64, eps: f64) -> FerrotorchResult<Tensor<T>>;
pub fn layer_norm<T: Float>(input: &Tensor<T>, normalized_shape: &[usize], weight: Option<&Tensor<T>>, bias: Option<&Tensor<T>>, eps: f64) -> FerrotorchResult<Tensor<T>>;
pub fn group_norm<T: Float>(input: &Tensor<T>, num_groups: usize, weight: Option<&Tensor<T>>, bias: Option<&Tensor<T>>, eps: f64) -> FerrotorchResult<Tensor<T>>;
pub fn rms_norm<T: Float>(input: &Tensor<T>, normalized_shape: &[usize], weight: Option<&Tensor<T>>, eps: f64) -> FerrotorchResult<Tensor<T>>;

// --- Dropout ---
pub fn dropout<T: Float>(input: &Tensor<T>, p: f64, training: bool) -> FerrotorchResult<Tensor<T>>;
pub fn dropout2d<T: Float>(input: &Tensor<T>, p: f64, training: bool) -> FerrotorchResult<Tensor<T>>;

// --- Pooling ---
pub fn max_pool2d<T: Float>(input: &Tensor<T>, kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2]) -> FerrotorchResult<Tensor<T>>;
pub fn avg_pool2d<T: Float>(input: &Tensor<T>, kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2]) -> FerrotorchResult<Tensor<T>>;
pub fn adaptive_avg_pool2d<T: Float>(input: &Tensor<T>, output_size: [usize; 2]) -> FerrotorchResult<Tensor<T>>;

// --- Embedding ---
pub fn embedding<T: Float>(input: &Tensor<T>, weight: &Tensor<T>, padding_idx: Option<usize>) -> FerrotorchResult<Tensor<T>>;

// --- Attention ---
pub fn scaled_dot_product_attention<T: Float>(query: &Tensor<T>, key: &Tensor<T>, value: &Tensor<T>, attn_mask: Option<&Tensor<T>>, dropout_p: f64, training: bool) -> FerrotorchResult<Tensor<T>>;

// --- Activations ---
pub fn relu<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>>;
pub fn gelu<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>>;
pub fn silu<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>>;
pub fn sigmoid<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>>;
pub fn tanh<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>>;
pub fn softmax<T: Float>(input: &Tensor<T>, dim: i64) -> FerrotorchResult<Tensor<T>>;
pub fn log_softmax<T: Float>(input: &Tensor<T>, dim: i64) -> FerrotorchResult<Tensor<T>>;
pub fn leaky_relu<T: Float>(input: &Tensor<T>, negative_slope: f64) -> FerrotorchResult<Tensor<T>>;
pub fn elu<T: Float>(input: &Tensor<T>, alpha: f64) -> FerrotorchResult<Tensor<T>>;
pub fn mish<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>>;

// --- Loss functions (functional form) ---
pub fn cross_entropy<T: Float>(input: &Tensor<T>, target: &Tensor<T>, reduction: Reduction, label_smoothing: f64) -> FerrotorchResult<Tensor<T>>;
pub fn mse_loss<T: Float>(input: &Tensor<T>, target: &Tensor<T>, reduction: Reduction) -> FerrotorchResult<Tensor<T>>;
pub fn binary_cross_entropy_with_logits<T: Float>(input: &Tensor<T>, target: &Tensor<T>, reduction: Reduction) -> FerrotorchResult<Tensor<T>>;
pub fn huber_loss<T: Float>(input: &Tensor<T>, target: &Tensor<T>, reduction: Reduction, delta: f64) -> FerrotorchResult<Tensor<T>>;
```

### Derive Macro (`ferrotorch-nn-derive`)

A proc-macro crate providing `#[derive(Module)]`. The macro inspects struct fields at compile time and generates the `Module<T>` trait implementation (except `forward()`, which the user writes manually).

```rust
use ferrotorch_nn::Module;
use ferrotorch_nn_derive::Module;

#[derive(Module)]
struct MyModel<T: Float> {
    linear1: Linear<T>,           // Sub-module: recursed into for parameters/train/eval
    linear2: Linear<T>,           // Sub-module
    #[module(skip)]
    dropout_p: f64,               // Skipped: not a parameter or sub-module
    training: bool,               // Managed automatically by generated train()/eval()
}

impl<T: Float> MyModel<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let x = self.linear1.forward(input)?;
        let x = functional::relu(&x)?;
        let x = functional::dropout(&x, self.dropout_p, self.training)?;
        self.linear2.forward(&x)
    }
}
```

The derive macro generates:
- `parameters()` / `parameters_mut()`: collects from all `Parameter<T>` fields and recurses into all `Module<T>` fields.
- `named_parameters()`: uses field names as prefixes (e.g., `"linear1.weight"`, `"linear1.bias"`, `"linear2.weight"`, `"linear2.bias"`).
- `train()` / `eval()`: sets `self.training` and recurses into sub-module fields.
- `is_training()`: returns `self.training`.
- `state_dict()` / `load_state_dict()`: serializes/deserializes using `named_parameters()` keys.

Field detection rules:
- Type is `Parameter<T>` -> registered as a leaf parameter.
- Type implements `Module<T>` -> recursed into as a sub-module.
- `#[module(skip)]` -> excluded from everything.
- `training: bool` -> managed by `train()`/`eval()`, not a parameter.

### SafeTensors Loading (`safetensors.rs`)

```rust
/// Load a SafeTensors file into a state dict.
///
/// Parses the SafeTensors binary format (header + raw tensor data), converts
/// tensors to the target dtype `T`, and returns a `StateDict<T>` keyed by
/// the tensor names from the file (which match PyTorch state_dict keys when
/// saved via `safetensors.torch.save_file()`).
///
/// Dtype conversion: if the file contains f16 or bf16 tensors and `T` is f32,
/// the data is upcast element-wise. If `T` is bf16 and the file has f32, the
/// data is downcast. Mismatched integer dtypes produce an error.
pub fn load_safetensors<T: Float>(path: &Path) -> FerrotorchResult<StateDict<T>>;

/// Save a state dict to SafeTensors format.
pub fn save_safetensors<T: Float>(state: &StateDict<T>, path: &Path) -> FerrotorchResult<()>;
```

The `load_state_dict()` method on `Module<T>` accepts a `strict` parameter:

```rust
/// Load parameters from a state dict.
///
/// If `strict` is true (default), all keys in the state dict must match the
/// module's named parameters exactly — missing or unexpected keys produce an
/// error. If `strict` is false, unexpected keys are silently ignored and
/// missing keys leave the existing parameter values unchanged.
fn load_state_dict(&mut self, state: &StateDict<T>, strict: bool) -> FerrotorchResult<()>;
```

### Gradient Clipping (`utils.rs`)

```rust
/// Clip the total gradient norm of all parameters in-place.
///
/// Computes the total norm across all parameter gradients (using `norm_type`,
/// typically 2.0), and if it exceeds `max_norm`, scales all gradients by
/// `max_norm / total_norm`. Returns the original total norm (before clipping).
/// Parameters with `None` gradients are skipped.
pub fn clip_grad_norm_<T: Float>(
    parameters: &[&Parameter<T>],
    max_norm: f64,
    norm_type: f64,
) -> FerrotorchResult<f64>;

/// Clamp all parameter gradients element-wise to [-clip_value, clip_value].
///
/// Parameters with `None` gradients are skipped.
pub fn clip_grad_value_<T: Float>(
    parameters: &[&Parameter<T>],
    clip_value: f64,
) -> FerrotorchResult<()>;
```

### Mixed Precision for Normalization Layers

Normalization layers (`BatchNorm1d`, `BatchNorm2d`, `LayerNorm`, `GroupNorm`, `RMSNorm`) must handle mixed precision correctly:

- **Running statistics** (`running_mean`, `running_var`) are always stored as `Tensor<f32>`, regardless of the module's type parameter `T`. These are not `Parameter<T>` -- they are non-learnable buffers updated during training.
- **When `T` is `bf16`**: The forward pass upcasts the input to `f32`, computes the normalization (mean, variance, affine transform) in `f32`, and downcasts the result back to `bf16`. This avoids the catastrophic precision loss that occurs when computing variance in `bf16`.
- **Weight and bias** (`gamma`, `beta`) follow the module's `T` -- they are `Parameter<T>` and participate in gradient descent at whatever precision the module uses.
- The functional API signatures accept a `Tensor<T>` for input and `Option<&Tensor<T>>` for weight/bias, but the batch_norm function internally manages the `f32` upcast when `T` is `bf16`. This is transparent to the caller.

### Weight Initialization (`init.rs`)

```rust
/// Nonlinearity hint for Kaiming initialization gain calculation.
pub enum NonLinearity {
    Linear,
    ReLU,
    LeakyReLU(f64),
    Tanh,
    Sigmoid,
    GELU,
    SiLU,
}

pub fn xavier_uniform<T: Float>(param: &mut Parameter<T>, gain: f64) -> FerrotorchResult<()>;
pub fn xavier_normal<T: Float>(param: &mut Parameter<T>, gain: f64) -> FerrotorchResult<()>;
pub fn kaiming_uniform<T: Float>(param: &mut Parameter<T>, nonlinearity: NonLinearity) -> FerrotorchResult<()>;
pub fn kaiming_normal<T: Float>(param: &mut Parameter<T>, nonlinearity: NonLinearity) -> FerrotorchResult<()>;
pub fn uniform<T: Float>(param: &mut Parameter<T>, low: f64, high: f64) -> FerrotorchResult<()>;
pub fn normal<T: Float>(param: &mut Parameter<T>, mean: f64, std: f64) -> FerrotorchResult<()>;
pub fn zeros<T: Float>(param: &mut Parameter<T>) -> FerrotorchResult<()>;
pub fn ones<T: Float>(param: &mut Parameter<T>) -> FerrotorchResult<()>;
pub fn constant<T: Float>(param: &mut Parameter<T>, value: f64) -> FerrotorchResult<()>;
```

### Loss Functions (`loss.rs`)

```rust
/// Reduction mode for loss functions.
pub enum Reduction {
    Mean,
    Sum,
    None,
}

pub struct CrossEntropyLoss {
    reduction: Reduction,
    label_smoothing: f64,
}

pub struct MSELoss {
    reduction: Reduction,
}

pub struct BCEWithLogitsLoss {
    reduction: Reduction,
}

pub struct HuberLoss {
    reduction: Reduction,
    delta: f64,
}

pub struct CTCLoss {
    reduction: Reduction,
    blank: usize,
    zero_infinity: bool,
}

pub struct TripletMarginLoss {
    reduction: Reduction,
    margin: f64,
    p: f64,
}
```

Each loss struct implements a `forward()` method (not the `Module` trait — losses are not modules in PyTorch either) that returns a `Tensor<T>` with the appropriate `GradFn<T>` attached:

```rust
impl CrossEntropyLoss {
    pub fn forward<T: Float>(
        &self,
        input: &Tensor<T>,
        target: &Tensor<T>,
    ) -> FerrotorchResult<Tensor<T>>;
}
```

### Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `ferrotorch-core` | workspace | `Tensor<T>`, `GradFn<T>`, `Float`, `FerrotorchError`, `Tensor::from_operation()`, backward engine |
| `ferrotorch-nn-derive` | workspace | `#[derive(Module)]` proc macro |
| `ferray-core` | workspace | `Element` trait (re-exported via ferrotorch-core's `Float`) |
| `ferray-random` | workspace | Random mask generation for Dropout, random init for weight initialization |
| `safetensors` | 0.5 | Parsing SafeTensors binary format for pre-trained weight loading |
| `indexmap` | 2.7 | Insertion-ordered map for `ModuleDict` |
| `thiserror` | 2.0 | Error derive macros (re-uses `FerrotorchError` from core, may add nn-specific variants) |
| `rayon` | 1.11 | Parallel im2col/col2im for convolution, parallel LSTM gate computation |
| `syn` | 2.0 | (ferrotorch-nn-derive only) Parsing Rust source for derive macro |
| `quote` | 1.0 | (ferrotorch-nn-derive only) Code generation for derive macro |
| `proc-macro2` | 1.0 | (ferrotorch-nn-derive only) Token stream manipulation |

### Convolution Implementation Strategy

`Conv2d` uses the im2col (image-to-column) approach:

1. **Forward**: Unfold input patches into a 2D column matrix, then compute `output = weight_matrix @ col_matrix + bias`. This reduces convolution to a single matmul, delegating to ferray-linalg.
2. **Backward (ConvBackward)**:
   - `grad_input`: Transpose the weight matrix, matmul with grad_output columns, then col2im to fold back to input shape.
   - `grad_weight`: Matmul grad_output columns with im2col(input) transposed.
   - `grad_bias`: Sum grad_output over batch and spatial dimensions.
3. `groups > 1`: Partition input/output channels into groups and apply the above per group.
4. `ConvTranspose2d`: Swaps the forward/backward relationship — forward is a col2im, backward is an im2col.

### LSTM Implementation Strategy

`LSTM` follows PyTorch's implementation:

1. **Forward**: For each time step, compute all four gates (input, forget, cell, output) as a single fused matmul: `gates = W_ih @ x_t + W_hh @ h_{t-1} + bias`. Split and apply sigmoid/tanh activations. Multi-layer: feed the output of layer `l` as input to layer `l+1`. Bidirectional: run a second pass in reverse and concatenate outputs.
2. **Backward**: Backpropagate through time (BPTT). The `GradFn<T>` stores all intermediate gate activations and hidden states from the forward pass. Gradients flow backward through the time steps, accumulating on the weight matrices.

### Test Strategy

Minimum 300 tests across all test files. The nn crate has the largest surface area of any ferrotorch crate and must be exhaustively tested.

1. **Numerical gradient checks**: For every `GradFn<T>` in `grad_fns/`, compare analytic gradient against finite-difference approximation: `(f(x+h) - f(x-h)) / 2h`. Use `h=1e-4` for f32, `h=1e-7` for f64.
2. **PyTorch reference tests**: For each module, compute forward + backward in PyTorch, serialize inputs/outputs/gradients as `.npy` files (via ferray-io), and assert ferrotorch-nn matches within tolerance.
3. **Train/eval behavioral tests**: Verify Dropout and BatchNorm produce different outputs in train vs eval mode.
4. **State dict round-trip**: `state_dict()` followed by `load_state_dict()` on a fresh module produces identical parameters. Test both strict and non-strict modes.
5. **Error paths**: Invalid shapes, out-of-bounds indices, mismatched state dict keys, zero-size dimensions.
6. **Statistical init tests**: Generate large tensors, compute sample mean/variance, assert within tolerance of theoretical values.
7. **Thread safety**: Build a model on one thread, send it to another, run forward + backward.
8. **Functional/module parity**: For every module that has a functional counterpart, verify that `Module::forward()` and the functional call produce identical outputs and gradients given the same inputs and parameters.
9. **Derive macro**: Verify `#[derive(Module)]` generates correct `parameters()`, `named_parameters()`, `train()`/`eval()` propagation, `state_dict()` round-trip, and `#[module(skip)]` exclusion. Test nested sub-modules (module containing a module containing parameters).
10. **SafeTensors loading**: Load a PyTorch-saved SafeTensors file, verify keys and shapes match, verify dtype conversion (bf16->f32, f16->f32), verify error on corrupted file.
11. **Mixed precision**: Verify normalization layers maintain f32 running stats with bf16 input, verify output dtype matches input dtype, verify numerical accuracy against PyTorch mixed-precision reference.
12. **Gradient clipping**: Verify `clip_grad_norm_` scales gradients to target norm, verify `clip_grad_value_` clamps to bounds, verify both skip `None` gradients, verify edge cases (zero gradients, single parameter, inf norm).

## Resolved Questions

### Q1: Should loss functions implement the Module trait?
**Decision**: No. Loss functions are structs with a `forward()` method, not `Module` implementors.

PyTorch's `nn.CrossEntropyLoss` technically inherits from `nn.Module`, but this is a historical accident — losses have no learnable parameters, no train/eval distinction, and no state dict. Making them plain structs simplifies the API and avoids confusion about whether loss "parameters" would appear in optimizer parameter groups. The `forward()` method takes predictions and targets as arguments and returns a scalar `Tensor<T>` with a `GradFn<T>`, which integrates with the backward engine identically to module outputs.

### Q2: Where do layer-level GradFn implementations live?
**Decision**: In `ferrotorch-nn/src/grad_fns/`, implementing the `GradFn<T>` trait from ferrotorch-core.

This was decided in Phase 1 (Q3): core keeps math ops (arithmetic, reduction, linalg, activation, shape, indexing, comparison), nn gets layer ops (conv, pool, norm, dropout, embedding, loss, attention). The trait is defined in core; the struct implementations live in nn. Core's backward engine dispatches via `Arc<dyn GradFn<T>>` and never needs to know the concrete type.

### Q3: Object safety of Module trait
**Decision**: `Module<T>` is object-safe. `Sequential` stores `Vec<Box<dyn Module<T>>>`.

The `forward()`, `parameters()`, `train()`, `eval()`, `is_training()`, `named_parameters()`, `state_dict()`, and `load_state_dict()` methods all use `&self` or `&mut self` and return owned types or trait-object-safe collections (`Vec`, `HashMap`). `parameters_mut()` returns `Vec<&mut Parameter<T>>` which is object-safe. This enables dynamic composition: users can build arbitrary layer sequences without compile-time type gymnastics.

### Q4: Parameter storage — newtype vs alias
**Decision**: Newtype struct wrapping `Tensor<T>`, not a type alias.

A newtype enforces the `requires_grad = true` invariant at construction time and provides a distinct type for optimizer APIs to accept. A type alias would allow accidentally passing a non-grad tensor to an optimizer. The newtype cost is a single `.tensor()` call when raw tensor access is needed, which is acceptable.

## Out of Scope
- GPU execution of layer operations -- `Device::Cuda` is defined but only `Device::Cpu` is functional. GPU kernels for conv, pool, norm, etc. are Phase 6 (ferrotorch-gpu)
- Optimizers (SGD, Adam, etc.) -- Phase 3 (ferrotorch-optim)
- ONNX and msgpack serialization -- SafeTensors loading is in scope (REQ-14), but ONNX export and other serialization formats are deferred
- Data loading and batching -- Phase 4 (ferrotorch-data)
- Pre-built model architectures (ResNet, ViT, etc.) -- Phase 5 (ferrotorch-vision)
- Python bindings for nn modules -- late phase (ferrotorch-python)
- GRU and vanilla RNN -- LSTM covers the primary RNN use case; GRU/RNN can be added as a follow-up without API changes
- Automatic mixed-precision autocast (selecting bf16 vs f32 per-op automatically) -- modules must work with bf16 tensors (REQ-15), but the autocast context manager that automatically selects precision per-operation is deferred
- Lazy modules (infer shapes on first forward) -- defer to avoid complexity; shapes must be specified at construction
- Custom user-defined autograd functions -- users can implement `GradFn<T>` directly since the trait is public in core
- Packed sequences for variable-length RNN inputs -- defer to a follow-up; fixed-length batches with padding are sufficient for Phase 2
