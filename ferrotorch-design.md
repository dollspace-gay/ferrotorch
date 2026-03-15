# ferrotorch — PyTorch in Rust

**A ground-up deep learning framework built on ferray, with PyTorch-compatible semantics.**

ferray gives us NumPy. ferrolearn gives us scikit-learn. ferrotorch completes the stack with PyTorch: reverse-mode autograd, neural network modules, optimizers, data loading, and GPU acceleration.

---

## Design Principles

1. **ferray is the array library, ferrotorch is the framework** — ferrotorch depends on ferray-core for storage and math kernels. It does not reimplement array operations.
2. **Tensor is the central type** — A wrapper around `ferray_core::NdArray<T, D>` that adds gradient tracking and device placement. All user-facing code works with `Tensor`, not raw arrays.
3. **Dynamic graphs by default** — Computation graphs are built on-the-fly during forward pass and discarded after backward, exactly like PyTorch's eager mode.
4. **Type safety where it helps, not where it hurts** — Compile-time dimension checking for common patterns (Linear layer shapes), dynamic dimensions for the rest. Don't fight the type system.
5. **GPU is not an afterthought** — Device abstraction is in the core type from day one, even if CPU is the only backend initially.
6. **Zero-panic guarantee** — Inherited from ferray. All public functions return `Result<T, FerrotorchError>`.

---

## Workspace Structure

```
ferrotorch/
├── Cargo.toml                    # Workspace root
├── ferrotorch-design.md          # This document
│
├── ferrotorch-core/              # Phase 1: Tensor + autograd engine
│   ├── src/
│   │   ├── lib.rs
│   │   ├── tensor.rs             # Tensor<T> — the central type
│   │   ├── storage.rs            # TensorStorage — owned data + device tag
│   │   ├── device.rs             # Device enum (Cpu, Cuda(id), Metal)
│   │   ├── dtype.rs              # Re-export ferray DType + bf16
│   │   ├── autograd/
│   │   │   ├── mod.rs
│   │   │   ├── graph.rs          # GraphNode, topological sort, backward engine
│   │   │   ├── tape.rs           # Thread-local gradient tape
│   │   │   ├── function.rs       # Function trait (forward + backward)
│   │   │   ├── no_grad.rs        # no_grad() context manager
│   │   │   └── checkpoint.rs     # Gradient checkpointing (trade compute for memory)
│   │   ├── grad_fns/             # VJP implementations for every differentiable op
│   │   │   ├── mod.rs
│   │   │   ├── arithmetic.rs     # add, sub, mul, div, neg, pow
│   │   │   ├── reduction.rs      # sum, mean, max, min, prod
│   │   │   ├── linalg.rs         # matmul, bmm, dot, mv, mm
│   │   │   ├── activation.rs     # relu, sigmoid, tanh, gelu, silu, softmax, log_softmax
│   │   │   ├── shape.rs          # reshape, transpose, permute, expand, squeeze, cat, stack, split, chunk
│   │   │   ├── indexing.rs       # gather, scatter, index_select, masked_select
│   │   │   ├── comparison.rs     # where_ (differentiable through selected branch)
│   │   │   ├── norm.rs           # layer_norm, batch_norm, group_norm (functional)
│   │   │   ├── conv.rs           # conv1d, conv2d, conv3d, conv_transpose2d
│   │   │   ├── pool.rs           # max_pool2d, avg_pool2d, adaptive_avg_pool2d
│   │   │   ├── dropout.rs        # dropout (mask-based, differentiable)
│   │   │   ├── embedding.rs      # embedding lookup
│   │   │   ├── loss.rs           # cross_entropy, mse, nll, bce, huber, ctc
│   │   │   └── attention.rs      # scaled_dot_product_attention
│   │   └── error.rs              # FerrotorchError
│   └── tests/
│
├── ferrotorch-nn/                # Phase 2: Neural network modules
│   ├── src/
│   │   ├── lib.rs
│   │   ├── module.rs             # Module trait, parameter registry, train/eval mode
│   │   ├── parameter.rs          # Parameter<T> — tensor registered for gradient descent
│   │   ├── linear.rs             # Linear (fully connected)
│   │   ├── conv.rs               # Conv1d, Conv2d, Conv3d, ConvTranspose2d
│   │   ├── norm.rs               # BatchNorm1d/2d, LayerNorm, GroupNorm, InstanceNorm, RMSNorm
│   │   ├── activation.rs         # ReLU, GELU, SiLU, Sigmoid, Tanh, Softmax, LeakyReLU, ELU, Mish
│   │   ├── dropout.rs            # Dropout, Dropout2d
│   │   ├── pooling.rs            # MaxPool2d, AvgPool2d, AdaptiveAvgPool2d
│   │   ├── rnn.rs                # RNN, LSTM, GRU
│   │   ├── attention.rs          # MultiheadAttention
│   │   ├── embedding.rs          # Embedding, EmbeddingBag
│   │   ├── container.rs          # Sequential, ModuleList, ModuleDict
│   │   ├── init.rs               # xavier_uniform, kaiming_normal, etc.
│   │   └── loss.rs               # CrossEntropyLoss, MSELoss, BCELoss, HuberLoss, etc.
│   └── tests/
│
├── ferrotorch-optim/             # Phase 3: Optimizers + schedulers
│   ├── src/
│   │   ├── lib.rs
│   │   ├── optimizer.rs          # Optimizer trait, step(), zero_grad()
│   │   ├── sgd.rs                # SGD (momentum, Nesterov, weight decay)
│   │   ├── adam.rs               # Adam, AdamW (decoupled weight decay)
│   │   ├── rmsprop.rs            # RMSprop
│   │   ├── adagrad.rs            # Adagrad
│   │   ├── lbfgs.rs              # L-BFGS (reuse ferrolearn-numerical math)
│   │   └── scheduler.rs          # StepLR, CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau, LinearWarmup
│   └── tests/
│
├── ferrotorch-data/              # Phase 4: Data loading
│   ├── src/
│   │   ├── lib.rs
│   │   ├── dataset.rs            # Dataset trait: len() + get(index) -> Sample
│   │   ├── dataloader.rs         # DataLoader: batching, shuffling, parallel prefetch
│   │   ├── sampler.rs            # RandomSampler, SequentialSampler, DistributedSampler
│   │   ├── collate.rs            # Collate functions (stack samples into batches)
│   │   └── transforms.rs         # Compose, Normalize, ToTensor, RandomCrop, RandomFlip
│   └── tests/
│
├── ferrotorch-vision/            # Phase 5: torchvision equivalent
│   ├── src/
│   │   ├── lib.rs
│   │   ├── models/               # ResNet, VGG, EfficientNet, ViT, etc.
│   │   ├── transforms.rs         # Image-specific transforms (Resize, CenterCrop, ColorJitter)
│   │   └── datasets.rs           # MNIST, CIFAR-10, ImageNet loader
│   └── tests/
│
├── ferrotorch-gpu/               # Phase 6: GPU backend
│   ├── src/
│   │   ├── lib.rs
│   │   ├── cuda/                 # CUDA backend (cudarc + cuBLAS + cuDNN + cuFFT)
│   │   │   ├── mod.rs
│   │   │   ├── allocator.rs      # Caching memory allocator
│   │   │   ├── stream.rs         # CUDA stream management
│   │   │   ├── kernels/          # Custom CUDA kernels (elementwise, reduction, etc.)
│   │   │   ├── blas.rs           # cuBLAS bindings (matmul, gemm)
│   │   │   ├── dnn.rs            # cuDNN bindings (conv, pooling, normalization)
│   │   │   └── fft.rs            # cuFFT bindings
│   │   ├── metal/                # Metal backend (Apple Silicon) — future
│   │   ├── vulkan/               # Vulkan compute — future
│   │   └── cubecl/               # CubeCL portable kernels — future
│   └── tests/
│
├── ferrotorch-distributed/       # Phase 7: Multi-GPU / multi-node
│   ├── src/
│   │   ├── lib.rs
│   │   ├── backend.rs            # Communication backend (NCCL, Gloo, MPI)
│   │   ├── process_group.rs      # Process group management
│   │   ├── collective.rs         # allreduce, broadcast, allgather, reduce_scatter
│   │   ├── ddp.rs                # DistributedDataParallel wrapper
│   │   ├── fsdp.rs               # FullyShardedDataParallel
│   │   └── pipeline.rs           # Pipeline parallelism (GPipe-style)
│   └── tests/
│
├── ferrotorch-serialize/         # Phase 3: Model serialization
│   ├── src/
│   │   ├── lib.rs
│   │   ├── state_dict.rs         # Save/load state dicts (weights only)
│   │   ├── safetensors.rs        # SafeTensors format (HuggingFace compatible)
│   │   ├── onnx.rs               # ONNX export
│   │   └── checkpoint.rs         # Training checkpoint (model + optimizer + epoch)
│   └── tests/
│
├── ferrotorch-jit/               # Phase 8: JIT compilation / tracing
│   ├── src/
│   │   ├── lib.rs
│   │   ├── trace.rs              # Trace a forward pass into a frozen graph
│   │   ├── graph.rs              # IR graph representation
│   │   ├── optimize.rs           # Graph optimizations (fusion, constant folding, dead code)
│   │   └── codegen.rs            # Code generation (LLVM? Cranelift? CUDA PTX?)
│   └── tests/
│
├── ferrotorch-python/            # Python bindings (PyO3)
│   ├── src/
│   │   ├── lib.rs
│   │   ├── tensor.rs             # Python Tensor class
│   │   ├── nn.rs                 # Python nn.Module wrappers
│   │   └── optim.rs              # Python optimizer wrappers
│   └── tests/
│
└── ferrotorch/                   # Top-level re-export crate
    └── src/
        └── lib.rs                # pub use ferrotorch_core::*; etc.
```

---

## Phase Plan

### Phase 1: Autograd Engine (ferrotorch-core)

The foundation. Nothing else works without this.

**Tensor type:**

```rust
/// The central type. Wraps ferray's NdArray with gradient tracking and device placement.
pub struct Tensor<T: Element = f32> {
    /// The actual data, stored on a device.
    storage: TensorStorage<T>,
    /// Shape and strides (borrowed from ferray's dimension system).
    shape: Vec<usize>,
    /// Gradient accumulated during backward pass.
    grad: Option<Box<Tensor<T>>>,
    /// Node in the computation graph (None if this is a leaf / detached).
    grad_fn: Option<Arc<dyn GradFn<T>>>,
    /// Whether this tensor participates in gradient computation.
    requires_grad: bool,
    /// Whether this is a leaf tensor (created by the user, not by an operation).
    is_leaf: bool,
}
```

**Computation graph:**

Each operation creates a `GraphNode` that stores:
- References to input tensors (via `Arc` for shared ownership)
- A `backward` closure that computes gradients w.r.t. inputs given the output gradient
- Metadata for topological sorting

```rust
pub trait GradFn<T: Element>: Send + Sync {
    /// Compute gradients of inputs given gradient of output.
    /// Returns one gradient per input tensor.
    fn backward(&self, grad_output: &Tensor<T>) -> Vec<Tensor<T>>;

    /// Input tensors (for graph traversal).
    fn inputs(&self) -> &[TensorRef<T>];
}
```

**Backward pass:**

```rust
impl<T: Element> Tensor<T> {
    /// Compute gradients of all leaf tensors that contribute to this tensor.
    pub fn backward(&self) -> Result<(), FerrotorchError> {
        // 1. Topological sort from self to all leaves
        // 2. Walk in reverse, calling each node's GradFn::backward()
        // 3. Accumulate gradients on leaf tensors (additive for shared inputs)
    }
}
```

**no_grad context:**

```rust
/// Disable gradient tracking for inference.
pub fn no_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    GRAD_ENABLED.with(|g| g.set(false));
    let result = f();
    GRAD_ENABLED.with(|g| g.set(true));
    result
}
```

**Grad functions to implement (Phase 1 scope):**

| Category | Operations | Backward complexity |
|----------|-----------|-------------------|
| Arithmetic | add, sub, mul, div, neg, pow, sqrt, abs | Simple (elementwise) |
| Reduction | sum, mean, prod | Broadcast gradient back |
| Linalg | matmul, bmm, mm, mv, dot | Transpose + matmul |
| Activation | relu, sigmoid, tanh, gelu, silu, softmax, log_softmax | Elementwise with mask/Jacobian |
| Shape | reshape, transpose, permute, expand, contiguous, cat, stack, split, squeeze, unsqueeze, flatten | Reshape/transpose gradient |
| Indexing | gather, scatter_add, index_select, masked_fill | Sparse gradient accumulation |
| Comparison | where_ | Route gradient to selected branch |

**Acceptance criteria for Phase 1:**
- `Tensor` supports f32 and f64
- Forward + backward for all ops above
- `backward()` computes correct gradients (verified against PyTorch numerically)
- `no_grad()` context works
- Thread-safe (computation graphs are `Send + Sync`)
- Gradient checkpointing for memory-efficient deep graphs

---

### Phase 2: Neural Network Modules (ferrotorch-nn)

**Module trait:**

```rust
pub trait Module<T: Element = f32>: Send + Sync {
    /// Forward pass. Takes input tensor(s), returns output tensor(s).
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>, FerrotorchError>;

    /// Iterate over all parameters (for optimizer registration).
    fn parameters(&self) -> Vec<&Parameter<T>>;

    /// Iterate over all parameters mutably.
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>>;

    /// Set training mode (affects dropout, batchnorm).
    fn train(&mut self);

    /// Set evaluation mode.
    fn eval(&mut self);

    /// Named parameters for state dict serialization.
    fn named_parameters(&self) -> Vec<(&str, &Parameter<T>)>;

    /// Load a state dict.
    fn load_state_dict(&mut self, state: &StateDict<T>) -> Result<(), FerrotorchError>;

    /// Export a state dict.
    fn state_dict(&self) -> StateDict<T>;
}
```

**Parameter type:**

```rust
/// A tensor that's registered for gradient descent.
/// Always has requires_grad=true. Stored inside Module implementations.
pub struct Parameter<T: Element = f32> {
    pub data: Tensor<T>,
}
```

**Layers to implement (Phase 2 scope):**

| Layer | PyTorch equivalent | Key details |
|-------|-------------------|-------------|
| `Linear` | `nn.Linear` | weight (out×in) + optional bias |
| `Conv2d` | `nn.Conv2d` | im2col + matmul approach for forward, col2im for backward |
| `BatchNorm2d` | `nn.BatchNorm2d` | Running mean/var, affine params, different train/eval behavior |
| `LayerNorm` | `nn.LayerNorm` | Normalize over last N dims |
| `GroupNorm` | `nn.GroupNorm` | Normalize over channel groups |
| `RMSNorm` | (common in LLMs) | Like LayerNorm without mean centering |
| `Dropout` | `nn.Dropout` | Random mask during training, identity during eval |
| `Embedding` | `nn.Embedding` | Lookup table, sparse gradient |
| `MultiheadAttention` | `nn.MultiheadAttention` | Q/K/V projections + scaled_dot_product_attention |
| `LSTM` | `nn.LSTM` | Bidirectional, multi-layer, packed sequences |
| `Sequential` | `nn.Sequential` | Chain modules |
| `ModuleList` | `nn.ModuleList` | List with parameter registration |

**Loss functions (in ferrotorch-nn or separate):**

| Loss | PyTorch equivalent | Notes |
|------|-------------------|-------|
| `CrossEntropyLoss` | `nn.CrossEntropyLoss` | LogSoftmax + NLLLoss, label smoothing |
| `MSELoss` | `nn.MSELoss` | Mean/sum reduction |
| `BCEWithLogitsLoss` | `nn.BCEWithLogitsLoss` | Numerically stable |
| `HuberLoss` | `nn.HuberLoss` | Smooth L1 |
| `CTCLoss` | `nn.CTCLoss` | For sequence models |
| `TripletMarginLoss` | `nn.TripletMarginLoss` | Metric learning |

**Weight initialization:**

| Function | PyTorch equivalent |
|----------|-------------------|
| `xavier_uniform` | `nn.init.xavier_uniform_` |
| `xavier_normal` | `nn.init.xavier_normal_` |
| `kaiming_uniform` | `nn.init.kaiming_uniform_` |
| `kaiming_normal` | `nn.init.kaiming_normal_` |
| `uniform` | `nn.init.uniform_` |
| `normal` | `nn.init.normal_` |
| `zeros` | `nn.init.zeros_` |
| `ones` | `nn.init.ones_` |

---

### Phase 3: Optimizers + Serialization (ferrotorch-optim, ferrotorch-serialize)

**Optimizer trait:**

```rust
pub trait Optimizer<T: Element = f32> {
    /// Perform one optimization step (update parameters using their .grad).
    fn step(&mut self) -> Result<(), FerrotorchError>;

    /// Zero out all parameter gradients.
    fn zero_grad(&mut self);

    /// Get/set learning rate.
    fn lr(&self) -> T;
    fn set_lr(&mut self, lr: T);

    /// State dict for checkpointing.
    fn state_dict(&self) -> OptimizerState;
    fn load_state_dict(&mut self, state: &OptimizerState) -> Result<(), FerrotorchError>;
}
```

**Optimizers:**

| Optimizer | Key features |
|-----------|-------------|
| `SGD` | Momentum, Nesterov acceleration, weight decay |
| `Adam` | Adaptive learning rates, bias correction |
| `AdamW` | Decoupled weight decay (preferred for transformers) |
| `RMSprop` | Running mean of squared gradients |
| `Adagrad` | Accumulated gradient scaling |
| `LBFGS` | Quasi-Newton (reuse math from ferrolearn-numerical) |

**Learning rate schedulers:**

| Scheduler | When to use |
|-----------|------------|
| `StepLR` | Drop LR every N epochs |
| `CosineAnnealingLR` | Cosine decay to min LR |
| `OneCycleLR` | Super-convergence (warmup → max → anneal) |
| `ReduceLROnPlateau` | Reduce when metric stops improving |
| `LinearWarmup` | Linear ramp from 0 to base LR |
| `CosineWarmupScheduler` | Warmup + cosine decay (standard for transformers) |

**Serialization:**

| Format | Purpose |
|--------|---------|
| State dict (msgpack) | Fast save/load of model weights |
| SafeTensors | HuggingFace-compatible, memory-mapped, safe |
| ONNX | Interoperability with other frameworks |
| Training checkpoint | Model + optimizer state + epoch + RNG state |

---

### Phase 4: Data Loading (ferrotorch-data)

```rust
pub trait Dataset: Send + Sync {
    type Sample;

    fn len(&self) -> usize;
    fn get(&self, index: usize) -> Result<Self::Sample, FerrotorchError>;
}

pub struct DataLoader<D: Dataset> {
    dataset: Arc<D>,
    batch_size: usize,
    shuffle: bool,
    num_workers: usize,       // Rayon-based parallel prefetch
    drop_last: bool,
    collate_fn: Box<dyn CollateFn<D::Sample>>,
}

impl<D: Dataset> DataLoader<D> {
    /// Returns an iterator over batches.
    pub fn iter(&self) -> DataLoaderIter<D> { ... }
}
```

**Transforms** (composable pipeline):

```rust
pub trait Transform<Input, Output = Input>: Send + Sync {
    fn apply(&self, input: Input) -> Result<Output, FerrotorchError>;
}

// Compose transforms:
let transform = Compose::new(vec![
    Box::new(ToTensor),
    Box::new(Normalize::new(mean, std)),
    Box::new(RandomHorizontalFlip::new(0.5)),
]);
```

---

### Phase 5: Vision (ferrotorch-vision)

Pre-built model architectures and image datasets.

**Models:**

| Model | Purpose |
|-------|---------|
| ResNet (18/34/50/101/152) | Image classification backbone |
| VGG (11/13/16/19) | Classic CNN |
| EfficientNet (B0-B7) | Efficient mobile architectures |
| Vision Transformer (ViT) | Transformer-based image classification |
| U-Net | Semantic segmentation |
| YOLO (v5/v8 style) | Object detection |

**Datasets:**

| Dataset | Size | Task |
|---------|------|------|
| MNIST | 60K train, 10K test | Digit classification |
| CIFAR-10/100 | 50K/50K | Image classification |
| ImageNet (loader) | 1.2M train | Large-scale classification |

---

### Phase 6: GPU Backend (ferrotorch-gpu)

**Device abstraction (in ferrotorch-core from day one):**

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    Cuda(usize),      // GPU index
    // Metal,          // Future: Apple Silicon
    // Vulkan,         // Future: portable GPU compute
}

impl<T: Element> Tensor<T> {
    pub fn to(&self, device: Device) -> Result<Tensor<T>, FerrotorchError>;
    pub fn device(&self) -> Device;
    pub fn cuda(&self) -> Result<Tensor<T>, FerrotorchError> { self.to(Device::Cuda(0)) }
    pub fn cpu(&self) -> Result<Tensor<T>, FerrotorchError> { self.to(Device::Cpu) }
}
```

**CUDA backend (via cudarc):**

| Component | Library | Purpose |
|-----------|---------|---------|
| Memory allocator | Custom | Caching allocator (avoid cudaMalloc per op) |
| BLAS | cuBLAS | matmul, gemm, gemv |
| Convolution | cuDNN | conv2d forward/backward, pooling, normalization |
| FFT | cuFFT | GPU FFT for signal processing |
| Elementwise | Custom CUDA kernels | Activations, arithmetic, reductions |
| Streams | CUDA streams | Async kernel execution |
| Events | CUDA events | Synchronization points |

**Memory management:**

```rust
/// Caching allocator — reuse freed GPU memory blocks.
/// Matches PyTorch's CUDACachingAllocator pattern.
pub struct CudaAllocator {
    /// Free blocks organized by size (power-of-2 buckets).
    free_blocks: HashMap<usize, Vec<CudaBlock>>,
    /// Total allocated bytes.
    allocated_bytes: AtomicUsize,
    /// Peak allocated bytes.
    peak_bytes: AtomicUsize,
}
```

---

### Phase 7: Distributed Training (ferrotorch-distributed)

| Feature | PyTorch equivalent | Backend |
|---------|-------------------|---------|
| `DistributedDataParallel` | `torch.nn.parallel.DistributedDataParallel` | NCCL (GPU), Gloo (CPU) |
| `FullyShardedDataParallel` | `torch.distributed.fsdp.FSDP` | NCCL |
| AllReduce / Broadcast / AllGather | `torch.distributed` | NCCL / MPI |
| Pipeline parallelism | GPipe / PipeDream | Custom |
| Tensor parallelism | Megatron-style | Column/row parallel Linear |

---

### Phase 8: JIT / Graph Optimization (ferrotorch-jit)

**Tracing:**

```rust
/// Trace a forward function to produce a frozen computation graph.
pub fn trace<F>(f: F, example_inputs: &[Tensor]) -> Result<TracedModule, FerrotorchError>
where
    F: Fn(&[Tensor]) -> Result<Tensor, FerrotorchError>,
```

**Graph optimizations:**

| Optimization | Effect |
|-------------|--------|
| Constant folding | Evaluate constant subgraphs at trace time |
| Operator fusion | Fuse elementwise chains into single kernel |
| Dead code elimination | Remove unused computations |
| Memory planning | Pre-allocate activation buffers, reuse memory |
| Kernel selection | Choose optimal implementation per hardware |

---

## Dependency Chain

```
ferray-core ──────────────────────────────────────────┐
  (NdArray, Element, ufuncs, linalg, fft)             │
                                                      ▼
                                              ferrotorch-core
                                         (Tensor, autograd, grad_fns)
                                           │         │         │
                                           ▼         ▼         ▼
                                    ferrotorch-nn  ferrotorch-optim  ferrotorch-serialize
                                    (modules,      (SGD, Adam,       (state_dict,
                                     layers,        schedulers)       safetensors,
                                     loss fns)                        ONNX)
                                           │
                                           ▼
                                    ferrotorch-data
                                    (DataLoader, Dataset, transforms)
                                           │
                                           ▼
                                    ferrotorch-vision
                                    (ResNet, ViT, MNIST, CIFAR)

ferrotorch-gpu (CUDA/Metal backends) ──► plugs into ferrotorch-core via Device trait
ferrotorch-distributed ──► wraps ferrotorch-nn modules with gradient sync
ferrotorch-jit ──► traces ferrotorch-core computation graphs
ferrotorch-python ──► PyO3 bindings for everything above
```

---

## Edition & Toolchain

| Setting | Value |
|---------|-------|
| Rust edition | 2024 |
| MSRV | 1.85 |
| License | MIT OR Apache-2.0 |
| Error handling | `thiserror` 2.0, `FerrotorchError` |
| Parallelism | `rayon` 1.11 |
| Serialization | `serde` 1.0, `rmp-serde` 1.3, `safetensors` |
| CUDA | `cudarc` (bindings to CUDA driver/runtime) |
| Linear algebra | ferray-linalg (faer 0.24) |
| FFT | ferray-fft (rustfft 6.4) |
| Random | ferray-random |
| Python bindings | `pyo3` 0.24 |

---

## What Exists in the Ecosystem (and why we're still building this)

| Crate | Status | Why not use it |
|-------|--------|---------------|
| **tch-rs** | Mature | Wrapper around libtorch C++ — not pure Rust, massive binary, C++ build dependency |
| **candle** | Active (HuggingFace) | Good for inference, limited training support, no distributed |
| **burn** | Active | Closest competitor. Different design tradeoffs: backend-agnostic but complex trait system. ferrotorch is opinionated (ferray backend, PyTorch semantics) |
| **dfdx** | Stale | Const-generic shapes are elegant but impractical for dynamic models (transformers, RNNs) |

ferrotorch's niche: **PyTorch semantics in pure Rust, built on ferray's proven NumPy foundation, with the same eager-mode experience researchers expect.**

---

## Implementation Priority

| Phase | Crate | What ships | Enables |
|-------|-------|-----------|---------|
| **1** | ferrotorch-core | Tensor + autograd + ~50 grad functions | Training any model |
| **2** | ferrotorch-nn | Linear, Conv2d, BatchNorm, LSTM, Attention, losses | Building real architectures |
| **3** | ferrotorch-optim + ferrotorch-serialize | Adam/SGD + save/load | Complete training loop |
| **4** | ferrotorch-data | DataLoader + transforms | Practical training on real datasets |
| **5** | ferrotorch-vision | ResNet, ViT, MNIST, CIFAR | Out-of-box demos and benchmarks |
| **6** | ferrotorch-gpu | CUDA backend | Practical training speed |
| **7** | ferrotorch-distributed | DDP, FSDP | Multi-GPU / multi-node scaling |
| **8** | ferrotorch-jit | Tracing + optimization | Production inference |

Phases 1-4 are the MVP. A training loop that can train a transformer on CPU with Adam is usable. GPU (Phase 6) makes it practical. Everything else is polish.

---

## Success Metric

**Can we train GPT-2 (124M parameters) from scratch on a single GPU, matching PyTorch's throughput within 2x?**

If yes, ferrotorch is real. If not, it's a toy.
