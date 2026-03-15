# Feature: Phase 6 — GPU Backend (ferrotorch-gpu)

## Summary
The GPU acceleration crate for ferrotorch. Implements a CUDA backend via cudarc that extends the `StorageBuffer<T>` enum in ferrotorch-core with a `Cuda(CudaBuffer<T>)` variant, provides a caching memory allocator, CUDA stream/event management, and dispatches tensor operations to cuBLAS (matmul/GEMM), cuDNN (conv/pooling/normalization), cuFFT (FFT), and custom CUDA kernels (elementwise, activations, reductions). Enables `tensor.to(Device::Cuda(0))`, `tensor.cuda()`, and `tensor.cpu()` for seamless device transfer. All CPU-side tensor operations continue to work unchanged; the GPU backend plugs in via the existing `Device` abstraction defined in Phase 1.

## Requirements
- REQ-1: The `StorageBuffer<T>` enum in ferrotorch-core must be extended with a `Cuda(CudaBuffer<T>)` variant that holds a device pointer, byte length, and the device index. The `Tensor<T>` public API must not change — `tensor.device()` returns the correct `Device`, and all existing CPU code paths remain unaffected.
- REQ-2: `tensor.to(device)`, `tensor.cuda()`, and `tensor.cpu()` must perform synchronous host-to-device, device-to-host, and device-to-device memory transfers. Transferring a tensor that is already on the target device must return a zero-cost clone (shared storage via `Arc`) with no memory copy.
- REQ-3: `CudaAllocator` must implement a caching memory allocator with power-of-2 bucket sizing that reuses freed GPU memory blocks. It must track `allocated_bytes` and `peak_bytes` via atomics and expose `memory_allocated()`, `max_memory_allocated()`, and `empty_cache()` methods. Allocations that exceed all cached blocks must fall through to `cuMemAlloc`; returned blocks must be retained in the cache until `empty_cache()` is called.
- REQ-4: CUDA stream management must support creating named streams per device, executing kernels asynchronously on a stream, and synchronizing via CUDA events. Each device must have a default stream. `tensor.record_stream(stream)` must ensure the tensor's memory is not freed until the stream's pending work completes.
- REQ-5: cuBLAS integration must provide GPU-accelerated `matmul`, `gemm`, `gemv`, and `batched_gemm` for f32, f64, and f16. cuBLAS handles must be cached per-device and associated with the active CUDA stream. All cuBLAS calls must use the caching allocator's workspace memory rather than cuBLAS-internal allocations.
- REQ-6: cuDNN integration must provide GPU-accelerated `conv2d_forward`, `conv2d_backward_data`, `conv2d_backward_filter`, `max_pool2d`, `avg_pool2d`, `batch_norm`, and `layer_norm`. Convolution algorithm selection must use cuDNN's heuristic finder with workspace size limits, caching the selected algorithm per input shape to avoid repeated searches.
- REQ-7: cuFFT integration must provide GPU-accelerated 1D, 2D, and 3D real-to-complex and complex-to-complex FFTs for f32 and f64, matching the interface used by ferray-fft on CPU.
- REQ-8: Custom CUDA kernels must implement elementwise operations (add, sub, mul, div, neg, pow, sqrt, abs, exp, log), activations (relu, sigmoid, tanh, gelu, silu), and reductions (sum, mean, max, min, prod) with configurable block/grid dimensions. Kernel launch must go through a unified dispatch layer that selects block size based on tensor element count.
- REQ-9: Every GPU operation must validate that all input tensors reside on the same CUDA device before dispatching. Device-mismatched inputs must return `Err(FerrotorchError::DeviceMismatch { .. })` without launching any kernel.
- REQ-10: The GPU backend must be gated behind a `cuda` Cargo feature flag. When the feature is disabled, `Device::Cuda(_)` remains in the enum but all operations on CUDA tensors return `Err(FerrotorchError::DeviceUnavailable { .. })`. The crate must compile and pass tests on machines without NVIDIA GPUs or CUDA toolkit installed.
- REQ-11: All public functions must return `Result<T, FerrotorchError>`. CUDA driver errors, cuBLAS status codes, cuDNN status codes, and out-of-memory conditions must be mapped to descriptive `FerrotorchError` variants — never panics.
- REQ-12: GPU tensors must participate in the autograd graph identically to CPU tensors. `backward()` on a GPU-resident scalar must compute gradients on GPU without implicit device transfers. Gradient accumulation on GPU leaf tensors must use GPU-side addition.
- REQ-13: Mixed-precision training must provide an `autocast(|| { ... })` context manager that automatically casts eligible operations to f16 or bf16. Policy-based op casting must route matmul and conv ops to f16, while reductions and normalization ops remain in f32, matching PyTorch's default `torch.cuda.amp` policy.
- REQ-14: `GradScaler` must scale the loss before backward, unscale gradients before the optimizer step, and detect inf/nan in gradients to skip the optimizer step when overflow occurs. The scale factor must dynamically adjust: increase after N consecutive successful steps, halve on overflow. This matches PyTorch's `torch.cuda.amp.GradScaler` behavior.
- REQ-15: Multi-GPU memory management must support `cuda::set_device(id)` to set the active device for the current thread, with per-device `CudaAllocator` instances. Each device's allocator must be independent — `empty_cache()` on device 0 must not affect device 1's cache. This matches `torch.cuda.set_device()` semantics.
- REQ-16: CUDA graph capture must provide `cuda::graph::capture(|| { ... })` that records a sequence of GPU operations into a `CudaGraph` object. `graph.replay()` must re-execute the captured sequence with minimal kernel launch overhead. Graphs must be invalidated if tensor shapes change. This matches `torch.cuda.CUDAGraph` semantics.
- REQ-17: Benchmark mode must provide `cuda::benchmark_mode(true)` that enables cuDNN algorithm auto-tuning. When enabled, the first convolution with a given input shape runs all available algorithms, selects the fastest, and caches the result. Subsequent calls with the same shape use the cached algorithm. This matches `torch.backends.cudnn.benchmark = True` behavior.

## Acceptance Criteria
- [ ] AC-1: `StorageBuffer<T>` has a `Cuda(CudaBuffer<T>)` variant. Existing CPU tests in ferrotorch-core pass without modification after the enum extension.
- [ ] AC-2: `Tensor::<f32>::rand([1024, 1024]).to(Device::Cuda(0))` allocates on GPU, and `.cpu()` round-trips with element-wise equality to the original. Device-to-device transfer between `Cuda(0)` and `Cuda(1)` works on multi-GPU systems.
- [ ] AC-3: `CudaAllocator` reuses freed blocks — allocating and freeing a 1 MiB buffer 1000 times calls `cuMemAlloc` at most once. `memory_allocated()` reports the correct live byte count. `empty_cache()` releases all cached blocks back to the CUDA driver.
- [ ] AC-4: A custom CUDA stream can launch a matmul asynchronously. `stream.synchronize()` blocks until completion. CUDA events inserted between two kernels on different streams enforce correct ordering.
- [ ] AC-5: GPU matmul via cuBLAS matches CPU matmul output (ferray-linalg) within `rtol=1e-5, atol=1e-5` for f32 on matrices up to 4096x4096. Batched GEMM on a [32, 512, 512] tensor produces correct results.
- [ ] AC-6: cuDNN conv2d forward on a [1, 64, 224, 224] input with a [128, 64, 3, 3] kernel produces output matching PyTorch `F.conv2d()` within `rtol=1e-4, atol=1e-4`. Algorithm caching avoids repeated heuristic searches for repeated calls with the same shapes.
- [ ] AC-7: GPU elementwise add of two [10_000_000]-element f32 tensors produces correct results and completes faster than the equivalent CPU operation (wall-clock, excluding transfer time).
- [ ] AC-8: GPU reductions (sum, mean, max) on a [1_000_000]-element tensor produce results matching CPU reductions within `rtol=1e-5, atol=1e-6`.
- [ ] AC-9: `tensor_a.on(Cuda(0)) + tensor_b.on(Cuda(1))` returns `Err(FerrotorchError::DeviceMismatch { .. })`, not a panic or silent device transfer.
- [ ] AC-10: Building with `--no-default-features` (cuda feature disabled) compiles successfully. `Tensor::rand([2, 3]).to(Device::Cuda(0))` returns `Err(FerrotorchError::DeviceUnavailable { .. })`.
- [ ] AC-11: A full autograd round-trip on GPU — forward (matmul + relu + sum), `backward()`, optimizer step — produces gradients matching the CPU autograd path within `rtol=1e-4, atol=1e-5`. No tensors are silently moved to CPU during backward.
- [ ] AC-12: `cargo test -p ferrotorch-gpu` passes with 0 failures. Minimum 120 tests covering allocator, streams, cuBLAS, cuDNN, custom kernels, device transfer, error paths, autograd integration, mixed precision, CUDA graphs, and benchmark mode.
- [ ] AC-13: `autocast(|| { let c = a.matmul(&b); c.sum() })` executes matmul in f16 and sum in f32 when inputs are f32. The output dtype of the autocast region's matmul must be f16. Reductions, normalization, and loss functions must remain in f32.
- [ ] AC-14: `GradScaler` with an initial scale of 65536.0 correctly scales loss, unscales gradients, and reduces scale factor by half when inf is detected. After 2000 consecutive successful steps (default `growth_interval`), the scale factor doubles. An optimizer step is skipped (no parameter update) when gradients contain inf/nan.
- [ ] AC-15: On a multi-GPU system, `cuda::set_device(0)` followed by tensor allocation places the tensor on device 0. `cuda::set_device(1)` followed by allocation places it on device 1. `empty_cache()` on device 0 does not affect the byte count returned by `memory_allocated()` on device 1.
- [ ] AC-16: `cuda::graph::capture(|| { /* matmul + relu + sum */ })` returns a `CudaGraph` object. `graph.replay()` produces identical output to executing the same operations outside a graph. Replaying a captured graph 1000 times has lower total kernel launch overhead than 1000 non-captured executions (measured via CUDA events).
- [ ] AC-17: With `cuda::benchmark_mode(true)`, the first `conv2d` call with a given shape triggers algorithm benchmarking (measurably slower). The second call with the same shape uses the cached algorithm and does not re-benchmark. Disabling benchmark mode (`cuda::benchmark_mode(false)`) reverts to heuristic algorithm selection.

## Architecture

### Crate Layout

```
ferrotorch-gpu/
├── Cargo.toml
├── src/
│   ├── lib.rs                      # Public re-exports, feature-gate logic
│   ├── cuda/
│   │   ├── mod.rs                  # CUDA backend entry point, device enumeration, initialization
│   │   ├── allocator.rs            # CudaAllocator — caching allocator with power-of-2 buckets
│   │   ├── stream.rs               # CudaStream, CudaEvent — async execution and synchronization
│   │   ├── device_guard.rs         # DeviceGuard RAII — set/restore active CUDA device
│   │   ├── blas.rs                 # cuBLAS wrapper — gemm, gemv, batched_gemm
│   │   ├── dnn.rs                  # cuDNN wrapper — conv2d, pooling, normalization
│   │   ├── fft.rs                  # cuFFT wrapper — 1D/2D/3D forward/inverse transforms
│   │   ├── amp.rs                  # Mixed-precision: autocast context manager, GradScaler
│   │   ├── graph.rs                # CUDA graph capture and replay
│   │   ├── benchmark.rs            # cuDNN benchmark mode (algorithm auto-tuning)
│   │   └── kernels/
│   │       ├── mod.rs              # Kernel launch dispatch (block/grid sizing)
│   │       ├── elementwise.ptx     # PTX source for elementwise ops
│   │       ├── elementwise.rs      # Rust wrappers for elementwise kernel launches
│   │       ├── activation.ptx      # PTX source for activation functions
│   │       ├── activation.rs       # Rust wrappers for activation kernel launches
│   │       ├── reduction.ptx       # PTX source for parallel reductions
│   │       └── reduction.rs        # Rust wrappers for reduction kernel launches
│   ├── buffer.rs                   # CudaBuffer<T> — typed wrapper around device pointer
│   └── error.rs                    # CUDA-specific error mapping into FerrotorchError
├── kernels/                        # CUDA C source files (compiled to PTX at build time)
│   ├── elementwise.cu
│   ├── activation.cu
│   └── reduction.cu
├── build.rs                        # Build script: compile .cu → .ptx via nvcc (optional, PTX can be shipped precompiled)
└── tests/
    ├── test_allocator.rs           # Caching behavior, peak tracking, empty_cache
    ├── test_stream.rs              # Async launch, event synchronization
    ├── test_transfer.rs            # Host↔device, device↔device transfers
    ├── test_blas.rs                # cuBLAS gemm, gemv, batched_gemm correctness
    ├── test_dnn.rs                 # cuDNN conv2d, pooling, normalization correctness
    ├── test_kernels.rs             # Custom kernel correctness (elementwise, activation, reduction)
    ├── test_autograd.rs            # Full forward/backward on GPU
    ├── test_amp.rs                 # autocast dtype casting, GradScaler scaling/overflow
    ├── test_graph.rs               # CUDA graph capture, replay, invalidation
    ├── test_benchmark.rs           # cuDNN benchmark mode algorithm caching
    └── test_no_cuda.rs             # Feature-disabled compilation and graceful errors
```

### Core Types

**CudaBuffer<T>** (`buffer.rs`):
```rust
/// Typed wrapper around a CUDA device pointer.
/// Owns the allocation and frees it (back to the caching allocator) on drop.
pub struct CudaBuffer<T: Element> {
    ptr: DevicePointer<T>,      // cudarc::driver::DevicePtr<T>
    len: usize,                 // Number of elements (not bytes)
    device_id: usize,           // Which GPU this lives on
    allocator: Arc<CudaAllocator>,
}
```

Drop returns the block to the caching allocator rather than calling `cuMemFree`.

**CudaAllocator** (`cuda/allocator.rs`):
```rust
/// Caching memory allocator — reuse freed GPU memory blocks.
/// Matches PyTorch's CUDACachingAllocator pattern.
pub struct CudaAllocator {
    /// Per-device free block pools, keyed by rounded-up size (power-of-2 buckets).
    pools: Vec<Mutex<HashMap<usize, Vec<DevicePointer<u8>>>>>,
    /// Total currently allocated bytes per device.
    allocated_bytes: Vec<AtomicUsize>,
    /// Peak allocated bytes per device (high-water mark).
    peak_bytes: Vec<AtomicUsize>,
    /// cudarc device handles.
    devices: Vec<Arc<CudaDevice>>,
}

impl CudaAllocator {
    pub fn alloc<T: Element>(&self, device_id: usize, len: usize) -> Result<CudaBuffer<T>, FerrotorchError>;
    pub fn free(&self, device_id: usize, ptr: DevicePointer<u8>, size: usize);
    pub fn memory_allocated(&self, device_id: usize) -> usize;
    pub fn max_memory_allocated(&self, device_id: usize) -> usize;
    pub fn empty_cache(&self, device_id: usize);
}
```

Allocation rounds the requested byte count up to the next power of two, checks the corresponding bucket for a cached block, and falls through to `cuMemAlloc` only on a cache miss. This eliminates the ~100us overhead of CUDA driver allocation on the hot path.

**CudaStream** (`cuda/stream.rs`):
```rust
pub struct CudaStream {
    inner: cudarc::driver::CudaStream,
    device_id: usize,
}

pub struct CudaEvent {
    inner: cudarc::driver::CudaEvent,
}

impl CudaStream {
    pub fn new(device_id: usize) -> Result<Self, FerrotorchError>;
    pub fn synchronize(&self) -> Result<(), FerrotorchError>;
    pub fn record_event(&self) -> Result<CudaEvent, FerrotorchError>;
    pub fn wait_event(&self, event: &CudaEvent) -> Result<(), FerrotorchError>;
}
```

Each device has a default stream. Library calls (cuBLAS, cuDNN) are bound to the active stream so overlapping compute and transfer is possible.

**DeviceGuard** (`cuda/device_guard.rs`):
```rust
/// RAII guard that sets the active CUDA device on creation
/// and restores the previous device on drop.
pub struct DeviceGuard {
    previous_device: usize,
}

impl DeviceGuard {
    pub fn new(device_id: usize) -> Result<Self, FerrotorchError>;
}
```

### StorageBuffer Extension

Phase 1 defined `StorageBuffer<T>` in ferrotorch-core with only `Cpu(Vec<T>)`. Phase 6 extends it:

```rust
pub enum StorageBuffer<T: Element> {
    Cpu(Vec<T>),
    #[cfg(feature = "cuda")]
    Cuda(CudaBuffer<T>),
}
```

The `cuda` feature flag lives on ferrotorch-core (re-exported from ferrotorch-gpu). When disabled, the variant is absent and all GPU code paths are compiled out. `TensorStorage<T>` and `Tensor<T>` require no structural changes — the `device` field on `TensorStorage` already reports `Device::Cuda(n)` when the buffer is `StorageBuffer::Cuda(_)`.

### Device Transfer Path

`tensor.to(device)` dispatches based on source and target:

| Source | Target | Action |
|--------|--------|--------|
| Cpu | Cpu | Return `Arc` clone (zero-cost) |
| Cpu | Cuda(n) | `cuMemcpyHtoD` via allocator + stream |
| Cuda(n) | Cpu | `cuMemcpyDtoH` into new `Vec<T>` |
| Cuda(n) | Cuda(n) | Return `Arc` clone (zero-cost, same device) |
| Cuda(n) | Cuda(m) | `cuMemcpyDtoD` via peer access or staged through host |

Peer access (`cuDeviceCanAccessPeer`) is checked once at initialization and cached. If peer access is unavailable, device-to-device falls back to a host-staged copy.

### cuBLAS Integration (`cuda/blas.rs`)

One `CublasHandle` is created per device and cached in a global `Vec<Mutex<CublasHandle>>`. The handle is bound to the current stream before each call. Supported operations:

| Function | cuBLAS call | Supported dtypes |
|----------|------------|-----------------|
| `gpu_gemm` | `cublasSgemm` / `cublasDgemm` / `cublasHgemm` | f32, f64, f16 |
| `gpu_gemv` | `cublasSgemv` / `cublasDgemv` | f32, f64 |
| `gpu_batched_gemm` | `cublasSgemmStridedBatched` / `cublasDgemmStridedBatched` | f32, f64 |

Workspace memory for cuBLAS internals is allocated from the caching allocator via `cublasSetWorkspace`, preventing cuBLAS from making its own `cudaMalloc` calls.

### cuDNN Integration (`cuda/dnn.rs`)

One `CudnnHandle` per device, cached and stream-bound like cuBLAS. Convolution algorithm selection uses `cudnnGetConvolutionForwardAlgorithm_v7` (heuristic) with the result cached in a `HashMap<ConvKey, ConvAlgorithm>` keyed by `(input_shape, filter_shape, padding, stride, dilation)`.

| Function | cuDNN call | Notes |
|----------|-----------|-------|
| `gpu_conv2d_forward` | `cudnnConvolutionForward` | Algorithm auto-selected, workspace from caching allocator |
| `gpu_conv2d_backward_data` | `cudnnConvolutionBackwardData` | For autograd backward pass |
| `gpu_conv2d_backward_filter` | `cudnnConvolutionBackwardFilter` | For autograd backward pass |
| `gpu_max_pool2d` | `cudnnPoolingForward` | Returns indices for backward |
| `gpu_avg_pool2d` | `cudnnPoolingForward` | Mean reduction mode |
| `gpu_batch_norm` | `cudnnBatchNormalizationForwardTraining` | Running stats in train mode, frozen in eval |
| `gpu_layer_norm` | `cudnnNormalizationForwardTraining` | cuDNN 8+ layer norm support |

### cuFFT Integration (`cuda/fft.rs`)

Plans are created via `cufftPlan1d` / `cufftPlan2d` / `cufftPlan3d` and cached per `(shape, dtype, direction)` key. Plans are bound to the active stream. The interface mirrors ferray-fft:

| Function | cuFFT call | Dtypes |
|----------|-----------|--------|
| `gpu_rfft` | `cufftExecR2C` / `cufftExecD2Z` | f32, f64 |
| `gpu_irfft` | `cufftExecC2R` / `cufftExecZ2D` | f32, f64 |
| `gpu_fft` | `cufftExecC2C` / `cufftExecZ2Z` | f32, f64 |

### Custom CUDA Kernels (`cuda/kernels/`)

Written in CUDA C (`.cu` files in `kernels/`), compiled to PTX at build time via `build.rs` calling `nvcc --ptx`. The PTX is embedded into the binary via `include_str!` and loaded at runtime through cudarc's module loading API.

**Elementwise kernels** (`elementwise.cu`): One generic kernel template per operation, instantiated for f32, f64, and f16. Each thread handles one element. Grid dimensions: `(n + block_size - 1) / block_size` blocks of 256 threads.

**Activation kernels** (`activation.cu`): Fused forward and backward variants. relu forward stores the mask for backward reuse. gelu and silu use the same polynomial approximations as ferray-ufunc for bit-exact CPU/GPU consistency.

**Reduction kernels** (`reduction.cu`): Two-pass parallel reduction. First pass: each block reduces its chunk to a single value using shared memory and warp shuffle. Second pass: a single block reduces block-level results. For large tensors (>1M elements), a segmented reduction avoids atomic contention.

### Kernel Launch Dispatch (`cuda/kernels/mod.rs`)

All kernel launches go through a unified dispatch layer:

```rust
pub fn launch_elementwise<T: Element>(
    op: ElementwiseOp,
    inputs: &[&CudaBuffer<T>],
    output: &mut CudaBuffer<T>,
    stream: &CudaStream,
) -> Result<(), FerrotorchError>;

pub fn launch_reduction<T: Element>(
    op: ReductionOp,
    input: &CudaBuffer<T>,
    output: &mut CudaBuffer<T>,
    axis: Option<usize>,
    stream: &CudaStream,
) -> Result<(), FerrotorchError>;
```

The dispatch layer selects block size (128 for reductions, 256 for elementwise) and computes grid dimensions. It validates buffer sizes before launch and maps CUDA launch errors to `FerrotorchError::CudaKernelLaunch { .. }`.

### Autograd Integration

GPU tensors participate in the autograd graph with no special handling. `GradFn<T>` implementations in ferrotorch-core are device-agnostic — they call tensor operations (add, matmul, etc.) that dispatch to GPU kernels when the tensor is on a CUDA device. The dispatch happens at the op level (`ops/elementwise.rs`, `ops/linalg.rs`) by checking `tensor.device()`:

```rust
match self.device() {
    Device::Cpu => { /* existing ferray path */ }
    Device::Cuda(id) => { /* GPU kernel dispatch */ }
}
```

Gradients are allocated on the same device as the tensor they belong to. Gradient accumulation uses GPU-side elementwise add. No implicit device transfers occur during backward.

### Error Mapping (`error.rs`)

CUDA driver errors, cuBLAS status codes, and cuDNN status codes are mapped to `FerrotorchError` variants:

```rust
// Added to FerrotorchError in ferrotorch-core
#[error("CUDA error on device {device_id}: {message}")]
CudaError { device_id: usize, message: String },

#[error("CUDA out of memory on device {device_id}: requested {requested} bytes, {available} bytes available")]
CudaOutOfMemory { device_id: usize, requested: usize, available: usize },

#[error("CUDA kernel launch failed: {kernel} — {message}")]
CudaKernelLaunch { kernel: String, message: String },

#[error("device unavailable: {device:?} (compiled without cuda feature or no GPU detected)")]
DeviceUnavailable { device: Device },
```

### Mixed-Precision Training (`cuda/amp.rs`)

Implements automatic mixed-precision (AMP) training following PyTorch's `torch.cuda.amp` design. Two core components: `autocast` for automatic dtype casting and `GradScaler` for gradient scaling.

**autocast**:
```rust
/// Thread-local autocast state. When enabled, eligible ops dispatch to f16/bf16.
pub fn autocast<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    AUTOCAST_ENABLED.with(|cell| cell.set(true));
    let result = f();
    AUTOCAST_ENABLED.with(|cell| cell.set(false));
    result
}

/// Op casting policy — determines which ops run in reduced precision.
pub enum AutocastPolicy {
    /// Default: matmul, conv, linear in f16; reductions, norms, softmax, loss in f32
    Default,
    /// Custom policy with explicit allow/deny lists
    Custom { fp16_ops: HashSet<OpType>, fp32_ops: HashSet<OpType> },
}
```

The default policy matches PyTorch's categorization:
- **f16-eligible**: matmul, conv2d, linear, bmm, baddbmm
- **f32-required**: softmax, cross_entropy, layer_norm, batch_norm, sum, mean, log_softmax
- **Promote-to-widest**: add, sub, mul, div (cast to the widest input dtype)

Each GPU op dispatch checks `AUTOCAST_ENABLED` and casts inputs before calling the kernel. Original dtypes are tracked so autograd backward correctly casts gradients back.

**GradScaler**:
```rust
pub struct GradScaler {
    scale: f64,                  // Current loss scale factor (default: 65536.0)
    growth_factor: f64,          // Multiplicative increase (default: 2.0)
    backoff_factor: f64,         // Multiplicative decrease (default: 0.5)
    growth_interval: usize,      // Steps between growth attempts (default: 2000)
    consecutive_ok: usize,       // Counter for consecutive non-inf steps
    found_inf: bool,             // Whether current step has inf/nan gradients
}

impl GradScaler {
    pub fn new() -> Self;
    pub fn scale(&self, loss: &Tensor<f32>) -> Tensor<f32>;
    pub fn unscale_(&self, optimizer: &mut impl Optimizer) -> Result<(), FerrotorchError>;
    pub fn step(&mut self, optimizer: &mut impl Optimizer) -> Result<(), FerrotorchError>;
    pub fn update(&mut self);
}
```

`scale()` multiplies the loss by the scale factor before backward. `unscale_()` divides all parameter gradients by the scale factor and checks for inf/nan. `step()` calls `optimizer.step()` only if no inf/nan was detected. `update()` adjusts the scale factor: halve on inf/nan, double after `growth_interval` consecutive clean steps.

### Multi-GPU Memory Management

Each CUDA device gets its own `CudaAllocator` instance. The global allocator registry is indexed by device ID:

```rust
/// Global per-device allocator registry.
static ALLOCATORS: OnceLock<Vec<Arc<CudaAllocator>>> = OnceLock::new();

/// Set the active CUDA device for the current thread.
/// All subsequent tensor allocations use this device's allocator.
/// Matches `torch.cuda.set_device()` semantics.
pub fn set_device(device_id: usize) -> Result<(), FerrotorchError>;

/// Get the current thread's active CUDA device.
pub fn current_device() -> usize;
```

Thread-local device context is stored in a `thread_local!` cell. `set_device()` calls `cuCtxSetCurrent` and updates the thread-local. Allocators are fully independent — `empty_cache()` on device 0 releases only device 0's cached blocks.

### CUDA Graph Capture (`cuda/graph.rs`)

CUDA graphs capture a sequence of GPU operations and replay them with minimal launch overhead, eliminating per-kernel CPU-side dispatch costs.

```rust
pub struct CudaGraph {
    graph: cudarc::driver::CudaGraph,
    exec: cudarc::driver::CudaGraphExec,
}

/// Capture GPU operations into a replayable graph.
/// All GPU ops inside the closure are recorded, not executed.
/// The graph is instantiated into an executable on return.
pub fn capture<F, R>(f: F) -> Result<(R, CudaGraph), FerrotorchError>
where
    F: FnOnce() -> R,
{
    // 1. Begin capture on the current stream
    // 2. Execute closure (ops are recorded, not launched)
    // 3. End capture, instantiate graph executable
    // 4. Return result and CudaGraph handle
}

impl CudaGraph {
    /// Replay the captured graph. All operations execute with a single
    /// kernel launch submission, reducing CPU overhead.
    pub fn replay(&self) -> Result<(), FerrotorchError>;

    /// Update graph node parameters (e.g., new input data pointers)
    /// without re-capturing. Returns error if shapes changed.
    pub fn update(&mut self) -> Result<(), FerrotorchError>;
}
```

Graphs are invalidated if tensor shapes change between captures. The typical pattern is: warm up with one eager execution, capture, then replay in a training loop.

### Benchmark Mode (`cuda/benchmark.rs`)

cuDNN algorithm auto-tuning for convolutions. When enabled, the first convolution with each unique input shape benchmarks all available algorithms and caches the fastest.

```rust
/// Enable or disable cuDNN benchmark mode.
/// When true, convolution algorithm selection benchmarks all candidates
/// and caches the fastest. When false, uses cuDNN heuristic selection.
/// Matches `torch.backends.cudnn.benchmark` behavior.
pub fn benchmark_mode(enabled: bool);

/// Check if benchmark mode is currently enabled.
pub fn is_benchmark_mode() -> bool;

/// Benchmark cache — maps (input_shape, filter_shape, params) to the
/// algorithm that benchmarked fastest.
static BENCHMARK_CACHE: OnceLock<Mutex<HashMap<ConvKey, ConvAlgorithm>>> = OnceLock::new();
```

When benchmark mode is on, `gpu_conv2d_forward` calls `cudnnFindConvolutionForwardAlgorithm` (which times all algorithms) instead of `cudnnGetConvolutionForwardAlgorithm_v7` (which uses a heuristic). The result is cached in `BENCHMARK_CACHE`. This matches PyTorch's behavior: setting `torch.backends.cudnn.benchmark = True` enables exhaustive search on the first call, then reuses the result.

Note: benchmark mode is beneficial when input shapes are fixed (e.g., fixed batch size and image dimensions). When shapes vary per batch, the repeated benchmarking overhead outweighs the gains.

### Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `ferrotorch-core` | workspace | `Tensor<T>`, `StorageBuffer<T>`, `Device`, `FerrotorchError` |
| `cudarc` | 0.16 | Rust bindings to CUDA driver API, cuBLAS, cuDNN, cuFFT |
| `half` | 2.4 | f16 type for half-precision kernel support |

### Build Requirements

- NVIDIA GPU with compute capability 7.0+ (Volta and later) for f16 tensor core support
- CUDA toolkit 12.0+ installed (for nvcc, cuBLAS, cuDNN, cuFFT)
- `nvcc` on PATH for PTX compilation (build.rs), or precompiled PTX shipped in the repository

## Resolved Questions

### Q1: Metal backend timeline
**Decision: Phase 6.1 — add after CUDA is stable.** PyTorch added MPS (Metal Performance Shaders) as a second backend. We follow the same pattern: CUDA first, Metal as a follow-up milestone. The `Device` enum already has space for it. The `StorageBuffer` pattern (enum variants) makes adding backends additive. Metal support (Apple Silicon) will add a `Metal(MetalBuffer<T>)` variant to `StorageBuffer<T>` and a parallel `metal/` directory mirroring the `cuda/` structure, using the `metal-rs` crate for Rust bindings.

### Q2: Vulkan compute backend timeline
**Decision: Out of scope for Phase 6.** PyTorch's Vulkan backend is primarily for mobile inference, not a priority for a training framework. Vulkan compute via `vulkano` or `ash` would enable GPU acceleration on non-NVIDIA hardware (AMD, Intel), but the kernel portability challenge is significant — SPIR-V shaders are a different compilation target from PTX. Revisit after Metal (Phase 6.1) ships.

### Q3: CubeCL as a portable kernel language
**Decision: Do not use CubeCL.** PyTorch writes its own kernels. CubeCL is Burn's approach. We use cudarc for CUDA bindings and write our own kernels. This gives us full control over kernel implementations and avoids a heavy abstraction layer. Hand-written `.cu` kernels compiled to PTX via nvcc remain the approach. This also means our performance characteristics are fully under our control and we can tune kernels to match cuBLAS/cuDNN behavior exactly.

### Q4: f16/bf16 mixed-precision training scope
**Decision: Include in Phase 6.** Mixed-precision training is essential for practical GPU training — without it, GPU training is 2-3x slower than it needs to be. PyTorch's `torch.cuda.amp` is a core part of the CUDA experience, not an optional extra. The CUDA backend supports f16 via cuBLAS `cublasHgemm` and custom f16 kernels. bf16 support depends on cuBLAS `cublasBf16Gemm` (compute capability 8.0+, Ampere). Phase 6 includes:
- `autocast(|| { ... })` context manager that automatically casts eligible ops to f16/bf16
- `GradScaler` that scales loss, unscales gradients, and handles inf/nan detection
- Policy-based op casting: matmul/conv in f16, reductions/norms in f32 (matching PyTorch's default policy)

## Out of Scope
- Metal backend implementation — Phase 6.1, after CUDA is stable (Resolved Q1)
- Vulkan compute backend implementation — out of scope entirely, revisit after Metal (Resolved Q2)
- CubeCL portable kernel integration — rejected, we write our own kernels (Resolved Q3)
- Multi-GPU collective operations (allreduce, broadcast) — that is Phase 7 (ferrotorch-distributed)
- TensorRT or CUDA kernel fusion — that is Phase 8 (ferrotorch-jit)
- Custom user-defined CUDA kernels via a public plugin API — defer to post-Phase 8
- Windows CUDA support — Linux is the primary target; Windows support is a build system concern, not an architectural one
