# Feature: Unified Device-Aware Tensor Architecture

## Summary
Merge GPU and CPU tensors into a single `Tensor<T>` type that is device-aware internally. Operations dispatch to CPU or GPU kernels based on `tensor.device()`. `Module<T>`, autograd, and all existing APIs work unchanged on both devices. This eliminates `GpuTensor<T>` as a separate type and makes `model.forward(&gpu_tensor)` work out of the box.

## The Problem (User Bug Report)

1. `GpuTensor<T>` and `Tensor<T>` are completely separate types with no shared trait
2. `Module<T>` is hardcoded to `Tensor<T>` — can't run on GPU
3. GPU ops are standalone free functions, not methods on Tensor
4. No autograd on GPU — `GpuTensor` has no `backward()`, `grad()`, `requires_grad`

## Design

### Core Change: `StorageBuffer<T>` Gets a Cuda Variant

```rust
// ferrotorch-core/src/storage.rs

pub enum StorageBuffer<T: Element> {
    Cpu(Vec<T>),
    #[cfg(feature = "cuda")]
    Cuda {
        /// Opaque handle to GPU memory. Stored as raw bytes to avoid
        /// ferrotorch-core depending on cudarc directly.
        handle: GpuBufferHandle,
        /// Number of elements.
        len: usize,
    },
}
```

**The dependency problem**: ferrotorch-core can't depend on ferrotorch-gpu (circular). The solution: ferrotorch-core defines an **opaque GPU buffer handle** and a **dispatch trait**. ferrotorch-gpu registers its implementation at runtime.

```rust
// ferrotorch-core/src/gpu_dispatch.rs

/// Opaque handle to GPU memory. ferrotorch-core doesn't know what's inside —
/// ferrotorch-gpu provides the implementation.
pub struct GpuBufferHandle {
    /// Type-erased pointer to the actual GPU buffer (CudaBuffer<T> etc.)
    inner: Box<dyn std::any::Any + Send + Sync>,
    /// Device ordinal
    device_ordinal: usize,
}

/// Trait that GPU backends register to handle operations on GPU tensors.
/// ferrotorch-core calls these; ferrotorch-gpu implements them.
pub trait GpuBackend: Send + Sync {
    /// Allocate GPU memory and copy CPU data to device.
    fn cpu_to_gpu(&self, data: &[u8], elem_size: usize, device: usize) -> Result<GpuBufferHandle, FerrotorchError>;

    /// Copy GPU data back to CPU.
    fn gpu_to_cpu(&self, handle: &GpuBufferHandle, len: usize, elem_size: usize) -> Result<Vec<u8>, FerrotorchError>;

    /// GPU elementwise add (f32).
    fn add_f32(&self, a: &GpuBufferHandle, b: &GpuBufferHandle, len: usize) -> Result<GpuBufferHandle, FerrotorchError>;

    /// GPU elementwise mul (f32).
    fn mul_f32(&self, a: &GpuBufferHandle, b: &GpuBufferHandle, len: usize) -> Result<GpuBufferHandle, FerrotorchError>;

    /// GPU matmul (f32).
    fn matmul_f32(&self, a: &GpuBufferHandle, b: &GpuBufferHandle, m: usize, k: usize, n: usize) -> Result<GpuBufferHandle, FerrotorchError>;

    /// GPU relu (f32).
    fn relu_f32(&self, a: &GpuBufferHandle, len: usize) -> Result<GpuBufferHandle, FerrotorchError>;

    // ... one method per GPU-accelerated op
}

/// Global GPU backend registry. ferrotorch-gpu registers itself on init.
static GPU_BACKEND: OnceLock<Box<dyn GpuBackend>> = OnceLock::new();

pub fn register_gpu_backend(backend: Box<dyn GpuBackend>) { ... }
pub fn gpu_backend() -> Option<&'static dyn GpuBackend> { ... }
```

### How `Tensor<T>` Changes

```rust
// tensor.rs — minimal changes

impl<T: Float> Tensor<T> {
    /// Move tensor to a device. Returns a new tensor with data on target device.
    pub fn to(&self, device: Device) -> FerrotorchResult<Tensor<T>> {
        if self.device() == device {
            return Ok(self.clone());  // Already there — cheap Arc clone
        }
        match (self.device(), device) {
            (Device::Cpu, Device::Cuda(ordinal)) => {
                let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
                let data = self.data()?;  // &[T] — only works for CPU
                let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * std::mem::size_of::<T>()) };
                let handle = backend.cpu_to_gpu(bytes, std::mem::size_of::<T>(), ordinal)?;
                let storage = TensorStorage { data: StorageBuffer::Cuda { handle, len: self.numel() }, device };
                Tensor::from_storage(storage, self.shape().to_vec(), self.requires_grad())
            }
            (Device::Cuda(_), Device::Cpu) => {
                let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
                let handle = self.gpu_handle()?;
                let bytes = backend.gpu_to_cpu(handle, self.numel(), std::mem::size_of::<T>())?;
                let data: Vec<T> = unsafe { /* transmute bytes to Vec<T> */ };
                Tensor::from_storage(TensorStorage::cpu(data), self.shape().to_vec(), self.requires_grad())
            }
            _ => Err(FerrotorchError::InvalidArgument { message: "unsupported device transfer".into() })
        }
    }

    pub fn cuda(&self) -> FerrotorchResult<Tensor<T>> { self.to(Device::Cuda(0)) }
    pub fn cpu(&self) -> FerrotorchResult<Tensor<T>> { self.to(Device::Cpu) }

    /// Get CPU data slice. Returns Err if tensor is on GPU.
    pub fn data(&self) -> FerrotorchResult<&[T]> {
        match &self.inner.storage.data {
            StorageBuffer::Cpu(v) => Ok(&v[self.inner.offset..]),
            #[cfg(feature = "cuda")]
            StorageBuffer::Cuda { .. } => Err(FerrotorchError::InvalidArgument {
                message: "cannot access GPU tensor data as CPU slice — call .cpu() first".into()
            }),
        }
    }

    /// Get the opaque GPU buffer handle. Returns Err if tensor is on CPU.
    pub fn gpu_handle(&self) -> FerrotorchResult<&GpuBufferHandle> { ... }
}
```

### How Operations Dispatch

The key change in every grad_fn and op: check device, dispatch to GPU or CPU.

```rust
// grad_fns/arithmetic.rs — add()

pub fn add<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    // Device check
    if a.device() != b.device() {
        return Err(FerrotorchError::DeviceMismatch { expected: a.device(), got: b.device() });
    }

    let result = match a.device() {
        Device::Cpu => {
            // Existing CPU path — unchanged
            fast_add(a, b)?
        }
        Device::Cuda(_) => {
            // GPU dispatch via the registered backend
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let handle = backend.add_f32(a.gpu_handle()?, b.gpu_handle()?, a.numel())?;
            let storage = TensorStorage { data: StorageBuffer::Cuda { handle, len: a.numel() }, device: a.device() };
            Tensor::from_storage(storage, a.shape().to_vec(), false)?
        }
    };

    // Autograd — SAME for both CPU and GPU
    if needs_grad(a, b) {
        Tensor::from_operation(
            TensorStorage { data: result.inner.storage.data.clone(), device: result.device() },
            result.shape().to_vec(),
            Arc::new(AddBackward { a: a.clone(), b: b.clone() }),
        )
    } else {
        Ok(result)
    }
}
```

### How Autograd Works on GPU

**It already works.** The autograd engine operates on `Tensor<T>` nodes connected by `GradFn` edges. It doesn't care what device the data is on — it calls `GradFn::backward(grad_output)` which returns gradient tensors. As long as `backward()` dispatches ops correctly (CPU or GPU), the autograd graph traversal is device-agnostic.

The gradient tensors will be on the same device as the forward tensors because the backward ops (add, mul, matmul) dispatch based on `grad_output.device()`.

### What `GpuTensor<T>` Becomes

`GpuTensor<T>` is **deleted**. Its functionality moves into:
- `StorageBuffer::Cuda` — holds the GPU buffer
- `GpuBackend` trait — provides the GPU operations
- `Tensor::to()` / `Tensor::cuda()` / `Tensor::cpu()` — device transfers

The `ferrotorch-gpu` crate becomes the `GpuBackend` implementation:
```rust
// ferrotorch-gpu/src/backend_impl.rs

pub struct CudaBackend {
    devices: Vec<Arc<GpuDevice>>,
}

impl GpuBackend for CudaBackend {
    fn add_f32(&self, a: &GpuBufferHandle, b: &GpuBufferHandle, len: usize) -> Result<GpuBufferHandle, FerrotorchError> {
        // Extract CudaBuffer from handle, call gpu_add, wrap result
    }
    fn matmul_f32(&self, a: &GpuBufferHandle, b: &GpuBufferHandle, m: usize, k: usize, n: usize) -> Result<GpuBufferHandle, FerrotorchError> {
        // Extract CudaBuffer, call gpu_matmul_f32, wrap result
    }
    // ...
}

// On crate init:
pub fn init() {
    let backend = CudaBackend::new();
    ferrotorch_core::register_gpu_backend(Box::new(backend));
}
```

### What Module<T> Gets for Free

**Nothing changes in Module<T>.** Because `Tensor<T>` is now device-aware:

```rust
let model = Linear::new(784, 10, true)?;    // Creates CPU parameters
let x = rand::<f32>(&[32, 784])?.cuda()?;   // Input on GPU

// This Just Works™ — forward dispatches ops to GPU
let y = model.forward(&x)?;  // Output on GPU
let loss = y.sum_all()?;     // Sum on GPU
loss.backward()?;            // Backward on GPU

// Gradients are on GPU too
let grad = model.parameters()[0].grad()?;    // GPU gradient
```

Wait — there's a subtlety. The model's **parameters** (weight, bias) are on CPU. The **input** is on GPU. The matmul `input @ weight.T` has mismatched devices and will error.

**Solution**: `model.to(device)` must move all parameters to the target device:

```rust
impl<T: Float> dyn Module<T> {
    fn to(&mut self, device: Device) -> FerrotorchResult<()> {
        for param in self.parameters_mut() {
            *param = Parameter::new(param.tensor().to(device)?);
        }
        Ok(())
    }
}
```

Or add `to()` to the Module trait itself.

### Implementation Plan (Ordered by Dependency)

**Step 1: Core infrastructure** (ferrotorch-core)
- Add `GpuBufferHandle`, `GpuBackend` trait, global registry to `gpu_dispatch.rs`
- Extend `StorageBuffer<T>` with `Cuda` variant (feature-gated)
- Add `Tensor::to()`, `Tensor::cuda()`, `Tensor::cpu()`
- Add `Tensor::gpu_handle()`, `Tensor::is_cpu()`, `Tensor::is_cuda()`
- Make `Tensor::data()` return `Err` for GPU tensors
- Add `DeviceUnavailable` error variant

**Step 2: Op dispatch** (ferrotorch-core grad_fns)
- Add device check + GPU dispatch to each operation:
  - arithmetic: add, sub, mul, div, neg (5 ops)
  - linalg: mm, matmul, bmm (3 ops)
  - activation: relu, sigmoid, tanh, gelu, silu, softmax (6 ops)
  - reduction: sum, mean (2 ops)
- Pattern: `match a.device() { Cpu => existing_path, Cuda => backend.op() }`

**Step 3: Backend implementation** (ferrotorch-gpu)
- Implement `GpuBackend` for `CudaBackend`
- Bridge existing `gpu_add`, `gpu_matmul_f32`, etc. to the trait methods
- Register backend on `init()`
- Delete `GpuTensor<T>` (deprecated — keep for one release)

**Step 4: Module device support** (ferrotorch-nn)
- Add `Module::to(&mut self, device)` default method
- Add `Parameter::to(device)` helper
- Test: `model.to(Device::Cuda(0))` moves all weights to GPU

**Step 5: Integration tests**
- `Linear` forward on GPU produces correct output
- Backward on GPU produces correct gradients
- Full training step on GPU (forward → loss → backward → optimizer step)
- Mixed device error (CPU tensor + GPU tensor → DeviceMismatch)

### What Doesn't Change

- `Module<T>` trait signature — `forward(&self, &Tensor<T>) -> FerrotorchResult<Tensor<T>>`
- Autograd engine (`graph.rs`) — walks `Tensor<T>` nodes regardless of device
- `GradFn<T>` trait — `backward(&self, &Tensor<T>) -> Vec<Option<Tensor<T>>>`
- All CPU-only code paths — they work exactly as before
- `#[derive(Module)]` — no changes needed
- DataLoader, transforms, etc. — device-agnostic
- JIT tracing and IR — traces `Tensor<T>` ops regardless of device

### Risks and Mitigations

**Risk**: 190 `.data()` calls in ferrotorch-core break for GPU tensors.
**Mitigation**: `.data()` returns `Err` for GPU tensors. Backward ops that need raw data on GPU call through the `GpuBackend` trait instead. Most backward ops compute new tensors from other ops (add, mul) which auto-dispatch.

**Risk**: Performance regression from device dispatch checks on every op.
**Mitigation**: The check is a single `match` on an enum discriminant — essentially free (branch prediction handles it). The CPU hot path doesn't change.

**Risk**: Circular dependency between core and gpu crates.
**Mitigation**: Core defines the trait (`GpuBackend`) and handle (`GpuBufferHandle`). GPU implements the trait. No circular dependency — GPU depends on Core, Core doesn't depend on GPU.

### Estimated Effort

| Step | Files | Scope |
|------|-------|-------|
| 1 | 3 new files, 2 modified | Core infra |
| 2 | ~15 grad_fn files modified | Add match arms |
| 3 | 2 new files, delete GpuTensor | Backend bridge |
| 4 | 2 files modified | Module.to() |
| 5 | 1 new test file | Integration |

Total: ~25 files touched. The individual changes per file are small (add a `match` arm), but there are many files.
