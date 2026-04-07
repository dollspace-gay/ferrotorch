//! CUDA graph capture and replay infrastructure.
//!
//! A CUDA graph records a sequence of GPU operations (kernel launches, memcpys)
//! and replays them as a single driver submission. This eliminates per-kernel
//! launch overhead (~70μs on WSL2, ~5μs on native Linux per call) by collapsing
//! hundreds of launches into one.
//!
//! # Usage
//!
//! ```ignore
//! use ferrotorch_gpu::graph::{DeviceScalar, begin_capture, end_capture};
//!
//! // Pre-allocate all buffers BEFORE capture
//! let mut out = alloc_zeros_f32(768, &device)?;
//!
//! // Parameters that change between replays go in DeviceScalar
//! let mut pos = DeviceScalar::new(device.stream(), 0u32)?;
//!
//! // Capture
//! begin_capture(device.stream())?;
//! gpu_add_into(&a, &b, &mut out, &device)?;  // recorded, not executed
//! let graph = end_capture(device.stream())?;
//!
//! // Replay loop
//! for i in 0..100 {
//!     pos.update(i as u32)?;  // memcpy before replay
//!     graph.launch()?;         // replay all captured ops
//! }
//! ```

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits};

use crate::error::{GpuError, GpuResult};

// ---------------------------------------------------------------------------
// DeviceScalar — a single value in GPU memory, updatable before graph replay
// ---------------------------------------------------------------------------

/// A single scalar value stored in GPU device memory.
///
/// Used for CUDA graph capture: the graph records the device pointer (fixed
/// address), and the caller updates the value via [`update`](DeviceScalar::update)
/// before each [`CapturedGraph::launch`]. The update is a 4-or-8 byte
/// `cuMemcpyHtoDAsync` — effectively zero cost.
#[cfg(feature = "cuda")]
pub struct DeviceScalar<T: DeviceRepr + ValidAsZeroBits + Copy> {
    buf: CudaSlice<T>,
    stream: Arc<CudaStream>,
}

#[cfg(feature = "cuda")]
impl<T: DeviceRepr + ValidAsZeroBits + Copy> DeviceScalar<T> {
    /// Allocate a device scalar with the given initial value.
    pub fn new(stream: &Arc<CudaStream>, initial: T) -> GpuResult<Self> {
        let buf = stream.clone_htod(&[initial])?;
        Ok(Self {
            buf,
            stream: Arc::clone(stream),
        })
    }

    /// Update the device value. This is an async H→D memcpy of `size_of::<T>()`
    /// bytes. Must be called on the same stream as the graph to ensure ordering.
    pub fn update(&mut self, value: T) -> GpuResult<()> {
        self.stream.memcpy_htod(&[value], &mut self.buf)?;
        Ok(())
    }

    /// Borrow the underlying `CudaSlice` for use as a kernel parameter.
    /// The graph captures this pointer address; updating the value later
    /// changes what the kernel reads without re-capturing.
    #[inline]
    pub fn inner(&self) -> &CudaSlice<T> {
        &self.buf
    }
}

// ---------------------------------------------------------------------------
// CapturedGraph — a replayable CUDA graph
// ---------------------------------------------------------------------------

/// A captured and instantiated CUDA graph that can be replayed with
/// [`launch`](CapturedGraph::launch).
///
/// Created via [`begin_capture`] + GPU ops + [`end_capture`].
/// The graph holds references to all device memory used during capture.
/// Those buffers must remain allocated for the lifetime of the graph.
///
/// **Allocator pool integration (CL-278).** When created via
/// [`end_capture_with_pool`], the graph holds a strong reference to
/// the [`CapturePool`] that recorded its allocations. The pool keeps
/// every registered buffer alive until the last `CapturedGraph`
/// referencing it is dropped, which guarantees the device pointers
/// recorded in the graph remain valid across replays. Without the
/// pool, callers must manually keep buffers alive (the original
/// [`end_capture`] API).
#[cfg(feature = "cuda")]
pub struct CapturedGraph {
    graph: cudarc::driver::CudaGraph,
    /// Optional reference to the pool that owns the graph's
    /// allocations. Some(pool) when constructed via
    /// [`end_capture_with_pool`]. Dropping the graph drops this
    /// Arc, which (if it's the last reference) drops every buffer
    /// the pool holds. CL-278.
    pool: Option<Arc<CapturePool>>,
}

#[cfg(feature = "cuda")]
impl CapturedGraph {
    /// Replay all operations captured in this graph.
    ///
    /// Before calling this, update any [`DeviceScalar`] values and perform
    /// any pre-launch memcpys (e.g., position embeddings). All updates must
    /// be on the same stream the graph was captured on.
    pub fn launch(&self) -> GpuResult<()> {
        self.graph.launch()?;
        Ok(())
    }

    /// Number of buffers held alive by this graph's allocator pool.
    /// Returns 0 if the graph was created without a pool. CL-278.
    pub fn pool_buffer_count(&self) -> usize {
        self.pool
            .as_ref()
            .map(|p| p.buffer_count())
            .unwrap_or(0)
    }

    /// True if this graph holds a CapturePool reference. CL-278.
    pub fn has_pool(&self) -> bool {
        self.pool.is_some()
    }
}

// ---------------------------------------------------------------------------
// Capture API
// ---------------------------------------------------------------------------

/// Begin CUDA graph capture on the given stream.
///
/// All GPU operations (kernel launches, cuBLAS calls, memcpys) issued on this
/// stream after this call are **recorded but not executed**. Call
/// [`end_capture`] to finalize and instantiate the graph.
///
/// # Requirements
///
/// - All output buffers must be pre-allocated before capture begins.
/// - No `alloc_zeros` / `cpu_to_gpu` during capture (use `_into` variants).
/// - No CPU↔GPU synchronization during capture.
/// - Event tracking should be disabled during capture to avoid interference
///   (call `ctx.disable_event_tracking()` before, re-enable after).
#[cfg(feature = "cuda")]
pub fn begin_capture(stream: &Arc<CudaStream>) -> GpuResult<()> {
    stream.begin_capture(
        cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
    )?;
    Ok(())
}

/// End CUDA graph capture, instantiate, and return the replayable graph.
///
/// Returns `Err` if capture was not active or if instantiation fails.
///
/// The returned graph has no [`CapturePool`] attached. The caller is
/// responsible for keeping the buffers used by the captured kernels
/// alive for the graph's lifetime. Use [`end_capture_with_pool`]
/// for the lifetime-managed variant.
#[cfg(feature = "cuda")]
pub fn end_capture(stream: &Arc<CudaStream>) -> GpuResult<CapturedGraph> {
    let flags = cudarc::driver::sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
    let graph = stream
        .end_capture(flags)?
        .ok_or(GpuError::PtxCompileFailed {
            kernel: "CUDA graph capture returned null",
        })?;
    Ok(CapturedGraph { graph, pool: None })
}

/// End CUDA graph capture and attach a [`CapturePool`] reference to
/// the resulting [`CapturedGraph`]. CL-278.
///
/// The pool's tracked buffers are kept alive for the lifetime of the
/// returned graph: dropping the graph drops its `Arc<CapturePool>`,
/// which (if it's the last reference) drops every buffer the pool
/// recorded. This guarantees that the device pointers recorded in
/// the captured graph remain valid across replays.
///
/// Use this in concert with [`CapturePool::record_buffer`]: allocate
/// every buffer used during capture before calling `begin_capture`,
/// register each one with the pool, run the kernels under capture,
/// then call `end_capture_with_pool(stream, pool)` to seal the
/// lifetime relationship.
#[cfg(feature = "cuda")]
pub fn end_capture_with_pool(
    stream: &Arc<CudaStream>,
    pool: Arc<CapturePool>,
) -> GpuResult<CapturedGraph> {
    let mut graph = end_capture(stream)?;
    graph.pool = Some(pool);
    Ok(graph)
}

// ---------------------------------------------------------------------------
// CapturePool — memory pool for graph capture
// ---------------------------------------------------------------------------

/// A dedicated memory pool for CUDA graph capture.
///
/// Two responsibilities:
///
/// 1. **Sealed flag** — gates [`begin_capture_with_pool`] so the
///    caller can express "no more allocations after this point"
///    semantically. Sealed pools cannot satisfy new allocations
///    during capture.
///
/// 2. **Buffer lifetime tracking (CL-278)** — registered buffers
///    are kept alive by the pool itself, so they outlive any
///    [`CapturedGraph`] that holds an `Arc<CapturePool>`. Dropping
///    the graph drops the Arc, and dropping the last Arc drops
///    every registered buffer in registration order.
///
/// # Usage
///
/// ```ignore
/// use std::sync::Arc;
/// let pool = Arc::new(CapturePool::new());
///
/// // Allocate every buffer the captured kernels will read or
/// // write, and register each one with the pool so it stays alive
/// // for the graph's lifetime.
/// let mut buf_a = alloc_zeros_f32(1024, &device)?;
/// let mut buf_b = alloc_zeros_f32(1024, &device)?;
/// pool.record_buffer(buf_a.try_clone()?);
/// pool.record_buffer(buf_b.try_clone()?);
///
/// pool.seal();
/// begin_capture_with_pool(&pool, stream)?;
/// // ... launch kernels using buf_a and buf_b ...
/// let graph = end_capture_with_pool(stream, Arc::clone(&pool))?;
/// // Dropping `pool` here is safe — the graph holds its own Arc.
/// ```
#[cfg(feature = "cuda")]
pub struct CapturePool {
    sealed: std::sync::atomic::AtomicBool,
    /// Registered buffers (type-erased) kept alive for the graph's
    /// lifetime. Each entry is a Box<dyn Any + Send + Sync> wrapping
    /// the buffer's drop guard. CL-278.
    buffers: std::sync::Mutex<Vec<Box<dyn std::any::Any + Send + Sync + 'static>>>,
}

#[cfg(feature = "cuda")]
impl CapturePool {
    /// Create a new, unsealed capture pool.
    pub fn new() -> Self {
        Self {
            sealed: std::sync::atomic::AtomicBool::new(false),
            buffers: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Seal the pool, preventing any further allocations.
    pub fn seal(&self) {
        self.sealed
            .store(true, std::sync::atomic::Ordering::Release);
    }

    /// Unseal the pool, allowing allocations again.
    pub fn unseal(&self) {
        self.sealed
            .store(false, std::sync::atomic::Ordering::Release);
    }

    /// Check whether the pool is sealed.
    pub fn is_capture_pool_sealed(&self) -> bool {
        self.sealed.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Register a buffer with the pool so it stays alive for the
    /// lifetime of any [`CapturedGraph`] that holds this pool.
    /// CL-278.
    ///
    /// `buffer` can be any type that owns GPU memory (typically
    /// `CudaBuffer<f32>`, `CudaBuffer<f64>`, or `Arc<CudaBuffer<T>>`).
    /// The pool stores it in a type-erased `Box<dyn Any + Send +
    /// Sync>` and drops it (in registration order) when the pool
    /// itself is dropped.
    ///
    /// Returns the index of the registered buffer for diagnostic
    /// purposes.
    pub fn record_buffer<B>(&self, buffer: B) -> usize
    where
        B: Send + Sync + 'static,
    {
        let mut guard = self
            .buffers
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let idx = guard.len();
        guard.push(Box::new(buffer));
        idx
    }

    /// Number of buffers currently registered with the pool. CL-278.
    pub fn buffer_count(&self) -> usize {
        self.buffers
            .lock()
            .map(|g| g.len())
            .unwrap_or(0)
    }

    /// Drop every registered buffer immediately, in registration
    /// order. The pool itself remains usable; new buffers can still
    /// be registered after this call. CL-278.
    ///
    /// Use this when reusing a pool across multiple capture cycles.
    /// Calling clear while a [`CapturedGraph`] still holds an Arc
    /// to this pool is safe — the graph's strong reference keeps
    /// the pool struct alive, but the buffer slots are reset.
    pub fn clear_buffers(&self) {
        let mut guard = self
            .buffers
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        guard.clear();
    }
}

#[cfg(feature = "cuda")]
impl Default for CapturePool {
    fn default() -> Self {
        Self::new()
    }
}

/// Begin CUDA graph capture with a capture pool.
///
/// Like [`begin_capture`], but checks that the capture pool is not sealed
/// before starting capture. A sealed pool cannot satisfy allocations
/// during graph capture, which would cause CUDA errors.
///
/// # Errors
///
/// Returns [`GpuError::InvalidArgument`](GpuError) if the pool is sealed.
/// Returns a CUDA driver error if `begin_capture` fails.
#[cfg(feature = "cuda")]
pub fn begin_capture_with_pool(pool: &CapturePool, stream: &Arc<CudaStream>) -> GpuResult<()> {
    if pool.is_capture_pool_sealed() {
        return Err(GpuError::InvalidState {
            message: "cannot begin graph capture: capture pool is sealed".into(),
        });
    }
    begin_capture(stream)
}

/// Stub CapturePool when cuda feature is disabled. Provides the
/// same surface API as the cuda-enabled type so callers compile on
/// both feature configurations.
#[cfg(not(feature = "cuda"))]
pub struct CapturePool;

#[cfg(not(feature = "cuda"))]
impl CapturePool {
    /// Create an empty CapturePool. Without the cuda feature the
    /// pool has no internal state to initialize.
    pub fn new() -> Self {
        Self
    }

    /// No-op without the cuda feature: there is no real CUDA pool
    /// to seal because no real allocations can happen.
    pub fn seal(&self) {
        // Without the cuda feature there is no allocator state to
        // mutate; the CapturePool exists only so callers can write
        // feature-portable code.
    }

    /// No-op without the cuda feature: there is no real CUDA pool
    /// to unseal because no real allocations can happen.
    pub fn unseal(&self) {
        // Without the cuda feature there is no allocator state to
        // mutate; the CapturePool exists only so callers can write
        // feature-portable code.
    }

    /// Always returns `false` without the cuda feature since there
    /// is no real pool that could be in either state.
    pub fn is_capture_pool_sealed(&self) -> bool {
        false
    }

    /// Always returns 0 without the cuda feature since no real
    /// allocations can be tracked. CL-278.
    pub fn buffer_count(&self) -> usize {
        0
    }
}

#[cfg(not(feature = "cuda"))]
impl Default for CapturePool {
    fn default() -> Self {
        Self::new()
    }
}

/// Stub begin_capture_with_pool when cuda feature is disabled.
#[cfg(not(feature = "cuda"))]
pub fn begin_capture_with_pool<T>(_pool: &CapturePool, _stream: &T) -> GpuResult<()> {
    Err(GpuError::NoCudaFeature)
}

// ---------------------------------------------------------------------------
// Stubs when cuda feature is disabled
// ---------------------------------------------------------------------------

/// Stub DeviceScalar.
#[cfg(not(feature = "cuda"))]
pub struct DeviceScalar<T: Copy> {
    _phantom: std::marker::PhantomData<T>,
}

/// Stub CapturedGraph.
#[cfg(not(feature = "cuda"))]
pub struct CapturedGraph;

#[cfg(not(feature = "cuda"))]
impl CapturedGraph {
    pub fn launch(&self) -> GpuResult<()> {
        Err(GpuError::NoCudaFeature)
    }
}

#[cfg(not(feature = "cuda"))]
pub fn begin_capture<T>(_stream: &T) -> GpuResult<()> {
    Err(GpuError::NoCudaFeature)
}

#[cfg(not(feature = "cuda"))]
pub fn end_capture<T>(_stream: &T) -> GpuResult<CapturedGraph> {
    Err(GpuError::NoCudaFeature)
}

/// Stub `end_capture_with_pool` when the cuda feature is not enabled.
/// CL-278.
#[cfg(not(feature = "cuda"))]
pub fn end_capture_with_pool<T>(
    _stream: &T,
    _pool: std::sync::Arc<CapturePool>,
) -> GpuResult<CapturedGraph> {
    Err(GpuError::NoCudaFeature)
}

// ---------------------------------------------------------------------------
// Tests — CL-278 capture pool buffer tracking
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    #[test]
    fn capture_pool_buffer_count_starts_at_zero() {
        let pool = CapturePool::new();
        assert_eq!(pool.buffer_count(), 0);
    }

    #[test]
    fn capture_pool_record_buffer_increments_count() {
        let pool = CapturePool::new();
        let buf_a: Vec<f32> = vec![0.0; 10];
        let idx = pool.record_buffer(buf_a);
        assert_eq!(idx, 0);
        assert_eq!(pool.buffer_count(), 1);

        let buf_b: Vec<f64> = vec![0.0; 5];
        let idx = pool.record_buffer(buf_b);
        assert_eq!(idx, 1);
        assert_eq!(pool.buffer_count(), 2);
    }

    #[test]
    fn capture_pool_clear_buffers_resets_count_but_keeps_pool() {
        let pool = CapturePool::new();
        pool.record_buffer(vec![0u8; 16]);
        pool.record_buffer(vec![0u8; 32]);
        assert_eq!(pool.buffer_count(), 2);
        pool.clear_buffers();
        assert_eq!(pool.buffer_count(), 0);
        // Pool is still usable.
        pool.record_buffer(vec![0u8; 8]);
        assert_eq!(pool.buffer_count(), 1);
    }

    #[test]
    fn capture_pool_drop_releases_registered_buffers() {
        // Use Arc to detect when the inner buffer is dropped.
        let buf = Arc::new(vec![1.0f32, 2.0, 3.0]);
        let pool = CapturePool::new();
        pool.record_buffer(Arc::clone(&buf));
        assert_eq!(Arc::strong_count(&buf), 2);
        drop(pool);
        // Pool dropped → recorded Arc dropped → strong count back to 1.
        assert_eq!(Arc::strong_count(&buf), 1);
    }

    #[test]
    fn capture_pool_records_heterogeneous_types() {
        let pool = CapturePool::new();
        pool.record_buffer(vec![0.0f32; 4]);
        pool.record_buffer(vec![0.0f64; 4]);
        pool.record_buffer(vec![0u8; 4]);
        pool.record_buffer(Arc::new(42i32));
        assert_eq!(pool.buffer_count(), 4);
    }

    #[test]
    fn capture_pool_seal_unseal() {
        let pool = CapturePool::new();
        assert!(!pool.is_capture_pool_sealed());
        pool.seal();
        assert!(pool.is_capture_pool_sealed());
        pool.unseal();
        assert!(!pool.is_capture_pool_sealed());
    }
}
