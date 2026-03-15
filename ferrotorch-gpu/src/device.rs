//! CUDA device management.
//!
//! [`GpuDevice`] wraps a `cudarc::driver::CudaContext` and its default stream,
//! providing a safe, ergonomic entry point for all GPU operations.

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaStream};

use crate::error::GpuResult;
#[cfg(not(feature = "cuda"))]
use crate::error::GpuError;

/// Handle to a single CUDA GPU device.
///
/// Create one per device ordinal. The underlying CUDA context and default
/// stream are reference-counted, so cloning a `GpuDevice` is cheap.
#[cfg(feature = "cuda")]
pub struct GpuDevice {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    ordinal: usize,
}

#[cfg(feature = "cuda")]
impl GpuDevice {
    /// Initialize a CUDA device by ordinal (0-indexed).
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::Driver`] if the CUDA driver cannot create a context
    /// for the requested device (e.g. ordinal out of range, driver not loaded).
    pub fn new(ordinal: usize) -> GpuResult<Self> {
        let ctx = CudaContext::new(ordinal)?;
        let stream = ctx.default_stream();
        Ok(Self {
            ctx,
            stream,
            ordinal,
        })
    }

    /// The underlying CUDA context (reference-counted).
    #[inline]
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// The default CUDA stream for this device.
    #[inline]
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// The device ordinal (0-indexed).
    #[inline]
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }
}

#[cfg(feature = "cuda")]
impl Clone for GpuDevice {
    fn clone(&self) -> Self {
        Self {
            ctx: Arc::clone(&self.ctx),
            stream: Arc::clone(&self.stream),
            ordinal: self.ordinal,
        }
    }
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for GpuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuDevice")
            .field("ordinal", &self.ordinal)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// Stub when `cuda` feature is disabled
// ---------------------------------------------------------------------------

/// Stub `GpuDevice` when the `cuda` feature is not enabled.
///
/// Every method returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
#[derive(Clone, Debug)]
pub struct GpuDevice {
    ordinal: usize,
}

#[cfg(not(feature = "cuda"))]
impl GpuDevice {
    /// Always returns an error — compile with `features = ["cuda"]`.
    pub fn new(ordinal: usize) -> GpuResult<Self> {
        let _ = ordinal;
        Err(GpuError::NoCudaFeature)
    }

    /// The device ordinal.
    #[inline]
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }
}
