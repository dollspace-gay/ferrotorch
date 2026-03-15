//! GPU memory buffer.
//!
//! [`CudaBuffer`] owns a region of device memory via `cudarc::driver::CudaSlice`
//! and tracks its length and originating device ordinal.

#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;

/// Owned GPU memory buffer holding `len` elements of type `T`.
///
/// The buffer is tied to the device that allocated it. Dropping it frees the
/// underlying device allocation.
#[cfg(feature = "cuda")]
pub struct CudaBuffer<T> {
    pub(crate) data: CudaSlice<T>,
    pub(crate) len: usize,
    pub(crate) device_ordinal: usize,
}

#[cfg(feature = "cuda")]
impl<T> CudaBuffer<T> {
    /// Number of elements in this buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The ordinal of the device that owns this memory.
    #[inline]
    pub fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    /// Borrow the underlying `CudaSlice` for use with cudarc APIs.
    #[inline]
    pub fn inner(&self) -> &CudaSlice<T> {
        &self.data
    }

    /// Mutably borrow the underlying `CudaSlice`.
    #[inline]
    pub fn inner_mut(&mut self) -> &mut CudaSlice<T> {
        &mut self.data
    }
}

#[cfg(feature = "cuda")]
impl<T> std::fmt::Debug for CudaBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaBuffer")
            .field("len", &self.len)
            .field("device_ordinal", &self.device_ordinal)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// Stub when `cuda` feature is disabled
// ---------------------------------------------------------------------------

/// Stub `CudaBuffer` when the `cuda` feature is not enabled.
#[cfg(not(feature = "cuda"))]
#[derive(Debug)]
pub struct CudaBuffer<T> {
    pub(crate) _phantom: std::marker::PhantomData<T>,
    pub(crate) len: usize,
    pub(crate) device_ordinal: usize,
}

#[cfg(not(feature = "cuda"))]
impl<T> CudaBuffer<T> {
    /// Number of elements in this buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The ordinal of the device that owns this memory.
    #[inline]
    pub fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }
}
