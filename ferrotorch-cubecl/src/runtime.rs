//! Unified runtime selection for CubeCL backends.
//!
//! [`CubeDevice`] enumerates the three supported backends (CUDA, ROCm, WGPU),
//! each parameterised by a device ordinal. [`CubeRuntime`] holds a selected
//! device and will eventually own the CubeCL `Runtime` handle for kernel
//! launches.

use std::fmt;

// ---------------------------------------------------------------------------
// CubeDevice
// ---------------------------------------------------------------------------

/// A device selector for CubeCL backends.
///
/// The `usize` field is the device ordinal (e.g., GPU index 0, 1, ...).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CubeDevice {
    /// NVIDIA GPU via CUDA PTX codegen.
    Cuda(usize),
    /// Portable GPU via WGPU — AMD (Vulkan), Intel (Vulkan), Apple (Metal).
    Wgpu(usize),
    /// AMD GPU via native HIP/ROCm runtime.
    Rocm(usize),
}

impl CubeDevice {
    /// Device ordinal regardless of backend.
    #[inline]
    pub fn ordinal(&self) -> usize {
        match self {
            Self::Cuda(o) | Self::Wgpu(o) | Self::Rocm(o) => *o,
        }
    }

    /// Human-readable backend name.
    #[inline]
    pub fn backend_name(&self) -> &'static str {
        match self {
            Self::Cuda(_) => "cuda",
            Self::Wgpu(_) => "wgpu",
            Self::Rocm(_) => "rocm",
        }
    }
}

impl fmt::Display for CubeDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.backend_name(), self.ordinal())
    }
}

// ---------------------------------------------------------------------------
// CubeRuntime
// ---------------------------------------------------------------------------

/// CubeCL runtime wrapper that selects and holds the active backend.
///
/// Currently this is a thin wrapper around [`CubeDevice`]. As CubeCL
/// integration deepens, this will own the compiled-kernel cache and the
/// backend `Runtime` handle.
#[derive(Debug, Clone)]
pub struct CubeRuntime {
    device: CubeDevice,
}

impl CubeRuntime {
    /// Create a runtime targeting the given device.
    pub fn new(device: CubeDevice) -> Self {
        Self { device }
    }

    /// The device this runtime targets.
    #[inline]
    pub fn device(&self) -> &CubeDevice {
        &self.device
    }

    /// Auto-detect the best available backend, returning `None` if no GPU
    /// backend feature is enabled.
    ///
    /// Priority order: CUDA > ROCm > WGPU.
    pub fn auto() -> Option<Self> {
        // CUDA takes priority when available.
        #[cfg(feature = "cuda")]
        {
            return Some(Self {
                device: CubeDevice::Cuda(0),
            });
        }

        // ROCm for AMD-native workloads.
        #[cfg(feature = "rocm")]
        {
            return Some(Self {
                device: CubeDevice::Rocm(0),
            });
        }

        // WGPU is the most portable fallback.
        #[cfg(feature = "wgpu")]
        {
            return Some(Self {
                device: CubeDevice::Wgpu(0),
            });
        }

        // No backend feature enabled.
        #[allow(unreachable_code)]
        None
    }

    /// Returns `true` if any GPU backend feature was compiled in.
    pub fn is_available() -> bool {
        cfg!(any(feature = "cuda", feature = "rocm", feature = "wgpu"))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cube_device_ordinal() {
        assert_eq!(CubeDevice::Cuda(3).ordinal(), 3);
        assert_eq!(CubeDevice::Wgpu(1).ordinal(), 1);
        assert_eq!(CubeDevice::Rocm(0).ordinal(), 0);
    }

    #[test]
    fn cube_device_backend_name() {
        assert_eq!(CubeDevice::Cuda(0).backend_name(), "cuda");
        assert_eq!(CubeDevice::Wgpu(0).backend_name(), "wgpu");
        assert_eq!(CubeDevice::Rocm(0).backend_name(), "rocm");
    }

    #[test]
    fn cube_device_display() {
        assert_eq!(CubeDevice::Cuda(2).to_string(), "cuda:2");
        assert_eq!(CubeDevice::Wgpu(0).to_string(), "wgpu:0");
        assert_eq!(CubeDevice::Rocm(1).to_string(), "rocm:1");
    }

    #[test]
    fn cube_device_equality() {
        assert_eq!(CubeDevice::Cuda(0), CubeDevice::Cuda(0));
        assert_ne!(CubeDevice::Cuda(0), CubeDevice::Cuda(1));
        assert_ne!(CubeDevice::Cuda(0), CubeDevice::Wgpu(0));
    }

    #[test]
    fn cube_runtime_new_and_device() {
        let rt = CubeRuntime::new(CubeDevice::Wgpu(0));
        assert_eq!(*rt.device(), CubeDevice::Wgpu(0));
    }

    #[test]
    fn cube_runtime_auto_returns_something_or_none() {
        // With no backend features, auto() returns None.
        // With any backend feature, auto() returns Some.
        let result = CubeRuntime::auto();
        if CubeRuntime::is_available() {
            assert!(result.is_some());
        } else {
            assert!(result.is_none());
        }
    }

    #[test]
    fn cube_runtime_is_available_consistent() {
        // is_available() should match whether auto() succeeds.
        let available = CubeRuntime::is_available();
        let auto = CubeRuntime::auto();
        assert_eq!(available, auto.is_some());
    }

    #[test]
    fn cube_device_clone_and_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(CubeDevice::Cuda(0));
        set.insert(CubeDevice::Wgpu(0));
        set.insert(CubeDevice::Rocm(0));
        assert_eq!(set.len(), 3);

        // Duplicate should not increase size.
        set.insert(CubeDevice::Cuda(0));
        assert_eq!(set.len(), 3);
    }
}
