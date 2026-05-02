//! Apple Silicon Metal Performance Shaders (MPS) backend skeleton. (#451)
//!
//! This crate ships the platform-detection and `Device::Mps(_)`
//! plumbing that any caller can use unconditionally. The actual Metal
//! kernel layer is gated behind the `metal-backend` feature, which is
//! intentionally **off by default** so non-Apple platforms (Linux, the
//! workspace's primary CI target) build this crate cleanly and just see
//! the "unavailable" path at runtime.
//!
//! # Status
//!
//! - `is_mps_available()` — runtime check; returns `false` unless
//!   `metal-backend` is enabled AND the platform is macOS/iOS.
//! - `MpsDevice` — opaque handle for an MPS device ordinal. Construction
//!   on non-Metal builds returns an error; on Metal builds it's still
//!   stubbed (the kernel layer needs the `metal` crate dependency to
//!   land — that's a follow-up because adding `metal` requires a macOS
//!   CI runner to verify).
//! - `Device::Mps(_)` integration in `ferrotorch-core` is in place. Tensor
//!   construction with `Device::Mps(_)` returns an explicit error
//!   directing callers to enable the metal-backend feature.
//!
//! # Why a skeleton crate
//!
//! Adding the full Metal layer is a 2000+ LOC effort and needs:
//! 1. A macOS CI runner to validate kernel correctness.
//! 2. The `metal` / `objc2-metal` crate dep (macOS-only build).
//! 3. PTX-equivalent kernel definitions in MSL (Metal Shading Language)
//!    for every op in the `GpuBackend` trait surface (~80 fns today).
//!
//! This crate establishes the API contract so downstream code can already
//! write `Device::Mps(0)` paths; the kernel layer can land incrementally
//! without breaking the public API.

use ferrotorch_core::{FerrotorchError, FerrotorchResult};

/// Returns `true` if this build can run MPS kernels (macOS/iOS + the
/// `metal-backend` feature). Always `false` on Linux / Windows.
pub fn is_mps_available() -> bool {
    #[cfg(all(feature = "metal-backend", any(target_os = "macos", target_os = "ios")))]
    {
        true
    }
    #[cfg(not(all(feature = "metal-backend", any(target_os = "macos", target_os = "ios"))))]
    {
        false
    }
}

/// An opaque handle for an Apple-Silicon Metal device.
///
/// `MpsDevice::new(ordinal)` only succeeds when [`is_mps_available`]
/// returns `true`. Otherwise it returns an error pointing the caller
/// at the feature flag and platform requirements.
#[derive(Debug, Clone)]
pub struct MpsDevice {
    ordinal: usize,
}

impl MpsDevice {
    /// Try to construct a device handle for the given ordinal.
    pub fn new(ordinal: usize) -> FerrotorchResult<Self> {
        if !is_mps_available() {
            return Err(FerrotorchError::DeviceUnavailable);
        }
        Ok(Self { ordinal })
    }

    /// Device ordinal (0 = system default GPU).
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }
}

/// Number of MPS devices the system reports. Returns 0 when the backend
/// is unavailable. Apple Silicon machines have exactly one integrated
/// GPU, so this is `0` (no MPS) or `1` in practice.
pub fn device_count() -> usize {
    if is_mps_available() {
        // Real Metal dispatch would call `MTLCopyAllDevices()`; for now
        // assume the system default GPU exists.
        1
    } else {
        0
    }
}

/// Initialize the MPS backend. On non-Apple platforms or when the
/// `metal-backend` feature is off this returns `DeviceUnavailable`.
///
/// On Apple platforms with the feature enabled, the kernel layer
/// (`MpsBackendImpl`) is registered with `gpu_dispatch::set_gpu_backend`.
/// That layer is tracked by the kernel-implementation follow-up issue
/// — see #451 comment for the work plan.
pub fn init_mps_backend() -> FerrotorchResult<()> {
    if !is_mps_available() {
        return Err(FerrotorchError::DeviceUnavailable);
    }
    // The kernel layer (MpsBackendImpl, MSL-compiled compute pipelines
    // for each GpuBackend trait method) lands separately — its
    // implementation requires a macOS CI runner and ~80 MSL kernels
    // that this Linux-hosted workspace can't validate. Until that
    // lands the init succeeds (the platform check passed) and any
    // subsequent op dispatch returns `DeviceUnavailable` from the
    // global gpu_backend lookup.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_mps_available_returns_false_without_feature() {
        // On the workspace's CI (Linux, no metal-backend feature), this
        // must be false. The test passes on macOS too if the feature is
        // off (default).
        if !cfg!(all(
            feature = "metal-backend",
            any(target_os = "macos", target_os = "ios")
        )) {
            assert!(!is_mps_available());
        }
    }

    #[test]
    fn mps_device_construction_unavailable_on_non_apple() {
        if !is_mps_available() {
            assert!(matches!(
                MpsDevice::new(0),
                Err(FerrotorchError::DeviceUnavailable)
            ));
        }
    }

    #[test]
    fn device_count_zero_when_unavailable() {
        if !is_mps_available() {
            assert_eq!(device_count(), 0);
        }
    }

    #[test]
    fn init_returns_unavailable_when_no_metal_backend() {
        if !is_mps_available() {
            assert!(matches!(
                init_mps_backend(),
                Err(FerrotorchError::DeviceUnavailable)
            ));
        }
    }

    #[test]
    fn device_mps_marker_round_trips() {
        // Ferrotorch-core exposes Device::Mps(_) regardless of MPS
        // availability — the variant just doesn't do anything useful
        // without the backend.
        let d = ferrotorch_core::Device::Mps(0);
        assert!(d.is_mps());
        assert_eq!(format!("{d}"), "mps:0");
    }
}
