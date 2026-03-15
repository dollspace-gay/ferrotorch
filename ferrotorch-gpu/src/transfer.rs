//! Host-to-device and device-to-host memory transfers.
//!
//! These functions copy data between CPU (`&[T]` / `Vec<T>`) and GPU
//! ([`CudaBuffer`]) memory via the device's default CUDA stream.

use crate::buffer::CudaBuffer;
use crate::device::GpuDevice;
use crate::error::{GpuError, GpuResult};

/// Copy a host slice to device memory, returning a new [`CudaBuffer`].
///
/// The transfer uses the device's default CUDA stream and blocks until
/// the copy is complete.
///
/// # Errors
///
/// Returns [`GpuError::Driver`] if the CUDA memcpy fails.
#[cfg(feature = "cuda")]
pub fn cpu_to_gpu<T>(data: &[T], device: &GpuDevice) -> GpuResult<CudaBuffer<T>>
where
    T: cudarc::driver::DeviceRepr,
{
    let slice = device.stream().clone_htod(data)?;
    Ok(CudaBuffer {
        len: data.len(),
        device_ordinal: device.ordinal(),
        data: slice,
    })
}

/// Copy device memory back to a host `Vec<T>`.
///
/// # Errors
///
/// Returns [`GpuError::DeviceMismatch`] if the buffer's device ordinal does
/// not match the provided device, or [`GpuError::Driver`] on CUDA errors.
#[cfg(feature = "cuda")]
pub fn gpu_to_cpu<T>(buffer: &CudaBuffer<T>, device: &GpuDevice) -> GpuResult<Vec<T>>
where
    T: cudarc::driver::DeviceRepr,
{
    if buffer.device_ordinal() != device.ordinal() {
        return Err(GpuError::DeviceMismatch {
            expected: buffer.device_ordinal(),
            got: device.ordinal(),
        });
    }
    let vec = device.stream().clone_dtoh(&buffer.data)?;
    Ok(vec)
}

/// Allocate a zero-initialized [`CudaBuffer`] on the given device.
///
/// # Errors
///
/// Returns [`GpuError::Driver`] if the allocation fails.
#[cfg(feature = "cuda")]
pub fn alloc_zeros<T>(len: usize, device: &GpuDevice) -> GpuResult<CudaBuffer<T>>
where
    T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits,
{
    let slice = device.stream().alloc_zeros::<T>(len)?;
    Ok(CudaBuffer {
        len,
        device_ordinal: device.ordinal(),
        data: slice,
    })
}

// ---------------------------------------------------------------------------
// Stubs when `cuda` feature is disabled
// ---------------------------------------------------------------------------

/// Stub — always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn cpu_to_gpu<T>(_data: &[T], _device: &GpuDevice) -> GpuResult<CudaBuffer<T>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub — always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_to_cpu<T>(_buffer: &CudaBuffer<T>, _device: &GpuDevice) -> GpuResult<Vec<T>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub — always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn alloc_zeros<T>(_len: usize, _device: &GpuDevice) -> GpuResult<CudaBuffer<T>> {
    Err(GpuError::NoCudaFeature)
}

// ---------------------------------------------------------------------------
// Tests — require a real CUDA GPU
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;

    #[test]
    fn round_trip_f32() {
        let device = GpuDevice::new(0).expect("CUDA device 0");
        let host: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let gpu_buf = cpu_to_gpu(&host, &device).expect("cpu_to_gpu");
        assert_eq!(gpu_buf.len(), 5);
        assert_eq!(gpu_buf.device_ordinal(), 0);

        let back = gpu_to_cpu(&gpu_buf, &device).expect("gpu_to_cpu");
        assert_eq!(back, host);
    }

    #[test]
    fn round_trip_f64() {
        let device = GpuDevice::new(0).expect("CUDA device 0");
        let host: Vec<f64> = vec![1.0, -2.5, 3.14, 0.0, f64::MAX];

        let gpu_buf = cpu_to_gpu(&host, &device).expect("cpu_to_gpu");
        assert_eq!(gpu_buf.len(), 5);

        let back = gpu_to_cpu(&gpu_buf, &device).expect("gpu_to_cpu");
        assert_eq!(back, host);
    }

    #[test]
    fn alloc_zeros_f32() {
        let device = GpuDevice::new(0).expect("CUDA device 0");
        let buf = alloc_zeros::<f32>(1024, &device).expect("alloc_zeros");
        assert_eq!(buf.len(), 1024);

        let host = gpu_to_cpu(&buf, &device).expect("gpu_to_cpu");
        assert!(host.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn empty_transfer() {
        let device = GpuDevice::new(0).expect("CUDA device 0");
        let host: Vec<f32> = vec![];

        let gpu_buf = cpu_to_gpu(&host, &device).expect("cpu_to_gpu");
        assert_eq!(gpu_buf.len(), 0);
        assert!(gpu_buf.is_empty());

        let back = gpu_to_cpu(&gpu_buf, &device).expect("gpu_to_cpu");
        assert!(back.is_empty());
    }

    #[test]
    fn large_transfer() {
        let device = GpuDevice::new(0).expect("CUDA device 0");
        let n = 1_000_000;
        let host: Vec<f32> = (0..n).map(|i| i as f32).collect();

        let gpu_buf = cpu_to_gpu(&host, &device).expect("cpu_to_gpu");
        assert_eq!(gpu_buf.len(), n);

        let back = gpu_to_cpu(&gpu_buf, &device).expect("gpu_to_cpu");
        assert_eq!(back, host);
    }

    #[test]
    fn device_mismatch_rejected() {
        // This test only makes sense with 1 GPU — it just verifies the
        // mismatch check path by constructing a buffer that claims ordinal 99.
        let device = GpuDevice::new(0).expect("CUDA device 0");
        let host: Vec<f32> = vec![1.0];
        let mut buf = cpu_to_gpu(&host, &device).expect("cpu_to_gpu");
        // Manually tamper with the ordinal to simulate a mismatch.
        buf.device_ordinal = 99;

        let err = gpu_to_cpu(&buf, &device).unwrap_err();
        match err {
            GpuError::DeviceMismatch { expected: 99, got: 0 } => {}
            other => panic!("unexpected error: {other}"),
        }
    }
}
