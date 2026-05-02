//! GPU `masked_sum` / `masked_mean` integration tests via cuBLAS+reduce_sum. (#597)
//!
//! Validates the lowering: `masked_sum` = `sum(data * mask_as_float)`,
//! `masked_mean` = `masked_sum / count_valid` (with NaN on all-masked).

#![cfg(feature = "cuda")]

use ferrotorch_core::masked::{MaskedTensor, masked_mean, masked_sum};
use ferrotorch_core::{Device, Tensor, TensorStorage};
use ferrotorch_gpu::init_cuda_backend;

fn ensure_cuda() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

fn cpu_t_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn cpu_t_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

#[test]
fn masked_sum_f32_matches_cpu() {
    ensure_cuda();
    let data = cpu_t_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let mask = vec![true, false, true, true, false, true];

    let cpu_sum = masked_sum(&MaskedTensor::new(data.clone(), mask.clone()).unwrap())
        .unwrap()
        .item()
        .unwrap();
    // 1 + 3 + 4 + 6 = 14
    assert!((cpu_sum - 14.0).abs() < 1e-5);

    let gpu_data = data.to(Device::Cuda(0)).unwrap();
    let gpu_mt = MaskedTensor::new(gpu_data, mask).unwrap();
    let gpu_sum = masked_sum(&gpu_mt).unwrap().cpu().unwrap().item().unwrap();
    assert!(
        (gpu_sum - cpu_sum).abs() < 1e-4,
        "gpu_sum={gpu_sum} cpu_sum={cpu_sum}"
    );
}

#[test]
fn masked_sum_f64_matches_cpu() {
    ensure_cuda();
    let data = cpu_t_f64(&[10.5, 20.5, 30.5, 40.5], &[4]);
    let mask = vec![true, true, false, true];
    let expected = 10.5 + 20.5 + 40.5;

    let gpu_data = data.to(Device::Cuda(0)).unwrap();
    let mt = MaskedTensor::new(gpu_data, mask).unwrap();
    let result = masked_sum(&mt).unwrap().cpu().unwrap().item().unwrap();
    assert!((result - expected).abs() < 1e-10);
}

#[test]
fn masked_sum_all_masked_returns_zero() {
    ensure_cuda();
    let data = cpu_t_f32(&[1.0, 2.0, 3.0], &[3]);
    let mask = vec![false, false, false];
    let gpu_data = data.to(Device::Cuda(0)).unwrap();
    let mt = MaskedTensor::new(gpu_data, mask).unwrap();
    let result = masked_sum(&mt).unwrap().cpu().unwrap().item().unwrap();
    assert!(result.abs() < 1e-7);
}

#[test]
fn masked_mean_f32_matches_cpu() {
    ensure_cuda();
    let data = cpu_t_f32(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let mask = vec![true, true, false, true];
    let cpu_mean = (1.0 + 2.0 + 4.0) / 3.0;

    let gpu_data = data.to(Device::Cuda(0)).unwrap();
    let mt = MaskedTensor::new(gpu_data, mask).unwrap();
    let gpu_mean = masked_mean(&mt).unwrap().item().unwrap();
    assert!(
        (gpu_mean - cpu_mean).abs() < 1e-5,
        "gpu={gpu_mean} cpu={cpu_mean}"
    );
}

#[test]
fn masked_mean_all_masked_is_nan() {
    ensure_cuda();
    let data = cpu_t_f32(&[1.0, 2.0, 3.0], &[3]);
    let mask = vec![false, false, false];
    let gpu_data = data.to(Device::Cuda(0)).unwrap();
    let mt = MaskedTensor::new(gpu_data, mask).unwrap();
    let v = masked_mean(&mt).unwrap().item().unwrap();
    assert!(v.is_nan());
}
