//! GPU 1-D FFT integration tests via cuFFT. (#579)
//!
//! Verifies the `linalg::fft` / `ifft` / `rfft` / `irfft` GPU dispatch path
//! against the CPU reference implementation. All result tensors must remain
//! on GPU end-to-end.

#![cfg(feature = "cuda")]

use ferrotorch_core::fft::{fft, ifft, irfft, rfft};
use ferrotorch_core::{Device, Tensor, TensorStorage};
use ferrotorch_gpu::init_cuda_backend;

fn ensure_cuda() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

fn cpu_t_f32(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false)
        .expect("cpu tensor f32")
}

fn cpu_t_f64(data: Vec<f64>, shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false)
        .expect("cpu tensor f64")
}

fn assert_close_f32(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (av - bv).abs() < tol,
            "at index {i}: a={av} b={bv} (tol={tol})"
        );
    }
}

fn assert_close_f64(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (av - bv).abs() < tol,
            "at index {i}: a={av} b={bv} (tol={tol})"
        );
    }
}

#[test]
fn fft_f32_matches_cpu() {
    ensure_cuda();
    // Complex input shape [4, 2] = 4 complex samples.
    // Use [1+0i, 2+0i, 3+0i, 4+0i] so DFT is real-valued and closed-form.
    // DFT: [10, -2+2i, -2, -2-2i]
    let data: Vec<f32> = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0];
    let a_cpu = cpu_t_f32(data, &[4, 2]);
    let a_gpu = a_cpu.clone().to(Device::Cuda(0)).unwrap();

    let out_cpu = fft(&a_cpu, None).unwrap();
    let out_gpu = fft(&a_gpu, None).unwrap();
    assert!(out_gpu.is_cuda());
    assert_eq!(out_gpu.shape(), out_cpu.shape());

    let cpu_data = out_cpu.data().unwrap();
    let gpu_data = out_gpu.cpu().unwrap().data().unwrap().to_vec();
    assert_close_f32(&gpu_data, cpu_data, 1e-4);
}

#[test]
fn fft_ifft_roundtrip_f32() {
    ensure_cuda();
    // ifft(fft(x)) ≈ x — verify on GPU end-to-end. 8 complex samples = 16 floats.
    let data: Vec<f32> = (0..8).flat_map(|i| [i as f32, (i as f32) * 0.5]).collect();
    let a_cpu = cpu_t_f32(data.clone(), &[8, 2]);
    let a_gpu = a_cpu.clone().to(Device::Cuda(0)).unwrap();

    let f = fft(&a_gpu, None).unwrap();
    assert!(f.is_cuda());
    let back = ifft(&f, None).unwrap();
    assert!(back.is_cuda());

    let host = back.cpu().unwrap().data().unwrap().to_vec();
    assert_close_f32(&host, &data, 1e-4);
}

#[test]
fn fft_batched_f32() {
    ensure_cuda();
    // 2 batches of 4 complex samples each: [B=2, n=4, 2].
    // First batch: [1, 2, 3, 4] (real)
    // Second batch: [4, 3, 2, 1] (real)
    let data: Vec<f32> = vec![
        1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, // batch 0
        4.0, 0.0, 3.0, 0.0, 2.0, 0.0, 1.0, 0.0, // batch 1
    ];
    let a_cpu = cpu_t_f32(data, &[2, 4, 2]);
    let a_gpu = a_cpu.clone().to(Device::Cuda(0)).unwrap();

    let out_cpu = fft(&a_cpu, None).unwrap();
    let out_gpu = fft(&a_gpu, None).unwrap();
    assert!(out_gpu.is_cuda());
    let gpu_host = out_gpu.cpu().unwrap().data().unwrap().to_vec();
    assert_close_f32(&gpu_host, out_cpu.data().unwrap(), 1e-4);
}

#[test]
fn fft_f64_matches_cpu() {
    ensure_cuda();
    let data: Vec<f64> = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0];
    let a_cpu = cpu_t_f64(data, &[4, 2]);
    let a_gpu = a_cpu.clone().to(Device::Cuda(0)).unwrap();

    let out_cpu = fft(&a_cpu, None).unwrap();
    let out_gpu = fft(&a_gpu, None).unwrap();
    assert!(out_gpu.is_cuda());
    let gpu_host = out_gpu.cpu().unwrap().data().unwrap().to_vec();
    assert_close_f64(&gpu_host, out_cpu.data().unwrap(), 1e-10);
}

#[test]
fn rfft_irfft_roundtrip_f32() {
    ensure_cuda();
    // rfft(real) → spectrum, then irfft(spectrum, n) → real, must match input.
    let data: Vec<f32> = (0..8).map(|i| (i as f32).sin()).collect();
    let a_cpu = cpu_t_f32(data.clone(), &[8]);
    let a_gpu = a_cpu.clone().to(Device::Cuda(0)).unwrap();

    let spectrum = rfft(&a_gpu, None).unwrap();
    assert!(spectrum.is_cuda());
    assert_eq!(spectrum.shape(), &[5, 2]); // 8/2+1 = 5 complex bins

    let back = irfft(&spectrum, Some(8)).unwrap();
    assert!(back.is_cuda());
    assert_eq!(back.shape(), &[8]);

    let host = back.cpu().unwrap().data().unwrap().to_vec();
    assert_close_f32(&host, &data, 1e-4);
}

#[test]
fn rfft_f32_matches_cpu_spectrum() {
    ensure_cuda();
    // Compare GPU spectrum to CPU spectrum bin-by-bin.
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let a_cpu = cpu_t_f32(data.clone(), &[8]);
    let a_gpu = a_cpu.clone().to(Device::Cuda(0)).unwrap();

    let cpu_spec = rfft(&a_cpu, None).unwrap();
    let gpu_spec = rfft(&a_gpu, None).unwrap();
    assert!(gpu_spec.is_cuda());
    assert_eq!(gpu_spec.shape(), cpu_spec.shape());

    let cpu_data = cpu_spec.data().unwrap();
    let gpu_data = gpu_spec.cpu().unwrap().data().unwrap().to_vec();
    assert_close_f32(&gpu_data, cpu_data, 1e-3);
}

#[test]
fn rfft_irfft_roundtrip_f64() {
    ensure_cuda();
    let data: Vec<f64> = (0..16).map(|i| (i as f64) * 0.1).collect();
    let a_cpu = cpu_t_f64(data.clone(), &[16]);
    let a_gpu = a_cpu.clone().to(Device::Cuda(0)).unwrap();

    let spectrum = rfft(&a_gpu, None).unwrap();
    assert!(spectrum.is_cuda());
    let back = irfft(&spectrum, Some(16)).unwrap();
    assert!(back.is_cuda());

    let host = back.cpu().unwrap().data().unwrap().to_vec();
    assert_close_f64(&host, &data, 1e-10);
}

#[test]
fn fft_pad_truncate_falls_back_to_cpu() {
    // Non-matching `n` currently routes through CPU — verify it still
    // returns the right answer (regression test for the dispatch gate).
    ensure_cuda();
    let data: Vec<f32> = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0];
    let a_cpu = cpu_t_f32(data, &[4, 2]);
    let a_gpu = a_cpu.clone().to(Device::Cuda(0)).unwrap();

    let out = fft(&a_gpu, Some(8)).unwrap();
    let ref_out = fft(&a_cpu, Some(8)).unwrap();
    // CPU fall-back returns CPU tensor; just compare values.
    let out_data: Vec<f32> = if out.is_cuda() {
        out.cpu().unwrap().data().unwrap().to_vec()
    } else {
        out.data().unwrap().to_vec()
    };
    assert_close_f32(&out_data, ref_out.data().unwrap(), 1e-4);
}
