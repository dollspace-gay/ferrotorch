//! GPU `as_strided_scatter` integration tests. (#574)
//!
//! Exercises the full `Tensor<T>::as_strided_scatter` code path on a real
//! CUDA device, verifying that the f32 and f64 GPU dispatch produces
//! byte-identical output to the CPU implementation.
//!
//! `/rust-gpu-discipline` notes:
//! - We construct the inputs on GPU explicitly and verify the result tensor
//!   is also on GPU before reading it back. The dispatch path must NOT
//!   silently route through host.
//! - The result is materialised on host only via `.cpu()` at the end of each
//!   test for the value comparison — that is the explicit device boundary.

#![cfg(feature = "cuda")]

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
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("cpu tensor f32")
}

fn cpu_t_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("cpu tensor f64")
}

#[test]
fn scatter_into_zeros_f32_matches_cpu() {
    ensure_cuda();

    // dst: zeros [6]; src: [10, 20, 30] written into positions 0, 2, 4.
    let dst_cpu = cpu_t_f32(&[0.0; 6], &[6]);
    let src_cpu = cpu_t_f32(&[10.0, 20.0, 30.0], &[3]);
    let dst_gpu = dst_cpu.clone().to(Device::Cuda(0)).expect("dst -> gpu");
    let src_gpu = src_cpu.clone().to(Device::Cuda(0)).expect("src -> gpu");

    let out_gpu = dst_gpu
        .as_strided_scatter(&src_gpu, &[3], &[2], Some(0))
        .expect("gpu scatter");
    assert!(out_gpu.is_cuda(), "result should remain on GPU");
    assert_eq!(out_gpu.shape(), dst_cpu.shape());

    // Reference via CPU path.
    let out_cpu = dst_cpu
        .as_strided_scatter(&src_cpu, &[3], &[2], Some(0))
        .expect("cpu scatter");

    let gpu_host = out_gpu.cpu().expect(".cpu()").data().unwrap().to_vec();
    assert_eq!(gpu_host, out_cpu.data().unwrap().to_vec());
}

#[test]
fn scatter_preserves_non_view_positions_f32() {
    ensure_cuda();

    // dst: [1, 2, 3, 4, 5, 6]; src: [100, 200] written into positions 1, 3.
    // Result: [1, 100, 3, 200, 5, 6].
    let dst_cpu = cpu_t_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]);
    let src_cpu = cpu_t_f32(&[100.0, 200.0], &[2]);
    let dst_gpu = dst_cpu.clone().to(Device::Cuda(0)).unwrap();
    let src_gpu = src_cpu.clone().to(Device::Cuda(0)).unwrap();

    let out_gpu = dst_gpu
        .as_strided_scatter(&src_gpu, &[2], &[2], Some(1))
        .unwrap();
    assert!(out_gpu.is_cuda());
    let gpu_host = out_gpu.cpu().unwrap().data().unwrap().to_vec();
    assert_eq!(gpu_host, vec![1.0, 100.0, 3.0, 200.0, 5.0, 6.0]);
}

#[test]
fn scatter_2d_view_into_1d_dst_f32() {
    ensure_cuda();

    // dst: zeros [6]; src: 2x3 contiguous laid out at offset 0 with
    // strides [3, 1] — fully overwrites positions 0..6.
    let dst_cpu = cpu_t_f32(&[0.0; 6], &[6]);
    let src_cpu = cpu_t_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let dst_gpu = dst_cpu.clone().to(Device::Cuda(0)).unwrap();
    let src_gpu = src_cpu.clone().to(Device::Cuda(0)).unwrap();

    let out_gpu = dst_gpu
        .as_strided_scatter(&src_gpu, &[2, 3], &[3, 1], Some(0))
        .unwrap();
    assert!(out_gpu.is_cuda());
    assert_eq!(
        out_gpu.cpu().unwrap().data().unwrap().to_vec(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn scatter_f64_basic() {
    ensure_cuda();

    let dst_cpu = cpu_t_f64(&[0.0; 6], &[6]);
    let src_cpu = cpu_t_f64(&[10.0, 20.0, 30.0], &[3]);
    let dst_gpu = dst_cpu.clone().to(Device::Cuda(0)).unwrap();
    let src_gpu = src_cpu.clone().to(Device::Cuda(0)).unwrap();

    let out_gpu = dst_gpu
        .as_strided_scatter(&src_gpu, &[3], &[2], Some(0))
        .unwrap();
    assert!(out_gpu.is_cuda());
    let gpu_host = out_gpu.cpu().unwrap().data().unwrap().to_vec();
    assert_eq!(gpu_host, vec![10.0, 0.0, 20.0, 0.0, 30.0, 0.0]);
}

#[test]
fn scatter_rejects_device_mismatch() {
    ensure_cuda();

    // dst on GPU, src on CPU → DeviceMismatch.
    let dst_cpu = cpu_t_f32(&[0.0; 4], &[4]);
    let src_cpu = cpu_t_f32(&[1.0, 2.0], &[2]);
    let dst_gpu = dst_cpu.to(Device::Cuda(0)).unwrap();
    let err = dst_gpu
        .as_strided_scatter(&src_cpu, &[2], &[2], Some(0))
        .unwrap_err();
    match err {
        ferrotorch_core::error::FerrotorchError::DeviceMismatch { .. } => {}
        other => panic!("expected DeviceMismatch, got {other:?}"),
    }
}

#[test]
fn scatter_overlapping_view_each_position_holds_some_src_value() {
    ensure_cuda();

    // Sliding-window scatter: shape [3, 3], stride [1, 1] → writes 9
    // src values into 5 positions of a length-5 dst (positions 0..=4).
    // Multiple threads target the same dst index, so the *exact* surviving
    // value at each position depends on kernel-thread scheduling — torch
    // documents overlapping `as_strided_scatter` writes as undefined.
    //
    // Instead of fixing a specific schedule, this test verifies the
    // weaker but well-defined invariant: every output position must hold
    // one of the candidate src values reachable for that position
    // (and definitely not the original 0.0, since every position is
    // touched by at least one write).
    let dst_cpu = cpu_t_f32(&[0.0; 5], &[5]);
    let src_data: Vec<f32> = (0..9).map(|i| i as f32).collect();
    let src_cpu = cpu_t_f32(&src_data, &[3, 3]);
    let dst_gpu = dst_cpu.clone().to(Device::Cuda(0)).unwrap();
    let src_gpu = src_cpu.clone().to(Device::Cuda(0)).unwrap();

    let out_gpu = dst_gpu
        .as_strided_scatter(&src_gpu, &[3, 3], &[1, 1], Some(0))
        .unwrap();
    assert!(out_gpu.is_cuda());

    // Reachability table: for each dst position, the set of src indices
    // that map to it via (i, j) -> i + j with the given strides.
    // dst[0] reached by (0,0) -> src[0]
    // dst[1] reached by (0,1)/(1,0) -> src[1] or src[3]
    // dst[2] reached by (0,2)/(1,1)/(2,0) -> src[2], 4, or 6
    // dst[3] reached by (1,2)/(2,1) -> src[5] or src[7]
    // dst[4] reached by (2,2) -> src[8]
    let candidates: [Vec<f32>; 5] = [
        vec![0.0],
        vec![1.0, 3.0],
        vec![2.0, 4.0, 6.0],
        vec![5.0, 7.0],
        vec![8.0],
    ];

    let gpu_host = out_gpu.cpu().unwrap().data().unwrap().to_vec();
    for (i, v) in gpu_host.iter().enumerate() {
        assert!(
            candidates[i].contains(v),
            "dst[{i}] = {v}, expected one of {:?}",
            candidates[i]
        );
    }
}
