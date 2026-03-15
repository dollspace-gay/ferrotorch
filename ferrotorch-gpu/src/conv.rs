//! GPU-accelerated 2-D convolution via im2col + cuBLAS GEMM.
//!
//! Conv2d is decomposed into two steps:
//!
//! 1. **im2col** — rearrange each input receptive-field patch into a column
//!    of a matrix. Currently performed on the CPU.
//! 2. **GEMM** — matrix-multiply the reshaped weight matrix with the column
//!    matrix using cuBLAS SGEMM on the GPU.
//!
//! The GEMM is the compute-heavy part (O(C_out * C_in * kH * kW * H_out * W_out)
//! multiply-adds per batch element), so GPU acceleration still provides
//! significant speedups for large convolutions even with CPU im2col.
//!
//! # Layout
//!
//! All tensors use row-major (NCHW) layout, matching PyTorch's default:
//!
//! - **input**: `[B, C_in, H, W]`
//! - **weight**: `[C_out, C_in, kH, kW]`
//! - **bias**: `[C_out]`
//! - **output**: `[B, C_out, H_out, W_out]`
//!
//! # CPU fallback
//!
//! When the `cuda` feature is disabled, all functions return
//! [`GpuError::NoCudaFeature`].

use crate::blas::gpu_matmul_f32;
use crate::buffer::CudaBuffer;
use crate::device::GpuDevice;
use crate::error::{GpuError, GpuResult};
#[cfg(feature = "cuda")]
use crate::transfer::{cpu_to_gpu, gpu_to_cpu};

// ---------------------------------------------------------------------------
// CPU im2col (shared between cuda and non-cuda paths for compilation)
// ---------------------------------------------------------------------------

/// Extract image patches into columns on the CPU.
///
/// Given a 4-D input `[B, C, H, W]` (flattened row-major), produces a
/// flattened 3-D output `[B, C*kH*kW, H_out*W_out]` where each column
/// is one flattened receptive-field patch.
///
/// Returns `(columns, col_rows, col_cols)` where:
/// - `col_rows = C_in * kH * kW`
/// - `col_cols = H_out * W_out`
#[allow(clippy::too_many_arguments)]
fn im2col_cpu(
    input: &[f32],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> (Vec<f32>, usize, usize) {
    let h_out = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    let w_out = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    let col_rows = channels * kernel_h * kernel_w;
    let col_cols = h_out * w_out;

    let mut cols = vec![0.0f32; batch * col_rows * col_cols];

    for b in 0..batch {
        for c in 0..channels {
            for kh in 0..kernel_h {
                for kw in 0..kernel_w {
                    let row = c * kernel_h * kernel_w + kh * kernel_w + kw;
                    for oh in 0..h_out {
                        for ow in 0..w_out {
                            let ih = oh * stride_h + kh;
                            let iw = ow * stride_w + kw;
                            let col = oh * w_out + ow;

                            let val = if ih >= pad_h
                                && iw >= pad_w
                                && (ih - pad_h) < height
                                && (iw - pad_w) < width
                            {
                                let real_h = ih - pad_h;
                                let real_w = iw - pad_w;
                                input[b * channels * height * width
                                    + c * height * width
                                    + real_h * width
                                    + real_w]
                            } else {
                                0.0
                            };

                            cols[b * col_rows * col_cols + row * col_cols + col] = val;
                        }
                    }
                }
            }
        }
    }

    (cols, col_rows, col_cols)
}

// ---------------------------------------------------------------------------
// GPU conv2d — hybrid: CPU im2col + GPU GEMM
// ---------------------------------------------------------------------------

/// Compute a 2-D convolution on the GPU using im2col + cuBLAS GEMM.
///
/// # Arguments
///
/// - `input` — GPU buffer containing `[B, C_in, H, W]` flattened in row-major order.
/// - `weight` — GPU buffer containing `[C_out, C_in, kH, kW]` flattened in row-major order.
/// - `bias` — optional GPU buffer containing `[C_out]` bias values.
/// - `input_shape` — `[B, C_in, H, W]`.
/// - `weight_shape` — `[C_out, C_in, kH, kW]`.
/// - `stride` — `(stride_h, stride_w)`.
/// - `padding` — `(pad_h, pad_w)`.
/// - `device` — the GPU device that owns all buffers.
///
/// # Returns
///
/// A tuple `(output_buffer, output_shape)` where `output_shape` is
/// `[B, C_out, H_out, W_out]`.
///
/// # Algorithm
///
/// 1. Copy input to CPU.
/// 2. Run im2col on CPU to produce columns `[B, C_in*kH*kW, H_out*W_out]`.
/// 3. Reshape weight on CPU to `[C_out, C_in*kH*kW]`.
/// 4. Upload weight_2d and columns to GPU.
/// 5. For each batch element: GPU GEMM: `weight_2d @ columns_b = output_b`.
/// 6. If bias is present, broadcast-add to each spatial position.
/// 7. Return concatenated output.
///
/// # Errors
///
/// - [`GpuError::ShapeMismatch`] if buffer lengths are inconsistent with shapes.
/// - [`GpuError::DeviceMismatch`] if buffers are on different devices.
/// - [`GpuError::Driver`] or [`GpuError::Blas`] on CUDA runtime errors.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "cuda")]
pub fn gpu_conv2d_f32(
    input: &CudaBuffer<f32>,
    weight: &CudaBuffer<f32>,
    bias: Option<&CudaBuffer<f32>>,
    input_shape: [usize; 4],
    weight_shape: [usize; 4],
    stride: (usize, usize),
    padding: (usize, usize),
    device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f32>, [usize; 4])> {
    let [batch, c_in, h, w] = input_shape;
    let [c_out, c_in_w, kh, kw] = weight_shape;

    // Validate channel consistency.
    if c_in != c_in_w {
        return Err(GpuError::ShapeMismatch {
            op: "conv2d",
            expected: vec![c_in],
            got: vec![c_in_w],
        });
    }

    // Validate buffer sizes.
    let expected_input_len = batch * c_in * h * w;
    if input.len() != expected_input_len {
        return Err(GpuError::ShapeMismatch {
            op: "conv2d",
            expected: input_shape.to_vec(),
            got: vec![input.len()],
        });
    }

    let expected_weight_len = c_out * c_in * kh * kw;
    if weight.len() != expected_weight_len {
        return Err(GpuError::ShapeMismatch {
            op: "conv2d",
            expected: weight_shape.to_vec(),
            got: vec![weight.len()],
        });
    }

    if let Some(b) = bias {
        if b.len() != c_out {
            return Err(GpuError::ShapeMismatch {
                op: "conv2d",
                expected: vec![c_out],
                got: vec![b.len()],
            });
        }
    }

    // Validate devices.
    if input.device_ordinal() != device.ordinal() {
        return Err(GpuError::DeviceMismatch {
            expected: device.ordinal(),
            got: input.device_ordinal(),
        });
    }
    if weight.device_ordinal() != device.ordinal() {
        return Err(GpuError::DeviceMismatch {
            expected: device.ordinal(),
            got: weight.device_ordinal(),
        });
    }

    // Compute output spatial dimensions.
    let h_out = (h + 2 * padding.0 - kh) / stride.0 + 1;
    let w_out = (w + 2 * padding.1 - kw) / stride.1 + 1;
    let output_shape = [batch, c_out, h_out, w_out];

    // Handle degenerate case.
    if batch == 0 || c_out == 0 || h_out == 0 || w_out == 0 {
        let out = crate::transfer::alloc_zeros::<f32>(0, device)?;
        return Ok((out, output_shape));
    }

    // --- Step 1: Copy input to CPU ---
    let input_host = gpu_to_cpu(input, device)?;

    // --- Step 2: im2col on CPU ---
    let (cols, col_rows, col_cols) =
        im2col_cpu(&input_host, batch, c_in, h, w, kh, kw, stride.0, stride.1, padding.0, padding.1);

    // col_rows = C_in * kH * kW
    // col_cols = H_out * W_out

    // --- Step 3: Copy weight to CPU and reshape to [C_out, col_rows] ---
    // Weight is already stored as [C_out, C_in*kH*kW] in row-major order
    // when flattened, so no actual reshape is needed — just upload it.
    let weight_host = gpu_to_cpu(weight, device)?;

    // Upload weight_2d [C_out, col_rows] to GPU.
    let weight_gpu = cpu_to_gpu(&weight_host, device)?;

    // --- Step 4: Bias to CPU (if present) ---
    let bias_host = if let Some(b) = bias {
        Some(gpu_to_cpu(b, device)?)
    } else {
        None
    };

    // --- Step 5: For each batch element, GEMM on GPU ---
    // weight_2d: [C_out, col_rows]  @  cols_b: [col_rows, col_cols]  =  out_b: [C_out, col_cols]
    let out_elems_per_batch = c_out * col_cols;
    let mut output_host = Vec::with_capacity(batch * out_elems_per_batch);

    for b in 0..batch {
        // Extract this batch's columns: [col_rows, col_cols]
        let cols_start = b * col_rows * col_cols;
        let cols_end = cols_start + col_rows * col_cols;
        let cols_b = &cols[cols_start..cols_end];

        // Upload columns for this batch to GPU.
        let cols_gpu = cpu_to_gpu(cols_b, device)?;

        // GPU GEMM: [C_out, col_rows] @ [col_rows, col_cols] = [C_out, col_cols]
        let out_gpu = gpu_matmul_f32(&weight_gpu, &cols_gpu, c_out, col_rows, col_cols, device)?;

        // Copy result back to CPU.
        let mut out_b = gpu_to_cpu(&out_gpu, device)?;

        // --- Step 6: Add bias if present ---
        // out_b is [C_out, col_cols] in row-major. Bias is [C_out].
        // Add bias[c] to every element in row c.
        if let Some(ref bias_data) = bias_host {
            for c in 0..c_out {
                for j in 0..col_cols {
                    out_b[c * col_cols + j] += bias_data[c];
                }
            }
        }

        output_host.extend_from_slice(&out_b);
    }

    // Upload full output to GPU.
    let output_gpu = cpu_to_gpu(&output_host, device)?;

    Ok((output_gpu, output_shape))
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_conv2d_f32(
    _input: &CudaBuffer<f32>,
    _weight: &CudaBuffer<f32>,
    _bias: Option<&CudaBuffer<f32>>,
    _input_shape: [usize; 4],
    _weight_shape: [usize; 4],
    _stride: (usize, usize),
    _padding: (usize, usize),
    _device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f32>, [usize; 4])> {
    Err(GpuError::NoCudaFeature)
}

// ---------------------------------------------------------------------------
// Convenience: CPU-only conv2d reference (for testing)
// ---------------------------------------------------------------------------

/// Pure CPU conv2d for reference/testing.
///
/// Same im2col + matmul approach, entirely on the CPU.
/// Used by tests to verify GPU results.
#[cfg(test)]
fn cpu_conv2d_reference(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    input_shape: [usize; 4],
    weight_shape: [usize; 4],
    stride: (usize, usize),
    padding: (usize, usize),
) -> (Vec<f32>, [usize; 4]) {
    let [batch, c_in, h, w] = input_shape;
    let [c_out, _c_in_w, kh, kw] = weight_shape;

    let h_out = (h + 2 * padding.0 - kh) / stride.0 + 1;
    let w_out = (w + 2 * padding.1 - kw) / stride.1 + 1;
    let output_shape = [batch, c_out, h_out, w_out];

    let col_rows = c_in * kh * kw;
    let col_cols = h_out * w_out;

    let (cols, _, _) =
        im2col_cpu(input, batch, c_in, h, w, kh, kw, stride.0, stride.1, padding.0, padding.1);

    let mut output = Vec::with_capacity(batch * c_out * col_cols);

    for b in 0..batch {
        let cols_start = b * col_rows * col_cols;

        // weight_2d [C_out, col_rows] @ cols_b [col_rows, col_cols] = out_b [C_out, col_cols]
        for co in 0..c_out {
            for j in 0..col_cols {
                let mut sum = 0.0f32;
                for p in 0..col_rows {
                    sum += weight[co * col_rows + p] * cols[cols_start + p * col_cols + j];
                }
                if let Some(bias_data) = bias {
                    sum += bias_data[co];
                }
                output.push(sum);
            }
        }
    }

    (output, output_shape)
}

// ---------------------------------------------------------------------------
// Tests — require a real CUDA GPU
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;
    use crate::device::GpuDevice;
    use crate::transfer::{cpu_to_gpu, gpu_to_cpu};

    /// Helper: compare two f32 slices with tolerance.
    fn assert_close(got: &[f32], expected: &[f32], tol: f32, label: &str) {
        assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < tol,
                "{label}: element {i}: got {g}, expected {e}, diff {}",
                (g - e).abs(),
            );
        }
    }

    // -- Output shape correctness ---------------------------------------------

    #[test]
    fn conv2d_output_shape_no_padding() {
        // Input: [1, 1, 5, 5], Weight: [1, 1, 3, 3], stride=1, padding=0
        // H_out = (5 - 3) / 1 + 1 = 3
        // W_out = (5 - 3) / 1 + 1 = 3
        let dev = GpuDevice::new(0).expect("CUDA device 0");

        let input_data: Vec<f32> = (0..25).map(|i| i as f32).collect();
        let weight_data: Vec<f32> = vec![1.0; 9]; // 3x3 all-ones kernel

        let input = cpu_to_gpu(&input_data, &dev).expect("input to gpu");
        let weight = cpu_to_gpu(&weight_data, &dev).expect("weight to gpu");

        let (out, shape) = gpu_conv2d_f32(
            &input,
            &weight,
            None,
            [1, 1, 5, 5],
            [1, 1, 3, 3],
            (1, 1),
            (0, 0),
            &dev,
        )
        .expect("gpu_conv2d_f32");

        assert_eq!(shape, [1, 1, 3, 3]);
        assert_eq!(out.len(), 9);
    }

    #[test]
    fn conv2d_output_shape_with_padding() {
        // Input: [1, 1, 5, 5], Weight: [1, 1, 3, 3], stride=1, padding=1
        // H_out = (5 + 2 - 3) / 1 + 1 = 5
        // W_out = (5 + 2 - 3) / 1 + 1 = 5
        let dev = GpuDevice::new(0).expect("CUDA device 0");

        let input_data: Vec<f32> = (0..25).map(|i| i as f32).collect();
        let weight_data: Vec<f32> = vec![1.0; 9];

        let input = cpu_to_gpu(&input_data, &dev).expect("input to gpu");
        let weight = cpu_to_gpu(&weight_data, &dev).expect("weight to gpu");

        let (out, shape) = gpu_conv2d_f32(
            &input,
            &weight,
            None,
            [1, 1, 5, 5],
            [1, 1, 3, 3],
            (1, 1),
            (1, 1),
            &dev,
        )
        .expect("gpu_conv2d_f32");

        assert_eq!(shape, [1, 1, 5, 5]);
        assert_eq!(out.len(), 25);
    }

    #[test]
    fn conv2d_output_shape_stride2() {
        // Input: [1, 1, 6, 6], Weight: [1, 1, 3, 3], stride=2, padding=0
        // H_out = (6 - 3) / 2 + 1 = 2
        // W_out = (6 - 3) / 2 + 1 = 2
        let dev = GpuDevice::new(0).expect("CUDA device 0");

        let input_data: Vec<f32> = (0..36).map(|i| i as f32).collect();
        let weight_data: Vec<f32> = vec![1.0; 9];

        let input = cpu_to_gpu(&input_data, &dev).expect("input to gpu");
        let weight = cpu_to_gpu(&weight_data, &dev).expect("weight to gpu");

        let (out, shape) = gpu_conv2d_f32(
            &input,
            &weight,
            None,
            [1, 1, 6, 6],
            [1, 1, 3, 3],
            (2, 2),
            (0, 0),
            &dev,
        )
        .expect("gpu_conv2d_f32");

        assert_eq!(shape, [1, 1, 2, 2]);
        assert_eq!(out.len(), 4);
    }

    // -- Correctness vs CPU reference -----------------------------------------

    #[test]
    fn conv2d_correctness_vs_cpu() {
        let dev = GpuDevice::new(0).expect("CUDA device 0");

        // Input: [2, 3, 8, 8], Weight: [4, 3, 3, 3], stride=1, padding=1
        let input_shape = [2, 3, 8, 8];
        let weight_shape = [4, 3, 3, 3];
        let stride = (1, 1);
        let padding = (1, 1);

        let input_len: usize = input_shape.iter().product();
        let weight_len: usize = weight_shape.iter().product();

        // Deterministic non-trivial data.
        let input_data: Vec<f32> = (0..input_len)
            .map(|i| ((i * 7 + 13) % 100) as f32 / 100.0)
            .collect();
        let weight_data: Vec<f32> = (0..weight_len)
            .map(|i| ((i * 11 + 3) % 100) as f32 / 100.0 - 0.5)
            .collect();
        let bias_data: Vec<f32> = vec![0.1, -0.2, 0.3, -0.1];

        // CPU reference.
        let (expected_output, expected_shape) = cpu_conv2d_reference(
            &input_data,
            &weight_data,
            Some(&bias_data),
            input_shape,
            weight_shape,
            stride,
            padding,
        );

        // GPU.
        let input_gpu = cpu_to_gpu(&input_data, &dev).expect("input to gpu");
        let weight_gpu = cpu_to_gpu(&weight_data, &dev).expect("weight to gpu");
        let bias_gpu = cpu_to_gpu(&bias_data, &dev).expect("bias to gpu");

        let (out_gpu, out_shape) = gpu_conv2d_f32(
            &input_gpu,
            &weight_gpu,
            Some(&bias_gpu),
            input_shape,
            weight_shape,
            stride,
            padding,
            &dev,
        )
        .expect("gpu_conv2d_f32");

        assert_eq!(out_shape, expected_shape);

        let out_host = gpu_to_cpu(&out_gpu, &dev).expect("gpu_to_cpu");
        assert_close(&out_host, &expected_output, 1e-3, "conv2d vs cpu");
    }

    #[test]
    fn conv2d_correctness_no_bias() {
        let dev = GpuDevice::new(0).expect("CUDA device 0");

        let input_shape = [1, 2, 4, 4];
        let weight_shape = [3, 2, 3, 3];
        let stride = (1, 1);
        let padding = (0, 0);

        let input_len: usize = input_shape.iter().product();
        let weight_len: usize = weight_shape.iter().product();

        let input_data: Vec<f32> = (0..input_len)
            .map(|i| ((i * 3 + 7) % 50) as f32 / 50.0)
            .collect();
        let weight_data: Vec<f32> = (0..weight_len)
            .map(|i| ((i * 5 + 1) % 40) as f32 / 40.0 - 0.5)
            .collect();

        let (expected_output, expected_shape) = cpu_conv2d_reference(
            &input_data,
            &weight_data,
            None,
            input_shape,
            weight_shape,
            stride,
            padding,
        );

        let input_gpu = cpu_to_gpu(&input_data, &dev).expect("input to gpu");
        let weight_gpu = cpu_to_gpu(&weight_data, &dev).expect("weight to gpu");

        let (out_gpu, out_shape) = gpu_conv2d_f32(
            &input_gpu,
            &weight_gpu,
            None,
            input_shape,
            weight_shape,
            stride,
            padding,
            &dev,
        )
        .expect("gpu_conv2d_f32");

        assert_eq!(out_shape, expected_shape);

        let out_host = gpu_to_cpu(&out_gpu, &dev).expect("gpu_to_cpu");
        assert_close(&out_host, &expected_output, 1e-3, "conv2d no bias");
    }

    // -- 1x1 kernel -----------------------------------------------------------

    #[test]
    fn conv2d_1x1_kernel() {
        // 1x1 convolution is just a per-pixel linear layer.
        let dev = GpuDevice::new(0).expect("CUDA device 0");

        let input_shape = [1, 3, 4, 4];
        let weight_shape = [2, 3, 1, 1];
        let stride = (1, 1);
        let padding = (0, 0);

        let input_len: usize = input_shape.iter().product();
        let weight_len: usize = weight_shape.iter().product();

        let input_data: Vec<f32> = (0..input_len)
            .map(|i| i as f32 / input_len as f32)
            .collect();
        let weight_data: Vec<f32> = (0..weight_len)
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect();
        let bias_data: Vec<f32> = vec![0.5, -0.5];

        let (expected_output, expected_shape) = cpu_conv2d_reference(
            &input_data,
            &weight_data,
            Some(&bias_data),
            input_shape,
            weight_shape,
            stride,
            padding,
        );

        // 1x1 conv: output spatial dims = input spatial dims.
        assert_eq!(expected_shape, [1, 2, 4, 4]);

        let input_gpu = cpu_to_gpu(&input_data, &dev).expect("input to gpu");
        let weight_gpu = cpu_to_gpu(&weight_data, &dev).expect("weight to gpu");
        let bias_gpu = cpu_to_gpu(&bias_data, &dev).expect("bias to gpu");

        let (out_gpu, out_shape) = gpu_conv2d_f32(
            &input_gpu,
            &weight_gpu,
            Some(&bias_gpu),
            input_shape,
            weight_shape,
            stride,
            padding,
            &dev,
        )
        .expect("gpu_conv2d_f32");

        assert_eq!(out_shape, expected_shape);

        let out_host = gpu_to_cpu(&out_gpu, &dev).expect("gpu_to_cpu");
        assert_close(&out_host, &expected_output, 1e-4, "conv2d 1x1");
    }

    // -- Multi-batch ----------------------------------------------------------

    #[test]
    fn conv2d_multi_batch() {
        let dev = GpuDevice::new(0).expect("CUDA device 0");

        let input_shape = [4, 2, 6, 6];
        let weight_shape = [3, 2, 3, 3];
        let stride = (1, 1);
        let padding = (1, 1);

        let input_len: usize = input_shape.iter().product();
        let weight_len: usize = weight_shape.iter().product();

        let input_data: Vec<f32> = (0..input_len)
            .map(|i| ((i * 13 + 5) % 200) as f32 / 200.0 - 0.5)
            .collect();
        let weight_data: Vec<f32> = (0..weight_len)
            .map(|i| ((i * 17 + 11) % 100) as f32 / 100.0 - 0.5)
            .collect();

        let (expected_output, expected_shape) = cpu_conv2d_reference(
            &input_data,
            &weight_data,
            None,
            input_shape,
            weight_shape,
            stride,
            padding,
        );

        let input_gpu = cpu_to_gpu(&input_data, &dev).expect("input to gpu");
        let weight_gpu = cpu_to_gpu(&weight_data, &dev).expect("weight to gpu");

        let (out_gpu, out_shape) = gpu_conv2d_f32(
            &input_gpu,
            &weight_gpu,
            None,
            input_shape,
            weight_shape,
            stride,
            padding,
            &dev,
        )
        .expect("gpu_conv2d_f32");

        assert_eq!(out_shape, expected_shape);

        let out_host = gpu_to_cpu(&out_gpu, &dev).expect("gpu_to_cpu");
        assert_close(&out_host, &expected_output, 1e-3, "conv2d multi-batch");
    }

    // -- Shape validation errors ----------------------------------------------

    #[test]
    fn conv2d_channel_mismatch() {
        let dev = GpuDevice::new(0).expect("CUDA device 0");

        let input_data = vec![0.0f32; 1 * 3 * 4 * 4]; // C_in = 3
        let weight_data = vec![0.0f32; 2 * 5 * 3 * 3]; // C_in_w = 5 (mismatch!)

        let input = cpu_to_gpu(&input_data, &dev).expect("input to gpu");
        let weight = cpu_to_gpu(&weight_data, &dev).expect("weight to gpu");

        let err = gpu_conv2d_f32(
            &input,
            &weight,
            None,
            [1, 3, 4, 4],
            [2, 5, 3, 3],
            (1, 1),
            (0, 0),
            &dev,
        )
        .unwrap_err();

        match err {
            GpuError::ShapeMismatch { op: "conv2d", .. } => {}
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn conv2d_wrong_input_length() {
        let dev = GpuDevice::new(0).expect("CUDA device 0");

        // Claim shape [1, 1, 4, 4] = 16 elements, but buffer has 10.
        let input_data = vec![0.0f32; 10];
        let weight_data = vec![0.0f32; 1 * 1 * 3 * 3];

        let input = cpu_to_gpu(&input_data, &dev).expect("input to gpu");
        let weight = cpu_to_gpu(&weight_data, &dev).expect("weight to gpu");

        let err = gpu_conv2d_f32(
            &input,
            &weight,
            None,
            [1, 1, 4, 4],
            [1, 1, 3, 3],
            (1, 1),
            (0, 0),
            &dev,
        )
        .unwrap_err();

        match err {
            GpuError::ShapeMismatch { op: "conv2d", .. } => {}
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn conv2d_wrong_bias_length() {
        let dev = GpuDevice::new(0).expect("CUDA device 0");

        let input_data = vec![0.0f32; 1 * 1 * 5 * 5];
        let weight_data = vec![0.0f32; 2 * 1 * 3 * 3]; // C_out = 2
        let bias_data = vec![0.0f32; 5]; // should be 2, not 5

        let input = cpu_to_gpu(&input_data, &dev).expect("input to gpu");
        let weight = cpu_to_gpu(&weight_data, &dev).expect("weight to gpu");
        let bias = cpu_to_gpu(&bias_data, &dev).expect("bias to gpu");

        let err = gpu_conv2d_f32(
            &input,
            &weight,
            Some(&bias),
            [1, 1, 5, 5],
            [2, 1, 3, 3],
            (1, 1),
            (0, 0),
            &dev,
        )
        .unwrap_err();

        match err {
            GpuError::ShapeMismatch { op: "conv2d", .. } => {}
            other => panic!("unexpected error: {other}"),
        }
    }

    // -- Stride > 1 correctness -----------------------------------------------

    #[test]
    fn conv2d_stride2_correctness() {
        let dev = GpuDevice::new(0).expect("CUDA device 0");

        let input_shape = [1, 1, 6, 6];
        let weight_shape = [1, 1, 3, 3];
        let stride = (2, 2);
        let padding = (0, 0);

        let input_data: Vec<f32> = (0..36).map(|i| i as f32).collect();
        let weight_data: Vec<f32> = vec![
            1.0, 0.0, -1.0,
            2.0, 0.0, -2.0,
            1.0, 0.0, -1.0,
        ];

        let (expected_output, expected_shape) = cpu_conv2d_reference(
            &input_data,
            &weight_data,
            None,
            input_shape,
            weight_shape,
            stride,
            padding,
        );

        let input_gpu = cpu_to_gpu(&input_data, &dev).expect("input to gpu");
        let weight_gpu = cpu_to_gpu(&weight_data, &dev).expect("weight to gpu");

        let (out_gpu, out_shape) = gpu_conv2d_f32(
            &input_gpu,
            &weight_gpu,
            None,
            input_shape,
            weight_shape,
            stride,
            padding,
            &dev,
        )
        .expect("gpu_conv2d_f32");

        assert_eq!(out_shape, expected_shape);
        assert_eq!(out_shape, [1, 1, 2, 2]);

        let out_host = gpu_to_cpu(&out_gpu, &dev).expect("gpu_to_cpu");
        assert_close(&out_host, &expected_output, 1e-4, "conv2d stride 2");
    }
}
