//! Native bf16 GPU kernels written as CUDA C++ and compiled via nvrtc.
//!
//! Follows the PyTorch approach: one CUDA C++ source per kernel, loaded
//! through `__nv_bfloat16` / `__bfloat162float` / `__float2bfloat16`
//! intrinsics from `<cuda_bf16.h>`. The storage type is bf16
//! (`CudaSlice<u16>` on the Rust side, `__nv_bfloat16*` on the device);
//! the compute / reduction type is `float`, picked on each load and
//! rounded back on each store.  No whole-tensor f32 materialisation
//! anywhere in the forward pass.
//!
//! Each kernel entry point is declared `extern "C" __global__` so the
//! mangled symbol matches the `kernel_name` string.

#![cfg(feature = "cuda")]

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::device::GpuDevice;
use crate::error::{GpuError, GpuResult};
use crate::module_cache::get_or_compile_cuda;

/// How many threads per block for the row-wise reduction kernels
/// (RMSNorm, softmax). Matches the dim of Llama 3 8B's hidden size
/// (4096) divided by 16, i.e. every thread processes 16 elements on
/// the full forward.
const BLOCK_SIZE: u32 = 256;

fn to_cuda_err(
    kernel: &'static str,
    e: crate::module_cache::CudaCompileError,
) -> GpuError {
    eprintln!("{kernel}: {e}");
    match e {
        crate::module_cache::CudaCompileError::Driver(d) => GpuError::Driver(d),
        _ => GpuError::PtxCompileFailed { kernel },
    }
}

// ===========================================================================
// Elementwise kernels (mul, add, silu)
// ===========================================================================

const MUL_BF16_CU: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void mul_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ out,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float af = __bfloat162float(a[i]);
    float bf = __bfloat162float(b[i]);
    out[i] = __float2bfloat16(af * bf);
}
"#;

const ADD_BF16_CU: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void add_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ out,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float af = __bfloat162float(a[i]);
    float bf = __bfloat162float(b[i]);
    out[i] = __float2bfloat16(af + bf);
}
"#;

const SILU_BF16_CU: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void silu_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    __nv_bfloat16* __restrict__ out,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = __bfloat162float(a[i]);
    float sig = 1.0f / (1.0f + __expf(-x));
    out[i] = __float2bfloat16(x * sig);
}
"#;

fn launch_1d(n: usize) -> LaunchConfig {
    let grid = ((n as u32).saturating_add(BLOCK_SIZE - 1)) / BLOCK_SIZE;
    LaunchConfig {
        grid_dim: (grid.max(1), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn launch_binary(
    a: &cudarc::driver::CudaSlice<u16>,
    b: &cudarc::driver::CudaSlice<u16>,
    device: &GpuDevice,
    cuda_src: &'static str,
    kernel_name: &'static str,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    if a.len() != b.len() {
        return Err(GpuError::LengthMismatch {
            a: a.len(),
            b: b.len(),
        });
    }
    let n = a.len();
    if n == 0 {
        return Ok(device.stream().alloc_zeros::<u16>(0)?);
    }
    let ctx = device.context();
    let stream = device.stream();
    let f = get_or_compile_cuda(ctx, cuda_src, kernel_name, device.ordinal() as u32)
        .map_err(|e| to_cuda_err(kernel_name, e))?;

    let mut out = stream.alloc_zeros::<u16>(n)?;
    let cfg = launch_1d(n);
    let n_u32 = n as u32;
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a)
            .arg(b)
            .arg(&mut out)
            .arg(&n_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

/// Elementwise `out = a * b` on bf16 (u16-stored) GPU buffers.
pub fn gpu_mul_bf16(
    a: &cudarc::driver::CudaSlice<u16>,
    b: &cudarc::driver::CudaSlice<u16>,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    launch_binary(a, b, device, MUL_BF16_CU, "mul_bf16_kernel")
}

/// Elementwise `out = a + b` on bf16 (u16-stored) GPU buffers.
pub fn gpu_add_bf16(
    a: &cudarc::driver::CudaSlice<u16>,
    b: &cudarc::driver::CudaSlice<u16>,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    launch_binary(a, b, device, ADD_BF16_CU, "add_bf16_kernel")
}

/// Elementwise `out = silu(a) = a * sigmoid(a)` on bf16 GPU buffers.
pub fn gpu_silu_bf16(
    a: &cudarc::driver::CudaSlice<u16>,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    let n = a.len();
    if n == 0 {
        return Ok(device.stream().alloc_zeros::<u16>(0)?);
    }
    let ctx = device.context();
    let stream = device.stream();
    let f = get_or_compile_cuda(ctx, SILU_BF16_CU, "silu_bf16_kernel", device.ordinal() as u32)
        .map_err(|e| to_cuda_err("silu_bf16_kernel", e))?;

    let mut out = stream.alloc_zeros::<u16>(n)?;
    let cfg = launch_1d(n);
    let n_u32 = n as u32;
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a)
            .arg(&mut out)
            .arg(&n_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

// ===========================================================================
// Embedding gather
// ===========================================================================

const EMBEDDING_GATHER_BF16_CU: &str = r#"
#include <cuda_bf16.h>

// One block per output token; threads cooperate across `dim` columns.
extern "C" __global__ void embedding_gather_bf16_kernel(
    const __nv_bfloat16* __restrict__ weight,
    const unsigned int* __restrict__ indices,
    __nv_bfloat16* __restrict__ out,
    unsigned int n_tokens,
    unsigned int dim
) {
    unsigned int tok = blockIdx.x;
    if (tok >= n_tokens) return;
    unsigned int src_row = indices[tok];
    for (unsigned int c = threadIdx.x; c < dim; c += blockDim.x) {
        out[tok * dim + c] = weight[src_row * dim + c];
    }
}
"#;

/// `out[i, :] = weight[indices[i], :]`.  `weight` is `[vocab, dim]` bf16;
/// `indices` is `[n_tokens]` u32; `out` is `[n_tokens * dim]` bf16.
pub fn gpu_embedding_gather_bf16(
    weight: &cudarc::driver::CudaSlice<u16>,
    indices: &cudarc::driver::CudaSlice<u32>,
    dim: usize,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    let n_tokens = indices.len();
    if n_tokens == 0 || dim == 0 {
        return Ok(device.stream().alloc_zeros::<u16>(0)?);
    }
    let ctx = device.context();
    let stream = device.stream();
    let f = get_or_compile_cuda(
        ctx,
        EMBEDDING_GATHER_BF16_CU,
        "embedding_gather_bf16_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| to_cuda_err("embedding_gather_bf16_kernel", e))?;

    let total = n_tokens * dim;
    let mut out = stream.alloc_zeros::<u16>(total)?;
    let cfg = LaunchConfig {
        grid_dim: (n_tokens as u32, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_u32 = n_tokens as u32;
    let dim_u32 = dim as u32;
    unsafe {
        stream
            .launch_builder(&f)
            .arg(weight)
            .arg(indices)
            .arg(&mut out)
            .arg(&n_u32)
            .arg(&dim_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

// ===========================================================================
// RMSNorm (row-wise mean of squares → scale)
// ===========================================================================

const RMSNORM_BF16_CU: &str = r#"
#include <cuda_bf16.h>

// One block per row. Threads reduce a per-row mean-of-squares in shared
// memory using an f32 accumulator, then scale the row.
extern "C" __global__ void rmsnorm_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    unsigned int rows,
    unsigned int cols,
    float eps
) {
    extern __shared__ float shared[];
    unsigned int row = blockIdx.x;
    if (row >= rows) return;

    unsigned int tid = threadIdx.x;
    unsigned int stride = blockDim.x;

    float partial = 0.0f;
    for (unsigned int c = tid; c < cols; c += stride) {
        float x = __bfloat162float(input[row * cols + c]);
        partial += x * x;
    }
    shared[tid] = partial;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    float mean_sq = shared[0] / (float)cols;
    float inv_rms = rsqrtf(mean_sq + eps);

    for (unsigned int c = tid; c < cols; c += stride) {
        float x = __bfloat162float(input[row * cols + c]);
        float w = __bfloat162float(weight[c]);
        out[row * cols + c] = __float2bfloat16(x * inv_rms * w);
    }
}
"#;

/// `out[row, :] = (input[row, :] / rms(input[row, :])) * weight`.
pub fn gpu_rmsnorm_bf16(
    input: &cudarc::driver::CudaSlice<u16>,
    weight: &cudarc::driver::CudaSlice<u16>,
    rows: usize,
    cols: usize,
    eps: f32,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    if rows == 0 || cols == 0 {
        return Ok(device.stream().alloc_zeros::<u16>(rows * cols)?);
    }
    if input.len() < rows * cols {
        return Err(GpuError::ShapeMismatch {
            op: "rmsnorm_bf16",
            expected: vec![rows, cols],
            got: vec![input.len()],
        });
    }
    if weight.len() < cols {
        return Err(GpuError::ShapeMismatch {
            op: "rmsnorm_bf16",
            expected: vec![cols],
            got: vec![weight.len()],
        });
    }
    let ctx = device.context();
    let stream = device.stream();
    let f = get_or_compile_cuda(
        ctx,
        RMSNORM_BF16_CU,
        "rmsnorm_bf16_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| to_cuda_err("rmsnorm_bf16_kernel", e))?;

    let mut out = stream.alloc_zeros::<u16>(rows * cols)?;
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: BLOCK_SIZE * 4,
    };
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input)
            .arg(weight)
            .arg(&mut out)
            .arg(&rows_u32)
            .arg(&cols_u32)
            .arg(&eps)
            .launch(cfg)?;
    }
    Ok(out)
}

// ===========================================================================
// Softmax (row-wise over last axis, with optional causal mask offset)
// ===========================================================================

const SOFTMAX_BF16_CU: &str = r#"
#include <cuda_bf16.h>
#include <math_constants.h>

// Classic two-pass row softmax with f32 accumulator. One block per
// row; threads reduce max, then sum_exp, in shared memory.
extern "C" __global__ void softmax_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ out,
    unsigned int rows,
    unsigned int cols
) {
    extern __shared__ float shared[];
    unsigned int row = blockIdx.x;
    if (row >= rows) return;

    unsigned int tid = threadIdx.x;
    unsigned int stride = blockDim.x;

    // Pass 1: row max
    float thread_max = -CUDART_INF_F;
    for (unsigned int c = tid; c < cols; c += stride) {
        float v = __bfloat162float(input[row * cols + c]);
        if (v > thread_max) thread_max = v;
    }
    shared[tid] = thread_max;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float o = shared[tid + s];
            if (o > shared[tid]) shared[tid] = o;
        }
        __syncthreads();
    }
    float row_max = shared[0];
    __syncthreads();

    // Pass 2: sum_exp
    float thread_sum = 0.0f;
    for (unsigned int c = tid; c < cols; c += stride) {
        float v = __bfloat162float(input[row * cols + c]);
        thread_sum += __expf(v - row_max);
    }
    shared[tid] = thread_sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    float inv_sum = 1.0f / shared[0];
    __syncthreads();

    // Pass 3: write
    for (unsigned int c = tid; c < cols; c += stride) {
        float v = __bfloat162float(input[row * cols + c]);
        out[row * cols + c] = __float2bfloat16(__expf(v - row_max) * inv_sum);
    }
}
"#;

/// Row-wise softmax along the last axis. `input` is `[rows, cols]`.
pub fn gpu_softmax_bf16(
    input: &cudarc::driver::CudaSlice<u16>,
    rows: usize,
    cols: usize,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    if rows == 0 || cols == 0 {
        return Ok(device.stream().alloc_zeros::<u16>(rows * cols)?);
    }
    if input.len() < rows * cols {
        return Err(GpuError::ShapeMismatch {
            op: "softmax_bf16",
            expected: vec![rows, cols],
            got: vec![input.len()],
        });
    }
    let ctx = device.context();
    let stream = device.stream();
    let f = get_or_compile_cuda(
        ctx,
        SOFTMAX_BF16_CU,
        "softmax_bf16_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| to_cuda_err("softmax_bf16_kernel", e))?;

    let mut out = stream.alloc_zeros::<u16>(rows * cols)?;
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: BLOCK_SIZE * 4,
    };
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input)
            .arg(&mut out)
            .arg(&rows_u32)
            .arg(&cols_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

// ===========================================================================
// Rotary position embedding (half-rotation / Llama convention)
// ===========================================================================

const ROPE_HALF_BF16_CU: &str = r#"
#include <cuda_bf16.h>

// Applies half-rotation RoPE to a [heads, seq, head_dim] tensor.
// Pairs are (d, d + head_dim/2).  cos/sin caches are indexed by
// `(seq_offset + pos) * (head_dim/2) + d`.
extern "C" __global__ void rope_half_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ cos_cache,
    const __nv_bfloat16* __restrict__ sin_cache,
    __nv_bfloat16* __restrict__ out,
    unsigned int num_heads,
    unsigned int seq_len,
    unsigned int head_dim,
    unsigned int seq_offset
) {
    unsigned int half_dim = head_dim >> 1;
    unsigned int total = num_heads * seq_len * half_dim;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;

    unsigned int d = gid % half_dim;
    unsigned int tmp = gid / half_dim;
    unsigned int pos = tmp % seq_len;
    unsigned int head = tmp / seq_len;

    unsigned int base = head * seq_len * head_dim + pos * head_dim;
    unsigned int cs = (seq_offset + pos) * half_dim + d;

    float x0 = __bfloat162float(input[base + d]);
    float x1 = __bfloat162float(input[base + d + half_dim]);
    float c = __bfloat162float(cos_cache[cs]);
    float s = __bfloat162float(sin_cache[cs]);

    out[base + d] = __float2bfloat16(x0 * c - x1 * s);
    out[base + d + half_dim] = __float2bfloat16(x1 * c + x0 * s);
}
"#;

/// Half-rotation RoPE on `[heads, seq, head_dim]` bf16 tensor with
/// precomputed `cos_cache`, `sin_cache` of shape `[max_seq, head_dim/2]`.
/// `seq_offset` shifts into the cache for KV-cache continuation.
pub fn gpu_rope_half_bf16(
    input: &cudarc::driver::CudaSlice<u16>,
    cos_cache: &cudarc::driver::CudaSlice<u16>,
    sin_cache: &cudarc::driver::CudaSlice<u16>,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    seq_offset: usize,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    if head_dim == 0 || head_dim % 2 != 0 {
        return Err(GpuError::ShapeMismatch {
            op: "rope_half_bf16",
            expected: vec![head_dim],
            got: vec![head_dim],
        });
    }
    let total_io = num_heads * seq_len * head_dim;
    if input.len() < total_io {
        return Err(GpuError::ShapeMismatch {
            op: "rope_half_bf16",
            expected: vec![num_heads, seq_len, head_dim],
            got: vec![input.len()],
        });
    }
    let ctx = device.context();
    let stream = device.stream();
    let f = get_or_compile_cuda(
        ctx,
        ROPE_HALF_BF16_CU,
        "rope_half_bf16_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| to_cuda_err("rope_half_bf16_kernel", e))?;

    let half_dim = head_dim / 2;
    let total = num_heads * seq_len * half_dim;
    let mut out = stream.alloc_zeros::<u16>(total_io)?;
    let cfg = launch_1d(total);
    let (nh, sl, hd, so) = (
        num_heads as u32,
        seq_len as u32,
        head_dim as u32,
        seq_offset as u32,
    );
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input)
            .arg(cos_cache)
            .arg(sin_cache)
            .arg(&mut out)
            .arg(&nh)
            .arg(&sl)
            .arg(&hd)
            .arg(&so)
            .launch(cfg)?;
    }
    Ok(out)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn upload_bf16(dev: &GpuDevice, data: &[f32]) -> cudarc::driver::CudaSlice<u16> {
        let bits: Vec<u16> = data.iter().map(|&x| half::bf16::from_f32(x).to_bits()).collect();
        dev.stream().memcpy_stod(&bits).expect("upload bf16")
    }

    fn download_bf16(
        dev: &GpuDevice,
        buf: &cudarc::driver::CudaSlice<u16>,
    ) -> Vec<f32> {
        let bits = dev.stream().memcpy_dtov(buf).expect("download bf16");
        bits.into_iter()
            .map(|b| half::bf16::from_bits(b).to_f32())
            .collect()
    }

    #[test]
    fn mul_add_silu_bf16_nvrtc() {
        let dev = GpuDevice::new(0).expect("cuda device");
        let a = upload_bf16(&dev, &[1.0, 2.0, -3.0, 0.5, 4.0]);
        let b = upload_bf16(&dev, &[2.0, 3.0, 4.0, -1.0, 0.25]);

        let m = gpu_mul_bf16(&a, &b, &dev).expect("mul");
        let s = gpu_add_bf16(&a, &b, &dev).expect("add");
        let si = gpu_silu_bf16(&a, &dev).expect("silu");

        let m_host = download_bf16(&dev, &m);
        let s_host = download_bf16(&dev, &s);
        let si_host = download_bf16(&dev, &si);

        let m_exp = [2.0, 6.0, -12.0, -0.5, 1.0];
        let s_exp = [3.0, 5.0, 1.0, -0.5, 4.25];
        for (g, e) in m_host.iter().zip(m_exp.iter()) {
            assert!((g - e).abs() < e.abs() * 0.02 + 0.01, "mul {g} vs {e}");
        }
        for (g, e) in s_host.iter().zip(s_exp.iter()) {
            assert!((g - e).abs() < e.abs() * 0.02 + 0.01, "add {g} vs {e}");
        }
        let silu_ref: Vec<f32> = [1.0f32, 2.0, -3.0, 0.5, 4.0]
            .iter()
            .map(|&x| x * (1.0 / (1.0 + (-x).exp())))
            .collect();
        for (g, e) in si_host.iter().zip(silu_ref.iter()) {
            assert!(
                (g - e).abs() < e.abs() * 0.02 + 5e-3,
                "silu {g} vs {e}",
            );
        }
    }

    #[test]
    fn embedding_gather_bf16_picks_correct_rows() {
        let dev = GpuDevice::new(0).expect("cuda");
        let weight_f: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let weight = upload_bf16(&dev, &weight_f);
        let indices: Vec<u32> = vec![2, 0, 3];
        let idx = dev.stream().memcpy_stod(&indices).expect("indices");

        let out = gpu_embedding_gather_bf16(&weight, &idx, 3, &dev).expect("gather");
        let got = download_bf16(&dev, &out);
        assert_eq!(got, vec![6.0, 7.0, 8.0, 0.0, 1.0, 2.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn rmsnorm_bf16_matches_f32_ground_truth() {
        let dev = GpuDevice::new(0).expect("cuda");
        let rows = 2usize;
        let cols = 8usize;
        // Fill with known values.
        let x: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.25) - 1.0).collect();
        let w: Vec<f32> = (0..cols).map(|i| 1.0 + (i as f32) * 0.125).collect();
        let input = upload_bf16(&dev, &x);
        let weight = upload_bf16(&dev, &w);
        let out = gpu_rmsnorm_bf16(&input, &weight, rows, cols, 1e-5, &dev).expect("rmsnorm");
        let got = download_bf16(&dev, &out);

        // Reference computed in f32 with bf16 rounding of inputs (what
        // the kernel effectively sees after the __bfloat162float load).
        let x_bf: Vec<f32> = x.iter().map(|&v| half::bf16::from_f32(v).to_f32()).collect();
        let w_bf: Vec<f32> = w.iter().map(|&v| half::bf16::from_f32(v).to_f32()).collect();
        let mut expected = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            let row = &x_bf[r * cols..(r + 1) * cols];
            let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() / cols as f32;
            let inv_rms = (mean_sq + 1e-5).sqrt().recip();
            for c in 0..cols {
                expected.push(half::bf16::from_f32(row[c] * inv_rms * w_bf[c]).to_f32());
            }
        }
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < e.abs() * 0.02 + 5e-3,
                "rmsnorm[{i}]: got {g}, expected {e}",
            );
        }
    }

    #[test]
    fn softmax_bf16_rows_sum_to_one() {
        let dev = GpuDevice::new(0).expect("cuda");
        let rows = 3;
        let cols = 10;
        let input_f: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32 * 0.37).sin() + 1.0) * 2.0)
            .collect();
        let input = upload_bf16(&dev, &input_f);
        let out = gpu_softmax_bf16(&input, rows, cols, &dev).expect("softmax");
        let got = download_bf16(&dev, &out);

        for r in 0..rows {
            let row = &got[r * cols..(r + 1) * cols];
            let s: f32 = row.iter().sum();
            assert!(
                (s - 1.0).abs() < 0.05,
                "row {r} sum = {s}, expected ~1.0",
            );
            for &v in row {
                assert!(v >= 0.0, "softmax value {v} < 0");
            }
        }
    }

    #[test]
    fn softmax_bf16_matches_f32_ground_truth() {
        let dev = GpuDevice::new(0).expect("cuda");
        let rows = 2;
        let cols = 6;
        let input_f: Vec<f32> =
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5];
        let input = upload_bf16(&dev, &input_f);
        let out = gpu_softmax_bf16(&input, rows, cols, &dev).expect("softmax");
        let got = download_bf16(&dev, &out);

        for r in 0..rows {
            let row = &input_f[r * cols..(r + 1) * cols];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|&v| (v - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let expected: Vec<f32> = exps.into_iter().map(|e| e / sum).collect();
            for (i, (g, e)) in got[r * cols..(r + 1) * cols]
                .iter()
                .zip(expected.iter())
                .enumerate()
            {
                assert!(
                    (g - e).abs() < e.abs() * 0.03 + 5e-3,
                    "softmax[{r},{i}]: got {g}, expected {e}",
                );
            }
        }
    }

    #[test]
    fn rope_half_bf16_identity_at_pos_zero() {
        // At position 0 and seq_offset 0, cos=1, sin=0, so rotation
        // should be the identity.
        let dev = GpuDevice::new(0).expect("cuda");
        let num_heads = 2usize;
        let seq_len = 1usize;
        let head_dim = 8usize;
        let input: Vec<f32> = (0..num_heads * seq_len * head_dim)
            .map(|i| (i as f32) * 0.125 - 0.5)
            .collect();

        // cos_cache[0, d] = 1.0, sin_cache[0, d] = 0.0.
        let max_seq = 4;
        let half_dim = head_dim / 2;
        let mut cos_buf = vec![0.0f32; max_seq * half_dim];
        let sin_buf = vec![0.0f32; max_seq * half_dim];
        for d in 0..half_dim {
            cos_buf[0 * half_dim + d] = 1.0;
        }

        let input_g = upload_bf16(&dev, &input);
        let cos_g = upload_bf16(&dev, &cos_buf);
        let sin_g = upload_bf16(&dev, &sin_buf);

        let out = gpu_rope_half_bf16(
            &input_g, &cos_g, &sin_g, num_heads, seq_len, head_dim, 0, &dev,
        )
        .expect("rope");
        let got = download_bf16(&dev, &out);

        // bf16 round-trip of input values (captures the kernel's
        // effective input precision).
        let expected: Vec<f32> = input
            .iter()
            .map(|&v| half::bf16::from_f32(v).to_f32())
            .collect();
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-3,
                "rope identity[{i}]: got {g}, expected {e}",
            );
        }
    }
}
