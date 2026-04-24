//! Native bf16 GPU kernels.
//!
//! Hand-written PTX owned by Rust: no CUDA C++ source, no nvrtc
//! runtime compiler, no external toolchain at load time. Each kernel
//! is a `&'static str` containing PTX 7.0 targeting sm_52+, loaded
//! via `cudarc::driver::CudaContext::load_module` through the
//! existing `module_cache::get_or_compile` path.
//!
//! # The bf16 pattern (sm < 90)
//!
//! bf16 is the top 16 bits of an f32 with zero-padded low bits, so
//! in-register conversions are pure bit operations:
//!
//! - **bf16 → f32**: `mov.b32 %f, {0, %h}` where `%h` is a `.b16` and
//!   `%f` can be consumed as `.f32`.  This pattern is taken directly
//!   from NVIDIA's `cuda_bf16.hpp` (`__internal_bfloat162float`) and
//!   from PyTorch's `c10::detail::f32_from_bits` (`tmp <<= 16;
//!   memcpy(&res, &tmp)`).  It is lossless.
//!
//! - **f32 → bf16, round-to-nearest-even**: add the rounding bias
//!   `0x7FFF + bit[16]` to the f32 bits, then shift right 16.  Same
//!   pattern as the existing `F32_TO_BF16_PTX` and as PyTorch's
//!   `round_to_nearest_even` in `BFloat16.h`.
//!
//! All arithmetic happens in `.f32` registers per thread; storage is
//! always `u16` (`.b16`) in global memory.  No whole-tensor f32
//! intermediate materialisation.

#![cfg(feature = "cuda")]

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::device::GpuDevice;
use crate::error::{GpuError, GpuResult};
use crate::module_cache::get_or_compile;

const BLOCK_SIZE: u32 = 256;

// ===========================================================================
// Elementwise kernels (mul, add, silu)
// ===========================================================================

const MUL_BF16_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry mul_bf16_kernel(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %b, %out, %off;
    .reg .b16 %a_b16, %b_b16, %zero16;
    .reg .b32 %a_u32, %b_u32, %bits, %round, %lsb;
    .reg .f32 %va, %vb, %vr;
    .reg .pred %p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %b, [b_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 1;
    add.u64 %a, %a, %off;
    add.u64 %b, %b, %off;
    add.u64 %out, %out, %off;

    ld.global.b16 %a_b16, [%a];
    ld.global.b16 %b_b16, [%b];
    mov.b16 %zero16, 0;
    mov.b32 %a_u32, {%zero16, %a_b16};
    mov.b32 %b_u32, {%zero16, %b_b16};
    mov.b32 %va, %a_u32;
    mov.b32 %vb, %b_u32;

    mul.f32 %vr, %va, %vb;

    mov.b32 %bits, %vr;
    shr.u32 %lsb, %bits, 16;
    and.b32 %lsb, %lsb, 1;
    add.u32 %round, %bits, 0x7FFF;
    add.u32 %round, %round, %lsb;
    shr.u32 %bits, %round, 16;
    st.global.u16 [%out], %bits;

DONE:
    ret;
}
";

const ADD_BF16_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry add_bf16_kernel(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %b, %out, %off;
    .reg .b16 %a_b16, %b_b16, %zero16;
    .reg .b32 %a_u32, %b_u32, %bits, %round, %lsb;
    .reg .f32 %va, %vb, %vr;
    .reg .pred %p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %b, [b_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 1;
    add.u64 %a, %a, %off;
    add.u64 %b, %b, %off;
    add.u64 %out, %out, %off;

    ld.global.b16 %a_b16, [%a];
    ld.global.b16 %b_b16, [%b];
    mov.b16 %zero16, 0;
    mov.b32 %a_u32, {%zero16, %a_b16};
    mov.b32 %b_u32, {%zero16, %b_b16};
    mov.b32 %va, %a_u32;
    mov.b32 %vb, %b_u32;

    add.f32 %vr, %va, %vb;

    mov.b32 %bits, %vr;
    shr.u32 %lsb, %bits, 16;
    and.b32 %lsb, %lsb, 1;
    add.u32 %round, %bits, 0x7FFF;
    add.u32 %round, %round, %lsb;
    shr.u32 %bits, %round, 16;
    st.global.u16 [%out], %bits;

DONE:
    ret;
}
";

// SiLU: x * sigmoid(x) = x / (1 + exp(-x))
// sigmoid computed via ex2.approx.f32 on -x * log2(e).
const SILU_BF16_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry silu_bf16_kernel(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off;
    .reg .b16 %a_b16, %zero16;
    .reg .b32 %a_u32, %bits, %round, %lsb;
    .reg .f32 %va, %neg_a, %log2e, %x, %e, %one, %denom, %sig, %vr;
    .reg .pred %p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 1;
    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.b16 %a_b16, [%a];
    mov.b16 %zero16, 0;
    mov.b32 %a_u32, {%zero16, %a_b16};
    mov.b32 %va, %a_u32;

    neg.f32 %neg_a, %va;
    mov.f32 %log2e, 0f3FB8AA3B;
    mul.f32 %x, %neg_a, %log2e;
    ex2.approx.f32 %e, %x;
    mov.f32 %one, 0f3F800000;
    add.f32 %denom, %one, %e;
    div.approx.f32 %sig, %one, %denom;
    mul.f32 %vr, %va, %sig;

    mov.b32 %bits, %vr;
    shr.u32 %lsb, %bits, 16;
    and.b32 %lsb, %lsb, 1;
    add.u32 %round, %bits, 0x7FFF;
    add.u32 %round, %round, %lsb;
    shr.u32 %bits, %round, 16;
    st.global.u16 [%out], %bits;

DONE:
    ret;
}
";

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
    ptx: &'static str,
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
    let f = get_or_compile(ctx, ptx, kernel_name, device.ordinal() as u32)
        .map_err(|e| {
            eprintln!("{kernel_name}: {e}");
            GpuError::PtxCompileFailed { kernel: kernel_name }
        })?;

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
    launch_binary(a, b, device, MUL_BF16_PTX, "mul_bf16_kernel")
}

/// Elementwise `out = a + b` on bf16 (u16-stored) GPU buffers.
pub fn gpu_add_bf16(
    a: &cudarc::driver::CudaSlice<u16>,
    b: &cudarc::driver::CudaSlice<u16>,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    launch_binary(a, b, device, ADD_BF16_PTX, "add_bf16_kernel")
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
    let f = get_or_compile(ctx, SILU_BF16_PTX, "silu_bf16_kernel", device.ordinal() as u32)
        .map_err(|e| {
            eprintln!("silu_bf16_kernel: {e}");
            GpuError::PtxCompileFailed { kernel: "silu_bf16_kernel" }
        })?;

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

// One block per output token; threads stride over `dim` copying u16
// elements. No arithmetic, no precision concerns.
const EMBEDDING_GATHER_BF16_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry embedding_gather_bf16_kernel(
    .param .u64 weight_ptr,
    .param .u64 indices_ptr,
    .param .u64 out_ptr,
    .param .u32 n_tokens,
    .param .u32 dim
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg, %dim_reg, %src_row, %col, %src_elem, %dst_elem;
    .reg .u64 %weight, %indices, %out, %off;
    .reg .b16 %v16;
    .reg .pred %p_tok, %p_col;

    ld.param.u64 %weight, [weight_ptr];
    ld.param.u64 %indices, [indices_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n_tokens];
    ld.param.u32 %dim_reg, [dim];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;

    setp.ge.u32 %p_tok, %bid, %n_reg;
    @%p_tok bra DONE;

    cvt.u64.u32 %off, %bid;
    shl.b64 %off, %off, 2;
    add.u64 %indices, %indices, %off;
    ld.global.u32 %src_row, [%indices];

    mov.u32 %col, %r_tid;
LOOP:
    setp.ge.u32 %p_col, %col, %dim_reg;
    @%p_col bra DONE;

    mul.lo.u32 %src_elem, %src_row, %dim_reg;
    add.u32 %src_elem, %src_elem, %col;
    mul.lo.u32 %dst_elem, %bid, %dim_reg;
    add.u32 %dst_elem, %dst_elem, %col;

    cvt.u64.u32 %off, %src_elem;
    shl.b64 %off, %off, 1;
    add.u64 %off, %weight, %off;
    ld.global.b16 %v16, [%off];

    cvt.u64.u32 %off, %dst_elem;
    shl.b64 %off, %off, 1;
    add.u64 %off, %out, %off;
    st.global.b16 [%off], %v16;

    add.u32 %col, %col, %bdim;
    bra LOOP;

DONE:
    ret;
}
";

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
    let f = get_or_compile(
        ctx,
        EMBEDDING_GATHER_BF16_PTX,
        "embedding_gather_bf16_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| {
        eprintln!("embedding_gather_bf16_kernel: {e}");
        GpuError::PtxCompileFailed {
            kernel: "embedding_gather_bf16_kernel",
        }
    })?;

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
// RMSNorm — per-row, f32 accumulator, tree reduction in shared memory
// ===========================================================================

// One block per row. Each thread strides over `cols`, accumulating
// sum(x^2) in f32, storing partials to shared memory, then reducing
// via tree-sum. Second pass multiplies by inv_rms and bf16 weight.
const RMSNORM_BF16_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.shared .align 4 .f32 rmsnorm_bf16_sdata[256];

.visible .entry rmsnorm_bf16_kernel(
    .param .u64 in_ptr,
    .param .u64 w_ptr,
    .param .u64 out_ptr,
    .param .u32 rows,
    .param .u32 cols,
    .param .f32 eps
) {
    .reg .u32 %r_tid, %bid, %bdim, %rows_reg, %cols_reg, %j, %half, %otid;
    .reg .u64 %in, %w, %out, %row_off, %off, %sbase, %saddr;
    .reg .b16 %x_b16, %w_b16, %zero16;
    .reg .b32 %x_u32, %w_u32, %bits, %round, %lsb;
    .reg .f32 %x_f, %w_f, %sq_sum, %eps_r, %inv_rms, %mean_sq, %r_f, %r_w, %other, %n_f;
    .reg .pred %p, %lp, %rp;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %w, [w_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %rows_reg, [rows];
    ld.param.u32 %cols_reg, [cols];
    ld.param.f32 %eps_r, [eps];

    mov.u64 %sbase, rmsnorm_bf16_sdata;

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;

    setp.ge.u32 %p, %bid, %rows_reg;
    @%p bra DONE;

    cvt.u64.u32 %row_off, %bid;
    cvt.u64.u32 %off, %cols_reg;
    mul.lo.u64 %row_off, %row_off, %off;
    shl.b64 %row_off, %row_off, 1;
    cvt.rn.f32.u32 %n_f, %cols_reg;

    mov.b16 %zero16, 0;

    // Phase 1: sum(x^2) in f32
    mov.f32 %sq_sum, 0f00000000;
    mov.u32 %j, %r_tid;
SS:
    setp.ge.u32 %lp, %j, %cols_reg;
    @%lp bra SSD;
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 1;
    add.u64 %off, %in, %off;
    add.u64 %off, %off, %row_off;
    ld.global.b16 %x_b16, [%off];
    mov.b32 %x_u32, {%zero16, %x_b16};
    mov.b32 %x_f, %x_u32;
    fma.rn.f32 %sq_sum, %x_f, %x_f, %sq_sum;
    add.u32 %j, %j, %bdim;
    bra SS;
SSD:
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    st.shared.f32 [%saddr], %sq_sum;
    bar.sync 0;

    mov.u32 %half, %bdim;
SR:
    shr.u32 %half, %half, 1;
    setp.eq.u32 %rp, %half, 0;
    @%rp bra SRD;
    setp.ge.u32 %rp, %r_tid, %half;
    @%rp bra SRS;
    add.u32 %otid, %r_tid, %half;
    cvt.u64.u32 %off, %otid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %other, [%saddr];
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %sq_sum, [%saddr];
    add.f32 %sq_sum, %sq_sum, %other;
    st.shared.f32 [%saddr], %sq_sum;
SRS:
    bar.sync 0;
    bra SR;
SRD:
    ld.shared.f32 %sq_sum, [%sbase];
    div.approx.f32 %mean_sq, %sq_sum, %n_f;
    add.f32 %mean_sq, %mean_sq, %eps_r;
    sqrt.approx.f32 %inv_rms, %mean_sq;
    rcp.approx.f32 %inv_rms, %inv_rms;
    bar.sync 0;

    // Phase 2: out = x * inv_rms * weight, rounded to bf16
    mov.u32 %j, %r_tid;
NM:
    setp.ge.u32 %lp, %j, %cols_reg;
    @%lp bra NMD;
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 1;
    add.u64 %off, %in, %off;
    add.u64 %off, %off, %row_off;
    ld.global.b16 %x_b16, [%off];
    mov.b32 %x_u32, {%zero16, %x_b16};
    mov.b32 %x_f, %x_u32;

    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 1;
    add.u64 %off, %w, %off;
    ld.global.b16 %w_b16, [%off];
    mov.b32 %w_u32, {%zero16, %w_b16};
    mov.b32 %w_f, %w_u32;

    mul.f32 %r_f, %x_f, %inv_rms;
    mul.f32 %r_f, %r_f, %w_f;

    mov.b32 %bits, %r_f;
    shr.u32 %lsb, %bits, 16;
    and.b32 %lsb, %lsb, 1;
    add.u32 %round, %bits, 0x7FFF;
    add.u32 %round, %round, %lsb;
    shr.u32 %bits, %round, 16;

    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 1;
    add.u64 %off, %out, %off;
    add.u64 %off, %off, %row_off;
    st.global.u16 [%off], %bits;
    add.u32 %j, %j, %bdim;
    bra NM;
NMD:

DONE:
    ret;
}
";

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
    let f = get_or_compile(
        ctx,
        RMSNORM_BF16_PTX,
        "rmsnorm_bf16_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| {
        eprintln!("rmsnorm_bf16_kernel: {e}");
        GpuError::PtxCompileFailed {
            kernel: "rmsnorm_bf16_kernel",
        }
    })?;

    let mut out = stream.alloc_zeros::<u16>(rows * cols)?;
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
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
// Softmax — row-wise, f32 accumulator, two-pass tree reduction
// ===========================================================================

// One block per row. Pass 1: thread-local max, then shared-memory
// tree-max. Pass 2: thread-local sum of exp(v - row_max), then
// shared-memory tree-sum. Pass 3: write exp((v-row_max) * inv_sum)
// rounded to bf16.
const SOFTMAX_BF16_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.shared .align 4 .f32 softmax_bf16_sdata[256];

.visible .entry softmax_bf16_kernel(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 rows,
    .param .u32 cols
) {
    .reg .u32 %r_tid, %bid, %bdim, %rows_reg, %cols_reg, %j, %half, %otid;
    .reg .u64 %in, %out, %row_off, %off, %sbase, %saddr;
    .reg .b16 %x_b16, %zero16;
    .reg .b32 %x_u32, %bits, %round, %lsb;
    .reg .f32 %x_f, %tmax, %other, %row_max, %sum, %inv_sum, %e, %scale, %log2e, %y_f;
    .reg .pred %p, %lp, %rp, %gp;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %rows_reg, [rows];
    ld.param.u32 %cols_reg, [cols];

    mov.u64 %sbase, softmax_bf16_sdata;
    mov.f32 %log2e, 0f3FB8AA3B;

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;

    setp.ge.u32 %p, %bid, %rows_reg;
    @%p bra DONE;

    cvt.u64.u32 %row_off, %bid;
    cvt.u64.u32 %off, %cols_reg;
    mul.lo.u64 %row_off, %row_off, %off;
    shl.b64 %row_off, %row_off, 1;

    mov.b16 %zero16, 0;

    // Pass 1: thread-local max
    mov.f32 %tmax, 0fFF800000;   // -Inf
    mov.u32 %j, %r_tid;
MX:
    setp.ge.u32 %lp, %j, %cols_reg;
    @%lp bra MXD;
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 1;
    add.u64 %off, %in, %off;
    add.u64 %off, %off, %row_off;
    ld.global.b16 %x_b16, [%off];
    mov.b32 %x_u32, {%zero16, %x_b16};
    mov.b32 %x_f, %x_u32;
    setp.gt.f32 %gp, %x_f, %tmax;
    @%gp mov.f32 %tmax, %x_f;
    add.u32 %j, %j, %bdim;
    bra MX;
MXD:
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    st.shared.f32 [%saddr], %tmax;
    bar.sync 0;

    mov.u32 %half, %bdim;
MR:
    shr.u32 %half, %half, 1;
    setp.eq.u32 %rp, %half, 0;
    @%rp bra MRD;
    setp.ge.u32 %rp, %r_tid, %half;
    @%rp bra MRS;
    add.u32 %otid, %r_tid, %half;
    cvt.u64.u32 %off, %otid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %other, [%saddr];
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %tmax, [%saddr];
    setp.gt.f32 %gp, %other, %tmax;
    @%gp mov.f32 %tmax, %other;
    st.shared.f32 [%saddr], %tmax;
MRS:
    bar.sync 0;
    bra MR;
MRD:
    ld.shared.f32 %row_max, [%sbase];
    bar.sync 0;

    // Pass 2: thread-local sum of exp(v - row_max)
    mov.f32 %sum, 0f00000000;
    mov.u32 %j, %r_tid;
SE:
    setp.ge.u32 %lp, %j, %cols_reg;
    @%lp bra SED;
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 1;
    add.u64 %off, %in, %off;
    add.u64 %off, %off, %row_off;
    ld.global.b16 %x_b16, [%off];
    mov.b32 %x_u32, {%zero16, %x_b16};
    mov.b32 %x_f, %x_u32;
    sub.f32 %x_f, %x_f, %row_max;
    mul.f32 %scale, %x_f, %log2e;
    ex2.approx.f32 %e, %scale;
    add.f32 %sum, %sum, %e;
    add.u32 %j, %j, %bdim;
    bra SE;
SED:
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    st.shared.f32 [%saddr], %sum;
    bar.sync 0;

    mov.u32 %half, %bdim;
SER:
    shr.u32 %half, %half, 1;
    setp.eq.u32 %rp, %half, 0;
    @%rp bra SERD;
    setp.ge.u32 %rp, %r_tid, %half;
    @%rp bra SERS;
    add.u32 %otid, %r_tid, %half;
    cvt.u64.u32 %off, %otid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %other, [%saddr];
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %sum, [%saddr];
    add.f32 %sum, %sum, %other;
    st.shared.f32 [%saddr], %sum;
SERS:
    bar.sync 0;
    bra SER;
SERD:
    ld.shared.f32 %sum, [%sbase];
    rcp.approx.f32 %inv_sum, %sum;
    bar.sync 0;

    // Pass 3: write
    mov.u32 %j, %r_tid;
WR:
    setp.ge.u32 %lp, %j, %cols_reg;
    @%lp bra WRD;
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 1;
    add.u64 %off, %in, %off;
    add.u64 %off, %off, %row_off;
    ld.global.b16 %x_b16, [%off];
    mov.b32 %x_u32, {%zero16, %x_b16};
    mov.b32 %x_f, %x_u32;
    sub.f32 %x_f, %x_f, %row_max;
    mul.f32 %scale, %x_f, %log2e;
    ex2.approx.f32 %e, %scale;
    mul.f32 %y_f, %e, %inv_sum;

    mov.b32 %bits, %y_f;
    shr.u32 %lsb, %bits, 16;
    and.b32 %lsb, %lsb, 1;
    add.u32 %round, %bits, 0x7FFF;
    add.u32 %round, %round, %lsb;
    shr.u32 %bits, %round, 16;

    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 1;
    add.u64 %off, %out, %off;
    add.u64 %off, %off, %row_off;
    st.global.u16 [%off], %bits;
    add.u32 %j, %j, %bdim;
    bra WR;
WRD:

DONE:
    ret;
}
";

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
    let f = get_or_compile(
        ctx,
        SOFTMAX_BF16_PTX,
        "softmax_bf16_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| {
        eprintln!("softmax_bf16_kernel: {e}");
        GpuError::PtxCompileFailed {
            kernel: "softmax_bf16_kernel",
        }
    })?;

    let mut out = stream.alloc_zeros::<u16>(rows * cols)?;
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
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
// RoPE (half-rotation / Llama convention)
// ===========================================================================

// One thread per (head, pos, d<half_dim). Rotates the pair
// (d, d + half_dim) using cos/sin from the precomputed caches.
const ROPE_HALF_BF16_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry rope_half_bf16_kernel(
    .param .u64 in_ptr,
    .param .u64 cos_ptr,
    .param .u64 sin_ptr,
    .param .u64 out_ptr,
    .param .u32 num_heads,
    .param .u32 seq_len,
    .param .u32 head_dim,
    .param .u32 seq_offset
) {
    .reg .u32 %r_tid, %bid, %bdim, %gid, %nh_reg, %sl_reg, %hd_reg, %so_reg, %half_dim;
    .reg .u32 %d, %tmp, %pos, %head, %cs_idx, %base, %cs_base, %total;
    .reg .u64 %in, %cos_p, %sin_p, %out, %off, %off_base;
    .reg .b16 %x0_b16, %x1_b16, %c_b16, %s_b16, %zero16;
    .reg .b32 %x0_u, %x1_u, %c_u, %s_u, %bits0, %bits1, %round, %lsb;
    .reg .f32 %x0, %x1, %c, %s, %y0, %y1;
    .reg .pred %p;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %cos_p, [cos_ptr];
    ld.param.u64 %sin_p, [sin_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %nh_reg, [num_heads];
    ld.param.u32 %sl_reg, [seq_len];
    ld.param.u32 %hd_reg, [head_dim];
    ld.param.u32 %so_reg, [seq_offset];

    shr.u32 %half_dim, %hd_reg, 1;

    // total = num_heads * seq_len * half_dim
    mul.lo.u32 %total, %nh_reg, %sl_reg;
    mul.lo.u32 %total, %total, %half_dim;

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %gid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %gid, %total;
    @%p bra DONE;

    // d = gid % half_dim, tmp = gid / half_dim, pos = tmp % seq_len, head = tmp / seq_len
    rem.u32 %d, %gid, %half_dim;
    div.u32 %tmp, %gid, %half_dim;
    rem.u32 %pos, %tmp, %sl_reg;
    div.u32 %head, %tmp, %sl_reg;

    // base = head * seq_len * head_dim + pos * head_dim
    mul.lo.u32 %base, %head, %sl_reg;
    mul.lo.u32 %base, %base, %hd_reg;
    mul.lo.u32 %tmp, %pos, %hd_reg;
    add.u32 %base, %base, %tmp;

    // cs_idx = (seq_offset + pos) * half_dim + d
    add.u32 %cs_base, %so_reg, %pos;
    mul.lo.u32 %cs_idx, %cs_base, %half_dim;
    add.u32 %cs_idx, %cs_idx, %d;

    mov.b16 %zero16, 0;

    // Load input[base + d] and input[base + d + half_dim]
    add.u32 %tmp, %base, %d;
    cvt.u64.u32 %off, %tmp;
    shl.b64 %off, %off, 1;
    add.u64 %off_base, %in, %off;
    ld.global.b16 %x0_b16, [%off_base];

    add.u32 %tmp, %tmp, %half_dim;
    cvt.u64.u32 %off, %tmp;
    shl.b64 %off, %off, 1;
    add.u64 %off_base, %in, %off;
    ld.global.b16 %x1_b16, [%off_base];

    // Load cos/sin
    cvt.u64.u32 %off, %cs_idx;
    shl.b64 %off, %off, 1;
    add.u64 %off_base, %cos_p, %off;
    ld.global.b16 %c_b16, [%off_base];
    add.u64 %off_base, %sin_p, %off;
    ld.global.b16 %s_b16, [%off_base];

    // Upcast to f32
    mov.b32 %x0_u, {%zero16, %x0_b16};
    mov.b32 %x1_u, {%zero16, %x1_b16};
    mov.b32 %c_u, {%zero16, %c_b16};
    mov.b32 %s_u, {%zero16, %s_b16};
    mov.b32 %x0, %x0_u;
    mov.b32 %x1, %x1_u;
    mov.b32 %c, %c_u;
    mov.b32 %s, %s_u;

    // y0 = x0*c - x1*s;  y1 = x1*c + x0*s
    mul.f32 %y0, %x0, %c;
    fma.rn.f32 %y0, %x1, 0fBF800000, %y0;   // y0 -= x1 -- wrong, need s factor
    // Redo properly using fma: y0 = x0*c - x1*s = fma(-x1, s, x0*c)
    mul.f32 %y0, %x0, %c;
    neg.f32 %y1, %s;
    fma.rn.f32 %y0, %x1, %y1, %y0;
    mul.f32 %y1, %x1, %c;
    fma.rn.f32 %y1, %x0, %s, %y1;

    // Round-and-store y0 at (base + d)
    mov.b32 %bits0, %y0;
    shr.u32 %lsb, %bits0, 16;
    and.b32 %lsb, %lsb, 1;
    add.u32 %round, %bits0, 0x7FFF;
    add.u32 %round, %round, %lsb;
    shr.u32 %bits0, %round, 16;

    add.u32 %tmp, %base, %d;
    cvt.u64.u32 %off, %tmp;
    shl.b64 %off, %off, 1;
    add.u64 %off_base, %out, %off;
    st.global.u16 [%off_base], %bits0;

    // Round-and-store y1 at (base + d + half_dim)
    mov.b32 %bits1, %y1;
    shr.u32 %lsb, %bits1, 16;
    and.b32 %lsb, %lsb, 1;
    add.u32 %round, %bits1, 0x7FFF;
    add.u32 %round, %round, %lsb;
    shr.u32 %bits1, %round, 16;

    add.u32 %tmp, %tmp, %half_dim;
    cvt.u64.u32 %off, %tmp;
    shl.b64 %off, %off, 1;
    add.u64 %off_base, %out, %off;
    st.global.u16 [%off_base], %bits1;

DONE:
    ret;
}
";

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
    let f = get_or_compile(
        ctx,
        ROPE_HALF_BF16_PTX,
        "rope_half_bf16_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| {
        eprintln!("rope_half_bf16_kernel: {e}");
        GpuError::PtxCompileFailed {
            kernel: "rope_half_bf16_kernel",
        }
    })?;

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
    fn mul_add_silu_bf16_hand_ptx() {
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
        let x: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.25) - 1.0).collect();
        let w: Vec<f32> = (0..cols).map(|i| 1.0 + (i as f32) * 0.125).collect();
        let input = upload_bf16(&dev, &x);
        let weight = upload_bf16(&dev, &w);
        let out = gpu_rmsnorm_bf16(&input, &weight, rows, cols, 1e-5, &dev).expect("rmsnorm");
        let got = download_bf16(&dev, &out);

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
                (g - e).abs() < e.abs() * 0.03 + 8e-3,
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
                    (g - e).abs() < e.abs() * 0.04 + 5e-3,
                    "softmax[{r},{i}]: got {g}, expected {e}",
                );
            }
        }
    }

    #[test]
    fn rope_half_bf16_identity_at_pos_zero() {
        let dev = GpuDevice::new(0).expect("cuda");
        let num_heads = 2usize;
        let seq_len = 1usize;
        let head_dim = 8usize;
        let input: Vec<f32> = (0..num_heads * seq_len * head_dim)
            .map(|i| (i as f32) * 0.125 - 0.5)
            .collect();

        let max_seq = 4;
        let half_dim = head_dim / 2;
        let mut cos_buf = vec![0.0f32; max_seq * half_dim];
        let sin_buf = vec![0.0f32; max_seq * half_dim];
        for d in 0..half_dim {
            cos_buf[d] = 1.0;
        }

        let input_g = upload_bf16(&dev, &input);
        let cos_g = upload_bf16(&dev, &cos_buf);
        let sin_g = upload_bf16(&dev, &sin_buf);

        let out = gpu_rope_half_bf16(
            &input_g, &cos_g, &sin_g, num_heads, seq_len, head_dim, 0, &dev,
        )
        .expect("rope");
        let got = download_bf16(&dev, &out);

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
