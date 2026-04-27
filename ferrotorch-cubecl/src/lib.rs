//! Portable GPU backend for ferrotorch via CubeCL.
//!
//! CubeCL compiles a single kernel definition to CUDA PTX, AMD HIP/ROCm, and
//! WGPU (Vulkan/Metal/DX12). This crate wraps CubeCL's runtime and dispatches
//! real `#[cube]` kernels to the active backend — no CPU fallbacks.
//!
//! # Feature flags
//!
//! | Feature | Backend              | GPU vendors            |
//! |---------|----------------------|------------------------|
//! | `cuda`  | NVIDIA CUDA via PTX  | NVIDIA                 |
//! | `wgpu`  | WGPU (Vulkan/Metal)  | AMD, Intel, Apple, ... |
//! | `rocm`  | AMD HIP (native)     | AMD                    |
//!
//! Enable at least one backend feature to use GPU acceleration. Without any
//! backend feature [`CubeRuntime::new`] returns
//! `FerrotorchError::DeviceUnavailable` and [`CubeRuntime::auto`] returns
//! `None`.
//!
//! # Example
//!
//! ```rust,no_run
//! use ferrotorch_cubecl::{CubeDevice, CubeRuntime};
//!
//! // Auto-detect the best available backend
//! if let Some(rt) = CubeRuntime::auto() {
//!     println!("Using device: {:?}", rt.device());
//! }
//! ```

pub mod kernels;
pub mod ops;
pub mod quant;
pub mod runtime;

// Re-export runtime types.
pub use runtime::{CubeClient, CubeDevice, CubeRuntime};

// Re-export quantized-weight dequantization API.
pub use quant::{
    GgufBlockKind, dequantize_q4_0_to_gpu, dequantize_q4_1_to_gpu, dequantize_q5_0_to_gpu,
    dequantize_q5_1_to_gpu, dequantize_q8_0_to_gpu, dequantize_q8_1_to_gpu, split_q4_0_blocks,
    split_q4_1_blocks, split_q5_0_blocks, split_q5_1_blocks, split_q8_0_blocks, split_q8_1_blocks,
};
