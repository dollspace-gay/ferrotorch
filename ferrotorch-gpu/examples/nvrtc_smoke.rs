//! Quick nvrtc smoke: can we compile a CUDA C++ kernel with `__nv_bfloat16`?
#[cfg(feature = "cuda")]
fn main() {
    let src = r#"
#include <cuda_bf16.h>

extern "C" __global__ void mul_bf16_kernel(
    const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float af = __bfloat162float(a[i]);
    float bf = __bfloat162float(b[i]);
    out[i] = __float2bfloat16(af * bf);
}
"#;
    let opts = cudarc::nvrtc::CompileOptions {
        include_paths: vec!["/usr/local/cuda-12.8/include".to_string()],
        ..Default::default()
    };
    match cudarc::nvrtc::compile_ptx_with_opts(src, opts) {
        Ok(ptx) => {
            let s = ptx.to_src();
            println!("OK: compiled {} bytes of PTX", s.len());
            println!("first lines:");
            for l in s.lines().take(5) {
                println!("  {l}");
            }
        }
        Err(e) => {
            println!("nvrtc compile failed:\n{e:?}");
        }
    }
}
#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!(
        "nvrtc_smoke: built without the `cuda` feature; \
         rebuild `ferrotorch-gpu` with --features cuda to exercise nvrtc."
    );
    std::process::exit(1);
}
