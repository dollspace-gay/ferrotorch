# Audit: `ferrotorch-cubecl` + `ferrotorch-xpu` vs `torch.xpu` / `torch.mps` / ROCm backends

## ferrotorch-cubecl

Single-source GPU runtime targeting CUDA, AMD HIP/ROCm, and WGPU
(Vulkan/Metal/DX12) via [CubeCL](https://github.com/tracel-ai/cubecl).

### Modules

| Module | Role |
|---|---|
| `runtime` | `CubeClient`, `CubeDevice`, `CubeRuntime` |
| `kernels` | `#[cube]` kernels: kernel_add/sub/mul/div, kernel_relu/neg/abs/exp/ln/sqrt/sin/cos/tanh/sigmoid, kernel_matmul_naive |
| `ops` | High-level wrappers: `portable_add/sub/mul/relu/matmul` |
| `quant` | GGUF dequantization on GPU: `dequantize_q4_0/q4_1/q5_0/q5_1/q8_0/q8_1_to_gpu`, `split_qN_X_blocks` |
| `grammar` | Constrained-decoding token-mask compute via DFA: `DfaMaskInputs`, `compute_token_mask_dfa_to_gpu`, `kernel_compute_token_mask_dfa` |

### Coverage observations

- **Not a torch counterpart per se** — torch has no portable-kernel
  abstraction equivalent to CubeCL. Each torch backend (CUDA via ATen,
  ROCm via HIPIFY, MPS via Metal Performance Shaders, XPU via SYCL/oneAPI)
  is hand-written.
- ferrotorch's design choice: write kernels once in CubeCL → run on any
  backend. This is a meaningful improvement over torch's per-backend
  hand-rolled kernels.
- **Kernel count**: 14 elementwise kernels + 1 matmul + GGUF dequant +
  DFA mask. Compared to torch's per-backend kernel libraries (thousands of
  kernels each), this is **single-digit-percent coverage of op surface**.
- The portability buys breadth at the cost of depth — for the ops
  implemented, ferrotorch-cubecl runs on more hardware than any single
  torch backend.

### Strengths

- **GGUF dequantization on GPU** — directly relevant to LLM serving
  (load 4-bit weights, dequantize on device, no CPU bounce).
- **Constrained-decoding DFA mask** kernel — niche but essential for
  structured LLM output (JSON schema, regex constraints) where CPU mask
  generation would be a bottleneck.
- **Auto-detection** — `CubeRuntime::auto()` picks the best available
  backend at runtime.

### Gaps

The portable kernel set is **far smaller** than the CUDA-specific kernel
set in `ferrotorch-gpu`:

| Op family | ferrotorch-gpu (CUDA) | ferrotorch-cubecl (portable) |
|---|---|---|
| elementwise | many (incl. bf16) | 14 (f32 only — no bf16) |
| reductions | softmax, layernorm, rmsnorm, block_reduce | none |
| attention | flash_attention | none |
| matmul | cuBLAS f32/f64/bf16 (tn/nt/strided-batched) | naive matmul only |
| conv | cuDNN-backed conv2d | none |
| RoPE / repeat_kv / SDPA | yes | none |
| scatter / gather / embed_lookup | yes | none |
| transpose / permute | yes | none |
| dropout | yes | none |

The portable backend is a **technology demo / quantization helper / cross-
vendor accelerator path**, not yet a full GPU compute backend.

## ferrotorch-xpu

Intel XPU (Arc / DC GPU Max) backend, **wraps `ferrotorch-cubecl` with the
WGPU runtime**.

### Surface

`XpuDevice::new(ordinal)`, `XpuDevice::ordinal()`, `XpuDevice::device()`,
`XpuDevice::is_available()`, `XpuDevice::runtime()`. Plus macro-generated
ops:
- `xpu_binary!` × 5 (add, sub, mul, div, matmul)
- `xpu_unary!` × 9 (relu, exp, ln, sqrt, sin, cos, tanh, sigmoid, ...)

### Coverage vs `torch.xpu`

`torch.xpu` is Intel's official torch backend (SYCL/oneAPI). It exposes:
- device init, properties, `is_available`, `device_count`,
  `current_device`, `set_device`
- streams, events, synchronize
- caching allocator, memory_stats, empty_cache, set_per_process_memory_fraction
- random (manual_seed, get/set_rng_state)
- graph capture (`torch.xpu.graphs`)
- `torch.xpu.gpu_trace`

ferrotorch-xpu has only **device init + 14 macro-generated ops**.
**No streams, events, allocator stats, RNG, graph capture.**

Compared to `torch.xpu`'s breadth, ferrotorch-xpu is **~10% complete**.
This is acceptable for a "smoke test that XPU dispatch works" but not for
production use.

### Mechanism note

> Tensors with `Device::Xpu(ordinal)` live as CPU `Vec<T>` storage today;
> the device marker drives op dispatch through this crate, which uploads
> the inputs to the cubecl runtime, runs a real `#[cube]` kernel, and
> reads the result back.

This is a **roundtrip-on-every-op** design, which is correct but slow.
Native XPU storage (data resident on device between ops) is needed for
realistic perf.

## `torch.mps` (Apple Silicon)

ferrotorch has **no MPS backend at all** (#451 open).

If MPS lands, it should follow the same pattern as `ferrotorch-xpu`:
either via `ferrotorch-cubecl::wgpu` (Metal target) or a native
`metal-rs`/`mps-rs` integration.

## ROCm (AMD)

ferrotorch's ROCm story today: `ferrotorch-cubecl` with the `rocm`
feature exposes the HIP backend. There's **no `ferrotorch-rocm`
sister-crate** equivalent to `ferrotorch-gpu` (which is CUDA-specific).

torch has rocm support as a CUDA-API shim (HIPIFY) — torch.cuda calls
work on AMD with `HIP_VISIBLE_DEVICES`.

## Recommendations

1. **Decide the relationship between `ferrotorch-gpu` and
   `ferrotorch-cubecl`**:
   - Option A: keep them as today (gpu = CUDA-fast-path, cubecl = portable
     fallback for the ops cubecl supports).
   - Option B: rewrite `ferrotorch-gpu`'s kernels in CubeCL so they run on
     all backends, drop `ferrotorch-gpu` as a CUDA-only crate.
   - Option C: keep cudarc kernels in `ferrotorch-gpu` for the hot path,
     auto-route to cubecl only when on a non-NVIDIA device.

   Recommendation: **Option C**. Hand-tuned CUDA paths matter for
   training-throughput workloads; portability matters for everyone else.
2. **Move XPU's data residency to GPU memory** — currently every op
   roundtrips through host. Once cubecl's `Tensor::storage` supports
   device-resident allocation, switch.
3. **Expand cubecl op coverage** — softmax, layernorm/rmsnorm,
   transpose/permute, embed_lookup, attention. These are needed for any
   transformer model to run on cubecl-native devices (XPU / MPS / ROCm).
4. **Add `ferrotorch-mps`** — covered by #451, blocked on cubecl Metal
   maturity or a direct metal-rs integration.
5. **Add streams + events + allocator-stats to ferrotorch-xpu** if XPU
   becomes a tier-1 deploy target.
6. **Document the "tiered backend" model** explicitly:
   - Tier 1: NVIDIA via `ferrotorch-gpu` (cudarc + cuBLAS + cuDNN +
     hand kernels, full coverage)
   - Tier 2: Intel XPU via `ferrotorch-xpu` (cubecl wgpu, partial
     coverage)
   - Tier 3: AMD ROCm via `ferrotorch-cubecl` with `rocm` feature
     (cubecl rocm, partial coverage)
   - Tier 4: macOS via cubecl wgpu (not yet exposed as a crate)
   - Tier 5: WebGPU via cubecl wgpu (not yet exposed)

## Status

**ferrotorch-cubecl**: technology proof + GGUF dequant + DFA mask. ~10%
of full kernel coverage but a strong portability story.

**ferrotorch-xpu**: smoke-test layer. ~10% of `torch.xpu` surface.

**Do not split** — both crates have correct boundaries. The work to do is
fill in op coverage, not restructure.

## Related issues
- #451 — Add MPS backend for Apple Silicon GPU
