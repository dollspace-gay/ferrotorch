# Audit: `ferrotorch-gpu` vs `torch.cuda`

CUDA-specific GPU backend. Built on `cudarc`, parallels what `torch.cuda` +
ATen CUDA kernels provide.

## Modules

| Module | Role |
|---|---|
| `device` | `GpuDevice` — device handle, init, properties |
| `backend_impl` | `CudaBackendImpl`, `init_cuda_backend`, `get_cuda_device` — backend registration with core dispatcher |
| `tensor_bridge` | `GpuTensor`, `GpuFloat`, `cuda`, `cuda_default`, `tensor_to_cpu`, `tensor_to_gpu` |
| `transfer` | `cpu_to_gpu`, `gpu_to_cpu`, `alloc_zeros{,_f32,_f64}` |
| `buffer` | `CudaBuffer` |
| `allocator` | `CudaAllocator` (caching allocator) |
| `pool` | `cached_bytes`, `empty_cache`, `empty_cache_all`, `round_len` |
| `memory_guard` | `MemoryGuard` family — pressure listening, OOM policy, watchdog, reservations, hooks, stats |
| `module_cache` | Cached PTX module loading |
| `stream` | CUDA streams |
| `rng` | `CudaRngManager`, `PhiloxGenerator`, `PhiloxState`, `fork_rng`, `join_rng` |
| `graph` | CUDA graph capture: `CapturedGraph`, `CaptureMode`, `CapturePool`, `make_graphed_callable` |
| `kernels` | Hand-rolled CUDA kernels (add, mul, sub, neg, relu, gelu, layernorm, softmax, embed_lookup, transpose, permute_0213, dropout, broadcast variants, slice ops, small_matmul, small_bmm, causal_mask) |
| `bf16` | bf16-specific kernels (rmsnorm, rope_half, repeat_kv, softmax, silu, transpose-to/from-heads, scale, mul, add, embedding_gather, fatrelu, relu, causal_mask, block_reduce_max_abs) |
| `blas` | cuBLAS: `gpu_matmul_f32/f64`, `gpu_bmm_f32`, `gpu_matmul_bf16_bf16` (+ `_nt` non-transpose, strided-batched variants), `_into` variants |
| `conv` | cuDNN-style conv: `gpu_conv2d_f32` |
| `cusolver` | cuSOLVER linalg ops |
| `flash_attention` | `gpu_flash_attention_f32` |

## Coverage vs `torch.cuda`

torch.cuda has 67 top-level functions in `__init__.py` + memory.py + streams +
graphs + nccl + amp + nvtx + tunable + comm + jiterator + sparse + green_contexts + sanitizer.

| Capability | ferrotorch-gpu | torch.cuda | Status |
|---|---|---|---|
| Device init / `is_available()` | `GpuDevice::new()`, `init_cuda_backend()` | `is_available`, `init`, `is_initialized` | ✅ |
| `device_count()` | (via cudarc) | `device_count` | ✅ |
| `current_device()` / `set_device()` | (via `GpuDevice`) | yes | ✅ |
| `get_device_name/properties/capability` | partial (depends on cudarc) | yes | likely ✅ |
| Streams (`Stream`, `current_stream`, `default_stream`, `set_stream`, `synchronize`) | `stream` module | yes | ✅ |
| Events (`Event`, `record`, `wait`, `elapsed_time`) | not visible in lib.rs re-exports | `Event` | gap |
| RNG (`manual_seed`, `seed`, `get_rng_state`, `set_rng_state`, Philox) | `CudaRngManager`, `PhiloxGenerator`, `PhiloxState`, `fork_rng`, `join_rng` | ✅ + Philox | ✅ |
| CUDA graphs (`graph()`, `make_graphed_callables`, capture begin/end) | `begin_capture`, `end_capture`, `make_graphed_callable`, `CaptureMode`, `CapturePool`, `GraphCaptureGuard` | ✅ | ✅ |
| Caching allocator | `CudaAllocator`, `pool::cached_bytes`, `empty_cache`, `empty_cache_all` | `caching_allocator_*`, `empty_cache` | ✅ |
| Memory stats | `MemoryStats` (in memory_guard) | `memory_stats`, `memory_allocated`, `max_memory_allocated`, peak/accumulated reset | ✅ partial — verify all stats names |
| Per-process memory fraction | not visible | `set_per_process_memory_fraction` | gap |
| OOM handling | `OomPolicy`, `MemoryWatchdog`, `MemoryPressureListener`, `MemoryGuard` | torch raises OutOfMemoryError | **superset** |
| NCCL | none | `torch.cuda.nccl` | gap (covered partly by `ferrotorch-distributed`?) |
| AMP context (`torch.amp.autocast`) | core has it | core has it | (n/a here) |
| Profiling integration | none | `torch.cuda.profiler.start/stop` | gap |
| NVTX markers | none | `torch.cuda.nvtx.range_push/pop/range_start/end` | gap (worth adding for Nsight) |
| Sanitizer | none | `torch.cuda._sanitizer` (compute sanitizer integration) | gap |
| Tunable ops (autotune cuBLAS/cuDNN algos) | partial (`autotune` is in jit, not gpu) | `torch.cuda.tunable` | gap structurally (different home) |
| GDS (GPU Direct Storage) | none | `torch.cuda.gds` | gap |
| Comm (peer-to-peer) | none directly | `torch.cuda.comm` (broadcast, gather, scatter for multi-GPU) | gap (in `ferrotorch-distributed`?) |
| Green contexts (carve up GPU resources) | none | `torch.cuda.green_contexts` | gap (niche) |

## Strengths

- **Hand-rolled bf16 kernel set** — torch leans on cuDNN/cuBLAS; ferrotorch
  has bf16-native kernels for the common transformer ops (rmsnorm,
  rope_half, repeat_kv, embedding_gather, etc.) which is exactly what
  Llama-class models need.
- **Memory pressure / watchdog API** is *richer* than torch's. `MemoryGuard`,
  `OomPolicy`, `PressureLevel`, `MemoryReservation`, `MemoryHook` go
  beyond `torch.cuda.memory_stats`.
- **flash_attention** integrated as a kernel (`gpu_flash_attention_f32`).
- **CUDA graph capture** with multi-stream pools (`CapturePool`,
  `GraphPoolHandle`).
- **Philox RNG** — matches torch's CUDA RNG semantics for reproducibility.
- **`_into` variants** of every kernel for in-place / fused output buffers.

## Gaps

1. **`Event` API** — not visible in re-exports. Required for fine-grained
   timing and stream synchronization.
2. **`set_per_process_memory_fraction`** — useful in shared-GPU
   environments.
3. **NVTX integration** — adding `nvtx_push("name")` / `nvtx_pop()` would
   help with Nsight profiling. Independent of `ferrotorch-profiler`.
4. **`torch.cuda.tunable`** — autotuning cuBLAS/cuDNN heuristic algorithms
   per shape. Today autotune lives in `ferrotorch-jit::autotune` for
   compiled paths but not for eager BLAS calls.
5. **NCCL bindings** — should live in `ferrotorch-distributed`, not here,
   but a clear cross-reference is needed.
6. **Comm primitives** (peer-to-peer broadcast/gather/scatter that aren't
   collective) — also distributed crate's job.
7. **GPU profiler hooks** — `cuProfilerStart/Stop` for cupti / nsys
   integration.

## Documentation crate description

Crate description says "CUDA GPU backend for ferrotorch" — accurate, scope
is clear.

## Status

ferrotorch-gpu is **core-feature complete vs torch.cuda** for the things
that matter for training transformer models. Gaps (Events, NVTX, tunable,
per-process fraction) are quality-of-life rather than blocking.

**Do not split.** Crate matches torch.cuda's purpose 1:1.

**Verify**: spot-check that `Event`/`stream::Event` exists in
`stream.rs` even if not re-exported, and re-export it if so.
