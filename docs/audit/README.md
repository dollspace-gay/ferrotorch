# ferrotorch ↔ pytorch crate-by-crate audit (#562)

Working comparison of every workspace crate against the corresponding pytorch
module. Each section documents what's present, what's missing, and what's a
deliberate divergence.

Reference: pytorch shallow clone at `~/pytorch` (HEAD as of 2026-05-01).

## Status

| # | Crate | pytorch counterpart | Status | Coverage |
|---|---|---|---|---|
| [1](01-ferrotorch-core.md) | ferrotorch-core | torch._C, torch.autograd, torch.linalg, torch.fft, torch.special, torch.func, torch.sparse, torch.nested, quantization primitives | done | tensor/autograd/vmap ≈ 90%, linalg ≈ 25%, fft ≈ 38%, special ≈ 22% |
| [2](02-ferrotorch-nn.md) | ferrotorch-nn (+ -nn-derive) | torch.nn | done | layers ≈ 95%, functional ≈ 25%, Module trait ≈ 60% |
| [3](03-ferrotorch-optim.md) | ferrotorch-optim | torch.optim, torch.amp.GradScaler | done | **100% + extras** (Muon, KFAC, EMA) |
| [4](04-ferrotorch-data.md) | ferrotorch-data | torch.utils.data | done | ≈ 85% |
| [5](05-ferrotorch-train.md) | ferrotorch-train | (no torch counterpart; ~Lightning) | done | Lightning subset |
| [6](06-ferrotorch-vision.md) | ferrotorch-vision | torchvision | done | ≈ 25% (popular models 70%, datasets 5%, ops 0%) |
| [7](07-ferrotorch-jit.md) | ferrotorch-jit | torch.jit + torch.fx + torch.compile + torch.export + AOTAutograd | done | structurally complete, depth varies |
| [8](08-ferrotorch-serialize.md) | ferrotorch-serialize | torch.save/load + safetensors + ONNX + GGUF | done | broad — covers more formats than torch core |
| [9](09-ferrotorch-gpu.md) | ferrotorch-gpu | torch.cuda | done | feature-complete for transformer training |
| [10](10-ferrotorch-cubecl-xpu.md) | ferrotorch-cubecl + -xpu | torch.xpu, torch.mps, ROCm | done | cubecl ≈ 10% kernels, xpu ≈ 10% surface |
| [11](11-ferrotorch-distributed.md) | ferrotorch-distributed | torch.distributed | done | major axes ✓, missing DTensor + p2p + elastic |
| [12](12-ferrotorch-distributions.md) | ferrotorch-distributions | torch.distributions | done | 71% by count, 100% of common |
| [13](13-ferrotorch-profiler.md) | ferrotorch-profiler | torch.profiler | done | core loop ✓, missing TB handler / NVTX / multi-backend |
| [14](14-ferrotorch-hub-tokenize.md) | ferrotorch-hub + -tokenize | torch.hub + HF tokenizers | done | hub ≈ 60%, tokenize covers basic path |
| [15](15-ferrotorch-llama.md) | ferrotorch-llama | (HF transformers reference) | done | model architecture ✓, generation API thin |
| [16](16-ferray-ferrolearn-integration.md) | ferray + ferrolearn integration | sister projects (numpy / sklearn equivalents) | done | 4 of 17 ferray crates wired; ferrolearn not integrated at all |
| 17 | ferrotorch-ml (Phase 3) | sklearn metrics / preprocessing / CV / decomposition / datasets via ferrolearn | done | new workspace crate; 9 of 17 ferray crates now wired (core, ufunc, linalg, random, fft, stride-tricks, window, ma); ferrolearn-{core,metrics,preprocess,model-sel,decomp,datasets} bridged via Tensor↔ndarray adapter |

## Top-level pytorch surface NOT mapped to any existing ferrotorch crate

After full audit, almost everything *does* map somewhere. Genuinely missing:

| pytorch module | ferrotorch home | Status |
|---|---|---|
| `torch.amp.GradScaler` | `ferrotorch-optim::grad_scaler` | ✅ found in audit |
| `torch.ao` (modern quant pipeline) | core quant primitives + nn.qat | partial — flow orchestration absent |
| `torch.onnx` | `ferrotorch-serialize::onnx_export` | ✅ found in audit |
| `torch.export` | `ferrotorch-jit::export` | ✅ found in audit |
| `torch._dynamo` / `torch._inductor` | `ferrotorch-jit` (codegen, trace_with_breaks, autotune) | ✅ structurally |
| `torch.fx` | `ferrotorch-jit::graph` + `interpreter` | ✅ structurally |
| `torch.func` (functorch) | `ferrotorch-core::vmap` + autograd | ≈ 80%; missing `functionalize`, `linearize`, `functional_call`, `stack_module_state` |
| `torch.signal` | core (could extend `special`) | gap |
| `torch.futures` | (Rust async, n/a) | n/a |
| `torch.monitor` | (gap) | low priority |
| `torch.package` | `ferrotorch-serialize::checkpoint` covers most | gap |
| `torch.multiprocessing` | (n/a in Rust) | n/a |
| `torch.mps` (Apple Silicon) | (no crate) — #451 open | gap |
| `torch.mtia` | (vendor-specific) | n/a |

## Cross-cutting findings

### Things present that I initially thought missing
- ONNX export (in `ferrotorch-serialize`)
- GradScaler (in `ferrotorch-optim`)
- vmap family (in `ferrotorch-core`)
- Forward-mode AD with DualTensor (in `ferrotorch-core::autograd`)
- Quantization primitives (in `ferrotorch-core::quantize`)
- Sparse + nested + pruning + flex_attention (all in `ferrotorch-core`)
- torch.export equivalent (in `ferrotorch-jit::export`)
- AOTAutograd (in `ferrotorch-jit::aot_autograd`)
- Inductor-style codegen (in `ferrotorch-jit::codegen_*`)
- Pickle parsing for `.pt` files (in `ferrotorch-serialize::pytorch_import`)
- GGUF format (in `ferrotorch-serialize::gguf`)
- Constrained decoding / DFA mask (in `ferrotorch-llama::grammar` + `ferrotorch-cubecl::grammar`)
- Paged attention (in `ferrotorch-nn::paged_attention` + `ferrotorch-paged` sibling)

### Genuine biggest gaps (priority-ordered)
1. **`ferrotorch-nn::functional`** — only 22 of ~91 torch fns. ~70 thin wrappers needed. Largest API surface gap.
2. **Module trait additions** — buffers, modules iteration, hook registration, `zero_grad`, `apply` (`ferrotorch-nn`)
3. **`ferrotorch-core::linalg`** — 8 of ~33 fns. Eigendecomp / lstsq / lu / etc.
4. **`ferrotorch-vision` datasets** — only 3 (MNIST, CIFAR10, CIFAR100). `ImageFolder` is the single biggest practical absence.
5. **`ferrotorch-vision::ops`** — empty. NMS, RoIAlign, focal loss for detection.
6. **`ferrotorch-distributed` DTensor + DeviceMesh** — modern parallelism abstraction.
7. **`ferrotorch-distributed` send/recv + all_to_all** — point-to-point + MoE collective.
8. **Element types in core** — no f16, no complex, no integer/bool tensors (`ferray-core` change first).
9. **`ferrotorch-cubecl` op coverage** — needs softmax, layernorm, attention to be a real cross-vendor backend.
10. **Generation API in `ferrotorch-llama`** — beam, sampling, streaming explicit and documented.

### Things ferrotorch has that torch lacks
1. **CubeCL portable kernels** — single-source GPU code targeting NVIDIA, AMD, Intel, Apple, WebGPU.
2. **GGUF interop + dequantize on GPU** — direct llama.cpp compatibility.
3. **Constrained decoding (DFA)** as a first-class GPU kernel.
4. **Paged attention** in nn (vLLM-style) as built-in primitive.
5. **2:4 structured sparsity + magnitude pruning** as first-class core API.
6. **Forward-mode AD with `DualTensor`** as a primary, not afterthought, API.
7. **Memory pressure / OOM watchdog** beyond `torch.cuda.memory_stats`.
8. **`SimulatedBackend`** for in-process distributed testing.
9. **Pipeline parallelism (GPipe + Interleaved1F1B)** at first launch.
10. **Native `Learner` trainer** (`ferrotorch-train`) — torch leaves this to Lightning.
11. **AOTAutograd, Inductor-style codegen, autotune** in a single crate (`ferrotorch-jit`).
12. **KFAC natural gradient + Muon optimizer** as first-class.

### Crate boundaries
**No crate should be split or merged based on this audit.** Each crate maps
1:1 to a coherent pytorch concept. Where pytorch has multiple sub-namespaces
collapsed into one ferrotorch crate (jit covering jit/fx/compile/export),
the internal module structure already provides the boundaries cleanly.

The two pure-additions worth considering as new crates:
- **`ferrotorch-quantize`** — if PT2E / AO-style quantization flow grows
  beyond what fits in `ferrotorch-core::quantize` + `ferrotorch-nn::qat`.
- **`ferrotorch-mps`** — Apple Silicon backend (#451), parallel to
  `ferrotorch-gpu` and `ferrotorch-xpu`.

## Recommendation summary

For each existing crate, the audit doc lists 5-15 specific actions. The
**top 10 across the whole workspace**, in dependency order:

1. Fill out `ferrotorch-nn::functional` to torch.nn.functional parity
   (~70 thin wrappers).
2. Add buffers + module traversal + hook registration to the `Module` trait.
3. Expand `ferrotorch-core::linalg` to ~25 missing functions.
4. Add `ImageFolder` + `DatasetFolder` + `ferrotorch-vision::ops` (NMS, RoIAlign).
5. Add `DTensor` + `DeviceMesh` + `send`/`recv` + `all_to_all` to `ferrotorch-distributed`.
6. Add `from_pretrained(repo_id)` + sharded download + auth token to `ferrotorch-hub` (#509).
7. Add chat templating to `ferrotorch-tokenize`.
8. Document and expand the generation API in `ferrotorch-llama`.
9. Add NTK / YaRN RoPE scaling (#515 + extension).
10. Decide on integer/bool/complex tensor types in `ferray-core` (cross-cutting).
