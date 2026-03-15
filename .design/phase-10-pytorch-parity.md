# Feature: Phase 10 — Full PyTorch Feature Parity

## Summary
Close every remaining feature gap between ferrotorch and PyTorch. This phase targets the features researchers and production engineers actually depend on daily — einsum, forward/backward hooks, probability distributions, higher-order gradients, FlashAttention, profiling, sparse tensors, FFT, full linalg, functional transforms (vmap), and ecosystem integration (TensorBoard, model hub, GGUF). After this phase, there is no technical reason to choose PyTorch over ferrotorch.

## Requirements

### Tier 1 — Researchers use these daily

- REQ-1: **Einstein summation (`einsum`)** must accept a string equation and a list of tensors, supporting all valid subscript patterns: matrix multiply (`"ij,jk->ik"`), batch matmul (`"bij,bjk->bik"`), trace (`"ii->"`), outer product (`"i,j->ij"`), contraction (`"ijk,ikl->ijl"`), and implicit mode (output indices inferred). The implementation must parse the equation at runtime, compute the output shape, and execute via a combination of transpose, reshape, and matmul. Must be differentiable — `einsum` output must have a `GradFn` that propagates gradients to all inputs. Must support 2+ input tensors. Target: ferrotorch-core.

- REQ-2: **Forward and backward hooks** must be registrable on any `Module<T>`. `register_forward_hook(callback)` fires after `forward()` with `(module, input, output)` and can modify the output. `register_backward_hook(callback)` fires during backward with `(module, grad_input, grad_output)` and can modify `grad_input`. `register_forward_pre_hook(callback)` fires before forward and can modify input. Hooks must return a `HookHandle` that can be removed. Multiple hooks on the same module fire in registration order. Target: ferrotorch-nn Module trait.

- REQ-3: **Probability distributions** must provide a `Distribution<T>` trait with `sample()`, `log_prob(value)`, `entropy()`, and `rsample()` (reparameterized sample for gradient flow). Implementations: `Normal`, `Bernoulli`, `Categorical`, `Uniform`, `Beta`, `Gamma`, `Dirichlet`, `MultivariateNormal`, `Poisson`, `Exponential`. `rsample()` must produce tensors with `requires_grad=true` using the reparameterization trick (sample = mu + sigma * epsilon where epsilon ~ N(0,1)). Target: new `ferrotorch-distributions` crate.

- REQ-4: **Higher-order gradients** must allow computing gradients of gradients: `grad(grad(f(x), x), x)` for Hessians, Jacobians, and meta-learning (MAML). The autograd engine must support `create_graph=true` during backward, which retains the computation graph through the backward pass itself so a second backward can be called. `torch.autograd.grad(outputs, inputs, create_graph=true)` equivalent. Target: ferrotorch-core autograd.

- REQ-5: **FlashAttention** must implement memory-efficient scaled dot-product attention that avoids materializing the full `[B, H, N, N]` attention matrix. For sequence length N, standard attention uses O(N^2) memory; FlashAttention uses O(N) by computing attention in tiles that fit in SRAM. Must support causal masking. CPU implementation uses tiled algorithm; GPU implementation uses custom PTX/CubeCL kernel. Must be a drop-in replacement for `scaled_dot_product_attention`. Target: ferrotorch-nn attention module + ferrotorch-gpu kernel.

- REQ-6: **Profiler** must instrument tensor operations, CUDA kernel launches, and memory allocations with wall-clock timing. Must support three output formats: human-readable table (sorted by total time), Chrome trace JSON (loadable in chrome://tracing), and TensorBoard trace. Must capture: op name, input shapes, duration (CPU + GPU), memory allocated/freed. Must be usable as a context manager: `with_profiler(config, |prof| { ... })`. Target: new `ferrotorch-profiler` crate.

### Tier 2 — Power users need these

- REQ-7: **Sparse tensor support** must provide `SparseTensor<T>` in COO format (coordinate list: `indices: Tensor<i64>`, `values: Tensor<T>`, `shape: Vec<usize>`) with sparse-dense matmul (`sparse @ dense -> dense`), element-wise ops on values, `to_dense()`, and `from_dense(threshold)`. Sparse gradients for `Embedding` backward must use this. Target: ferrotorch-core.

- REQ-8: **FFT operations** must provide `fft(input, n, dim)`, `ifft`, `rfft`, `irfft`, `fft2d`, `ifft2d` by delegating to ferray-fft (rustfft 6.4 backend). Must support f32 and f64. Must be differentiable (FFT backward is IFFT scaled). Target: ferrotorch-core ops.

- REQ-9: **Full linear algebra** must provide `svd(input)`, `eig(input)`, `solve(A, b)`, `cholesky(input)`, `det(input)`, `inv(input)`, `qr(input)`, `matrix_rank(input)`, `pinv(input)`, `norm(input, ord)` by delegating to ferray-linalg (faer backend). Each must be differentiable where mathematically defined. Target: ferrotorch-core ops + grad_fns.

- REQ-10: **Functional transforms (`vmap`)** must provide `vmap(fn, in_dims, out_dims)` that vectorizes a function over a batch dimension without explicit loops. MVP implementation loops internally (correct but not vectorized); true batched dispatch is an optimization. Target: ferrotorch-core.

- REQ-11: **RNN utilities** must provide `pack_padded_sequence(input, lengths, batch_first, enforce_sorted)` and `pad_packed_sequence(packed, batch_first)` for variable-length sequence handling in LSTM/GRU. Target: ferrotorch-nn.

- REQ-12: **Autocast wired to operations** — the existing `autocast()` context must actually cause eligible operations to run in reduced precision. Policy: matmul, conv, linear in f16/bf16; reductions, norms, softmax in f32. Target: ferrotorch-core grad_fns.

### Tier 3 — Ecosystem and production

- REQ-13: **TensorBoard integration** must write scalar summaries, histograms, images, and computation graphs to TensorBoard event files (protobuf format). Users add `TensorBoardLogger` as a training callback. Must work without Python. Target: `ferrotorch-train` callback or new crate.

- REQ-14: **Model hub** must provide `hub::load("resnet50", pretrained=true)` that downloads SafeTensors weights from a configurable CDN, caches them at `~/.ferrotorch/hub/`, and loads into the model. Target: new `ferrotorch-hub` crate.

- REQ-15: **GGUF/GGML format** for LLM inference — parse the GGUF header, extract quantized weight tensors (Q4_0, Q4_1, Q8_0, Q8_1), and dequantize to ferrotorch tensors. Target: ferrotorch-serialize.

- REQ-16: **PagedAttention** for efficient LLM serving — manage KV cache as fixed-size pages that can be allocated, freed, and shared across sequences. Eliminates memory fragmentation for concurrent inference. Target: ferrotorch-nn.

- REQ-17: **`torch.special` equivalent** — special math functions: `erf`, `erfc`, `erfinv`, `gamma`, `lgamma`, `digamma`, `bessel_j0/j1`, `sinc`, `xlogy`, `log1p`, `expm1`. Target: ferrotorch-core.

- REQ-18: **Gradient penalty utilities** — `grad(outputs, inputs, create_graph)` and `jacobian(fn, inputs)` convenience functions for WGAN-GP, physics-informed NNs, etc. Depends on REQ-4. Target: ferrotorch-core autograd.

## Acceptance Criteria

### Tier 1

- [ ] AC-1: `einsum("ij,jk->ik", &[a, b])` produces the same result as `mm(a, b)`. `einsum("bij,bjk->bik", &[a, b])` matches `bmm`. Backward produces correct gradients verified against PyTorch for at least 10 equation patterns.
- [ ] AC-2: A forward hook registered on `Linear` captures output shape. A backward hook captures `grad_output`. Removing via `HookHandle::remove()` prevents subsequent calls. Multiple hooks fire in order.
- [ ] AC-3: `Normal::new(0.0, 1.0).rsample([1000])` returns a tensor with `requires_grad=true`. `log_prob(0.0)` returns `-0.5 * ln(2*pi)`. `Categorical::new(probs).sample()` returns indices in `0..num_classes`.
- [ ] AC-4: Given `f(x) = x^3`, `grad(f(x), x)` returns `3*x^2`, and `grad(grad(f(x), x), x)` returns `6*x`. MAML inner loop with `create_graph=true` allows outer-loop gradients.
- [ ] AC-5: FlashAttention on `[1, 8, 2048, 64]` Q/K/V matches standard attention within `rtol=1e-3`. Peak memory at most 2x input size. Causal masking correct.
- [ ] AC-6: Profiler captures op name, duration, and input shapes. Chrome trace JSON loads in chrome://tracing. Top-10 table shows cumulative time.

### Tier 2

- [ ] AC-7: Sparse-dense matmul matches dense matmul. `to_dense()` produces correct matrix.
- [ ] AC-8: `fft` matches `np.fft.fft` within `rtol=1e-5`. `rfft` + `irfft` round-trips. FFT backward correct.
- [ ] AC-9: `svd(A)` reconstructs A. `solve(A, b)` solves Ax=b. `det(A)` matches ferray-linalg. Each has backward.
- [ ] AC-10: `vmap(|x| x.matmul(&w), 0)(&batched_x)` matches batched matmul.
- [ ] AC-11: `pack_padded_sequence` with lengths `[5, 3, 2]` produces `batch_sizes = [3, 3, 2, 1, 1]`.
- [ ] AC-12: Inside `autocast(F16)`, `Linear::forward` on f32 inputs casts to f16 for matmul, returns f32.

### Tier 3

- [ ] AC-13: `TensorBoardLogger` writes valid TFEvents file loadable by `tensorboard --logdir=`.
- [ ] AC-14: `hub::load("resnet50", true)` downloads, caches, and loads weights correctly.
- [ ] AC-15: `load_gguf("model.gguf")` parses Q4_0 tensors and dequantizes matching llama.cpp.
- [ ] AC-16: PagedAttention serves 8 concurrent sequences without memory fragmentation.
- [ ] AC-17: `erf(tensor)` matches scipy.special.erf within `rtol=1e-6`. All special functions differentiable.
- [ ] AC-18: `grad(loss, params, create_graph=true)` + `grad(grad_norm, params)` computes WGAN-GP gradient penalty.

## Architecture

### New Crates

| Crate | Purpose |
|-------|---------|
| `ferrotorch-distributions` | Probability distributions (Normal, Bernoulli, Categorical, etc.) |
| `ferrotorch-profiler` | Operation profiling with Chrome trace and TensorBoard export |
| `ferrotorch-hub` | Model download, caching, and registry |

### Modifications to Existing Crates

| Crate | Additions |
|-------|----------|
| `ferrotorch-core` | einsum, sparse tensors, FFT ops, full linalg, vmap, higher-order gradients, special functions, autocast wiring |
| `ferrotorch-nn` | Forward/backward hooks, pack_padded_sequence, FlashAttention, PagedAttention |
| `ferrotorch-gpu` | FlashAttention CUDA kernel, sparse matmul kernel |
| `ferrotorch-serialize` | GGUF/GGML parser |
| `ferrotorch-train` | TensorBoard callback |

### Parallelism Plan

**Wave 1 (all independent — 6 agents):**
- einsum, hooks, distributions, profiler, sparse tensors, FFT

**Wave 2 (some deps — 5 agents):**
- full linalg, vmap, pack_padded_seq, special functions, TensorBoard

**Wave 3 (deps on waves 1-2 — 4 agents):**
- higher-order gradients, FlashAttention CPU, autocast wiring, model hub

**Wave 4 (deps on wave 3):**
- FlashAttention GPU, gradient penalty, PagedAttention, GGUF parser

### Key Design Decisions

**einsum**: Parse equation into contraction plan. 2-input: TTGT algorithm (transpose-transpose-gemm-transpose). N-input: greedy pairwise. Backward: einsum with rearranged indices.

**Higher-order gradients**: `create_graph=true` records backward ops in a new graph. `GradFn::backward()` produces tensors with `grad_fn` set. Hardest feature — touches core autograd engine.

**FlashAttention (CPU)**: Tile Q/K/V into blocks, use online softmax trick. Never materialize N×N matrix. Backward via recomputation.

**FlashAttention (GPU)**: CubeCL or PTX kernel with shared memory tiling, warp-level softmax reductions.

**Sparse tensors**: COO format (simplest). CSR deferred. Sparse-dense matmul via scatter-accumulate.

**vmap (MVP)**: Loop over batch dim internally. True batched dispatch deferred.

**Distributions**: Follow PyTorch API exactly. `rsample()` uses reparameterization trick for gradient flow.

## Out of Scope
- Python bindings (PyO3) — separate phase
- Graph neural network layers (GCN, GAT) — community contribution
- Audio/text domain-specific features — community contribution
- WebAssembly target — defer to after core stabilizes
- Quantization-aware training (QAT) — defer to after PTQ proven
