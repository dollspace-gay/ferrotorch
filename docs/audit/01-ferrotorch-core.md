# Audit: `ferrotorch-core` vs `torch` core

Covers: `torch._C` (tensor + dispatcher), `torch.autograd`, `torch.linalg`,
`torch.fft`, `torch.special`, `torch.func`, `torch.sparse`, `torch.nested`,
quantization primitives.

## Scale

| | ferrotorch-core | torch (reference) |
|---|---|---|
| LOC | ~14,364 across 25 files | C++ + Python, not directly comparable |
| Public functions | 562 | ~459 documented top-level `torch.*` ops + ~30 each in linalg/fft/special |
| Module files | tensor, dtype, device, storage, dispatch, gpu_dispatch, creation, methods, ops/, grad_fns/, autograd/, linalg, fft, special, sparse, nested, quantize, vmap, einops, einsum, flex_attention, pruning, shape, meta_propagate, profiler_hook, error, display, inplace, cpu_pool, ops_trait | torch._C, _tensor.py, autograd/, linalg/, fft/, special/, sparse/, nested/, masked/, _functorch/, _refs/, _prims/ |

## Coverage by area

### Element types

| ferrotorch-core | torch |
|---|---|
| f32, f64, bf16 (sealed via `ferray_core::Element`; `Float` trait gate) | f16, bf16, f32, f64, complex64, complex128, i8, i16, i32, i64, u8, u16, u32, u64, bool, qint8, quint8, qint32 |

**Gap:** no `f16` (IEEE half), no complex, no integer/bool tensor types.
This is a `ferray-core` change first.

### Devices

| ferrotorch-core | torch |
|---|---|
| Cpu, Cuda(idx), Xpu(idx), Meta | cpu, cuda, mps, xpu, hpu, mtia, meta, vulkan, ipu, ort, privateuseone, lazy |

**Gap:** MPS (#451 open), MTIA, HPU. Vulkan/IPU/lazy unlikely to be needed.

### Tensor methods (`methods.rs`, 54 fns)

Covered: add/sub/mul/div, neg, pow, sqrt, abs, exp, log, sin, cos, clamp,
relu, sigmoid, tanh, gelu (+with), silu, softmax, log_softmax, sum_all,
mean_all, prod_all, matmul, mm, mm_bt, bmm, mv, dot, t (transpose), einsum,
sum_dim, mean_dim, reshape, flatten, squeeze, unsqueeze, permute, transpose,
narrow, view, contiguous, chunk, split, size, dim.

**Gap:** dozens of method-form ops vs torch (see `_torch_docs.py` for the full
list — pytorch documents ~459 names, only ~50 are present as Tensor methods
here).

### Element-wise ops (`ops/elementwise.rs`, 28 fns)

Covered: SIMD f32/f64 (add, mul, exp, log, sqrt), fast_* fallbacks
(add/mul/sub/div/exp/log/sigmoid/tanh/sin/cos), unary_map / binary_map /
scalar_map combinators, sum / sum_axis / mean / nansum / nanmean / logsumexp /
logsumexp_dim.

### Indexing & scatter (`ops/indexing.rs`)

Covered: `gather`, `scatter`, `scatter_add`, `where_cond`.

**Gap:** `index_select`, `index_put`, `index_add`, `masked_fill`,
`masked_scatter`, `take`, `take_along_dim`, advanced indexing semantics on
the `Tensor` type itself.

### Search / sort / cumulative (`ops/search.rs`, `grad_fns/cumulative.rs`)

Covered: `bucketize`, `histc`, `meshgrid`, `searchsorted`, `topk`, `unique`,
`unique_consecutive`, `cummax`, `cummin`, `cumsum`, `cumprod`, `logcumsumexp`.

**Gap:** `sort`, `argsort`, `kthvalue`, `median`, `mode`, `quantile`,
`nanquantile`.

### Shape / manipulation (`grad_fns/shape.rs`, `ops/tensor_ops.rs`)

Covered: `cat`, `stack` (in vmap.rs), `expand`, `roll`, `tril`, `triu`,
`diag`, `diagflat`, `cdist`.

**Gap:** `dstack/hstack/vstack`, `block_diag`, `tile`, `repeat_interleave`,
`flip`, `fliplr`, `flipud`, `rot90`, `unfold`, `dsplit/hsplit/vsplit`,
`pad` (in `ferrotorch-nn::functional`), `as_strided`, `swapaxes/swapdims`.

### Activations (`grad_fns/activation.rs`)

Covered: `gelu`, `gelu_with` (+ `GeluApproximate`), `sigmoid`, `tanh`.
Plus method forms in `methods.rs`: `relu`, `silu`, `softmax`, `log_softmax`.

**Gap:** `leaky_relu`, `elu`, `selu`, `hardswish`, `hardtanh`, `mish`,
`softplus`, `softsign`, `prelu`, `glu`, `relu6`, `hardsigmoid`,
`celu`, `rrelu`, `softmin`, `softshrink`, `hardshrink`, `tanhshrink`,
`logsigmoid`.

### Autograd (`autograd/*`)

Covered:
- Reverse-mode: `backward`, `backward_with_grad`, `grad`, `grad_norm`,
  `gradient_penalty`, `vjp`, `jacobian`, `hessian`
- Forward-mode: `jvp`, `jvp_exact`, `jacfwd`, `DualTensor` + paired
  `dual_{add,mul,div,sub,neg,matmul,exp,log,sin,cos,tanh,relu,sigmoid}`
- Mode toggles: `no_grad`, `enable_grad`, `set_grad_enabled`,
  `is_grad_enabled`
- Anomaly: `AnomalyMode`, `ForwardBacktrace`, `check_gradient_anomaly`,
  `detect_anomaly`
- Higher-order utilities: `cond`, `scan`, `fixed_point`,
  `validate_cond_branches`
- Hooks: `HookHandle`
- vmap family: `vmap`, `vmap2`, `vmap3`, `vmap_many`, `vmap_multi_output`,
  `per_sample_grad`
- Saved tensors, gradcheck, checkpoint

**Gap:**
- `functionalize` (functional/no-side-effect mode for transforms)
- `linearize` (cached jvp)
- `functional_call` — call a `Module` with externally provided params (belongs
  in `ferrotorch-nn`; needs `Module` integration)
- `stack_module_state` — stack params of N copies of a `Module` for vmap
  ensembles (belongs in `ferrotorch-nn`)
- `inference_mode` (stricter than `no_grad`)
- `set_multithreading_enabled`
- `enforce_grad_layout_policy`

### Mixed precision

Covered: `AutocastCategory`, `AutocastDtype`, `autocast`, `autocast_dtype`,
`autocast_guard`, `set_autocast_debug`, `is_autocast_enabled`,
`autocast_ops` (per-op casting policy).

**Gap (high-impact):** **no `GradScaler`** for fp16/bf16 loss scaling. This
blocks practical mixed-precision training where loss values fall below the
fp16 representable range (~6e-5).

### linalg (`linalg.rs`, 8 fns)

Covered: `svd`, `solve`, `det`, `inv`, `qr`, `cholesky`, `matrix_norm`,
`pinv`.

**Gap (~25 fns):** `eig`, `eigvals`, `eigh`, `eigvalsh`, `slogdet`, `lstsq`,
`lu`, `lu_factor`, `lu_factor_ex`, `lu_solve`, `vector_norm`, `multi_dot`,
`solve_triangular`, `matrix_power`, `matrix_rank`, `matrix_exp`, `cross`,
`diagonal`, `householder_product`, `ldl_factor`, `ldl_factor_ex`,
`ldl_solve`, `svdvals`, `cond`, `tensorinv`, `tensorsolve`, `cholesky_ex`,
`inv_ex`, `solve_ex`.

This is the **single largest API-surface gap inside core**.

### fft (`fft.rs`, 6 fns)

Covered: `fft`, `ifft`, `fft2`, `ifft2`, `rfft`, `irfft` (1-D + 2-D, real and
complex).

**Gap:** N-d (`fftn`, `ifftn`, `rfftn`, `irfftn`), Hermitian (`hfft`, `ihfft`,
`hfftn`, `ihfftn`), frequency helpers (`fftfreq`, `rfftfreq`), shifts
(`fftshift`, `ifftshift`).

### special (`special.rs`, 9 fns)

Covered: `erf`, `erfc`, `erfinv`, `lgamma`, `digamma`, `log1p`, `expm1`,
`sinc`, `xlogy`.

**Gap:** `entr`, `psi`, `polygamma`, `erfcx`, `logit`, `expit`, `ndtr`,
`ndtri`, `log_ndtr`, `zeta`, `multigammaln`, `gammainc`, `gammaincc`,
`logsumexp`, `softmax`, `log_softmax`, `round` (special variant); the long
tail: `airy_ai`, `chebyshev_polynomial_{t,u,v,w}`, `hermite_polynomial_{h,he}`,
`laguerre_polynomial_l`, `legendre_polynomial_p`,
`modified_bessel_{i0,i1,k0,k1}`, `bessel_{j0,j1,y0,y1}`,
`scaled_modified_bessel_{k0,k1}`, `shifted_chebyshev_polynomial_*`,
`spherical_bessel_j0`.

### sparse (`sparse.rs`, `pruning.rs`)

Covered: `CooTensor`, `CsrTensor`, `SparseTensor`, `SemiStructuredSparseTensor`,
`sparse_matmul_24`, `apply_2_4_mask`, `magnitude_prune`, `sparsity_ratio`.

**Gap:** sparse autograd integration; sparse equivalents of softmax, addmm,
mm, sum, log_softmax; CSC and BSR/BSC formats; coalescing semantics; sparse
tensor construction (`torch.sparse_coo_tensor`, `torch.sparse_csr_tensor`);
to_sparse / to_dense methods on `Tensor`.

### nested (`nested.rs`)

Covered: `NestedTensor`, `PackedNestedTensor`,
`nested_scaled_dot_product_attention`.

**Gap:** general op dispatch on nested tensors (only SDPA today); construction
helpers (`torch.nested.nested_tensor`, `as_nested_tensor`, `narrow`, `unbind`).

### quantization (`quantize.rs`)

Covered: `FakeQuantize`, `MinMaxObserver`, `HistogramObserver`,
`PerChannelMinMaxObserver`, `Observer`, `QParams`, `QatLayer`, `QatModel`,
`QuantDtype`, `QuantScheme`, `QuantizedTensor`, `cuda_rng`, `dequantize`,
`prepare_qat`, `quantize`, `quantize_named_tensors`, `quantized_matmul`,
`fake_quantize_differentiable`.

**Gap:** PT2E quantization flow, FX-mode quant flow, weight-only / activation-
only schemes, BitsAndBytes-style 4-bit/8-bit linear layers, AWQ/GPTQ
calibrators, dynamic quantization passes. Primitives are present; flow
orchestration is not.

### Other present-and-good

- **einops**: `rearrange`, `rearrange_with`, `reduce`, `repeat` — native, beyond `torch.einsum`
- **einsum**: `einsum`, `einsum_differentiable`
- **flex_attention**: `flex_attention`
- **Dispatcher**: `DispatchKey`, `DispatchKeySet`, `Dispatcher`, `Kernel`, `gpu_dispatch` — structurally aligned with torch's c10 dispatcher
- **MemoryFormat**: enum on `Tensor` (mirrors `torch.contiguous_format`, `channels_last`)
- **Storage**: `StorageBuffer`, `TensorStorage`
- **Meta device**: shape-only tensors with `meta_propagate` for symbolic shape inference

### Strengths over upstream

1. Forward-mode AD is first-class (`DualTensor` + paired ops); torch's `forward_ad` API is manual/awkward
2. vmap family is broader (incl. `per_sample_grad`)
3. einops native
4. flex_attention integrated
5. 2:4 structured sparsity + pruning helpers as first-class API
6. Meta device + propagate works for any op via `meta_propagate.rs`

## Concrete recommendations

1. **Expand `linalg.rs`** to cover the missing 25 functions (highest impact).
2. **Expand `fft.rs`** with N-d, Hermitian, freq helpers, shifts.
3. **Expand `special.rs`** for the common functions (`logit`, `expit`,
   `polygamma`, `entr`, `gammainc/cc`, `logsumexp`, `ndtr/i/log_ndtr`); defer
   the Bessel/orthogonal-polynomial families behind a feature flag.
4. **Add `grad_scaler.rs`** next to `autograd/autocast.rs` (blocks fp16
   training).
5. **Add the activation tail** in `grad_fns/activation.rs` (leaky_relu, elu,
   selu, hardswish, mish, softplus, prelu, etc.).
6. **Add the indexing/shape tail** (`flip`, `fliplr/flipud`, `tile`,
   `repeat_interleave`, `as_strided`, `swapaxes`, `unfold`, `index_select`,
   `index_put`, `index_add`, `masked_fill`).
7. **Add `sort`, `argsort`, `kthvalue`, `median`, `quantile`** in
   `ops/search.rs`.
8. **`functional_call` + `stack_module_state`** belong in `ferrotorch-nn`
   (depend on `Module`).
9. **Element types**: integer/bool/complex tensors are a `ferray-core`
   project — track separately, not core's job to fix unilaterally.

**Do not split ferrotorch-core.** Its scope matches what torch's `_C` +
`_tensor.py` + `linalg/fft/special` thin sub-namespaces collapse into. The
gaps are op additions, not new crate boundaries.
