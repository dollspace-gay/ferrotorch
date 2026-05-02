# Audit: `ferrotorch-nn` (+ `ferrotorch-nn-derive`) vs `torch.nn`

Covers: `torch.nn.Module`, `torch.nn.Parameter`, `torch.nn.modules.*` (layers,
loss, container, conv, rnn, transformer, etc.), `torch.nn.functional`,
`torch.nn.init`, `torch.nn.utils`, `torch.nn.attention`.

## Scale

| | ferrotorch-nn | torch.nn |
|---|---|---|
| Module files | 33 | 26 (modules/) + functional + init + utils + attention |
| Public types (re-exports) | ~95 | ~150 |

## Coverage by area

### Module trait

| ferrotorch | torch.nn.Module |
|---|---|
| `forward(&Tensor) -> Result<Tensor>` | `forward(*args, **kwargs)` |
| `parameters() -> Vec<&Parameter>` | `parameters() -> Iterator[Parameter]` |
| `parameters_mut() -> Vec<&mut Parameter>` | (mutability via Python identity) |
| `named_parameters() -> Vec<(String, &Parameter)>` | `named_parameters()` |
| `train()` / `eval()` / `is_training()` | `train(mode=True)` / `eval()` / `training` |
| `to_device(Device)` | `to(device, dtype, ...)` |
| `state_dict()` / `load_state_dict()` | same + `assign=False`, hooks |

**Missing on Module trait:**
- `buffers()` / `named_buffers()` — no buffer concept (running statistics in
  BatchNorm, position tables, etc., must currently be stored as parameters or
  hand-rolled)
- `modules()` / `named_modules()` — submodule traversal
- `children()` / `named_children()` — direct children
- `register_parameter`, `register_buffer`, `register_module` — late binding
- `register_forward_hook`, `register_forward_pre_hook`,
  `register_backward_hook`, `register_full_backward_hook`,
  `register_state_dict_pre_hook`, `register_state_dict_post_hook`
- `apply(fn)` — recursive function over modules
- `zero_grad()` — currently must zero grads via optimizer
- `requires_grad_(bool)` — set on whole subtree
- `get_parameter`, `get_buffer`, `get_submodule` — string-path lookup
- `extra_repr()` — pretty-printing
- dtype casts (`half()`, `float()`, `double()`, `bfloat16()`) — partial
  coverage via generic `T`
- `share_memory()` — not applicable (no fork-based parallelism in Rust)

**Notes:** Parameters get most coverage via `Parameter::to(device)`,
`zeros/ones/from_slice` constructors, `Deref<Target=Tensor>`. Hook surface
is partially in `hooks.rs` (`HookHandle`, `HookedModule`, `ForwardHook`,
`BackwardHook`, `ForwardPreHook`) but not yet wired into the trait API.

### Activations

ferrotorch (28 types) ↔ torch (28 types):
CELU, ELU, GELU, GLU, HardSigmoid, HardSwish, Hardshrink, Hardtanh, LeakyReLU,
LogSigmoid, LogSoftmax, Mish, PReLU, RReLU, ReLU, ReLU6, SELU, SiLU, Sigmoid,
Softmax, Softmin, Softmax2d, Softplus, Softshrink, Softsign, Tanh, Tanhshrink,
Threshold.

**Coverage: 100%.** `MultiheadAttention` lives in `attention.rs` instead of
`activation.rs` — minor structural difference.

### Loss functions

ferrotorch (18) — `BCELoss`, `BCEWithLogitsLoss`, `CTCLoss`,
`CosineEmbeddingLoss`, `CrossEntropyLoss`, `GaussianNLLLoss`,
`HingeEmbeddingLoss`, `HuberLoss`, `KLDivLoss`, `L1Loss`, `MSELoss`,
`MarginRankingLoss`, `MultiLabelSoftMarginLoss`, `MultiMarginLoss`, `NLLLoss`,
`PoissonNLLLoss`, `SmoothL1Loss`, `TripletMarginLoss`.

torch (22). **Missing 4:** `NLLLoss2d` (deprecated wrapper),
`MultiLabelMarginLoss`, `SoftMarginLoss`, `LinearCrossEntropyLoss`,
`TripletMarginWithDistanceLoss`.

### Conv

ferrotorch (9): `Conv1d/2d/3d`, `ConvTranspose1d/2d/3d`,
`LazyConv1d/2d/3d`.

torch (12). **Missing:** `LazyConvTranspose1d/2d/3d`.

### Pooling

ferrotorch (15): `MaxPool1d/2d/3d`, `MaxUnpool2d`, `AvgPool1d/2d/3d`,
`AdaptiveMaxPool1d/2d/3d`, `AdaptiveAvgPool1d/2d/3d`, `FractionalMaxPool2d`,
`LPPool1d/2d`.

torch (18). **Missing:** `MaxUnpool1d/3d`, `FractionalMaxPool3d`, `LPPool3d`.

### Normalization

ferrotorch (10): `BatchNorm1d/2d/3d`, `GroupNorm`, `InstanceNorm1d/2d/3d`,
`LayerNorm`, `LocalResponseNorm`, `RMSNorm`.

torch (13 + lazy). **Missing:** `SyncBatchNorm` (distributed),
`LazyBatchNorm1d/2d/3d`, `LazyInstanceNorm1d/2d/3d`, `CrossMapLRN2d`.
`SyncBatchNorm` matters once distributed training is real.

### Padding

ferrotorch (15) ↔ torch (15): `CircularPad1d/2d/3d`, `ConstantPad1d/2d/3d`,
`ReflectionPad1d/2d/3d`, `ReplicationPad1d/2d/3d`, `ZeroPad1d/2d/3d`.

**Coverage: 100%.**

### Embedding

ferrotorch (2): `Embedding`, `EmbeddingBag`. ✅

### Linear

ferrotorch (2): `Linear`, `LazyLinear`.

torch (4). **Missing:** `Bilinear`. (`Identity` is in `identity.rs`.)

### Container

ferrotorch (5) ↔ torch (5): `Sequential`, `ModuleList`, `ModuleDict`,
`ParameterList`, `ParameterDict`. ✅

### Identity / shape / distance

ferrotorch: `Identity`, `Flatten`, `Unflatten`, `ChannelShuffle`,
`CosineSimilarity`, `PairwiseDistance`. ✅ matches torch.

### Upsample / fold / pixel shuffle

ferrotorch: `Upsample`, `Fold`, `Unfold`, `PixelShuffle`, `PixelUnshuffle`,
`GridSample`, `affine_grid`, `interpolate`, `grid_sample` (functional),
`fold` / `unfold` (functional), `pixel_shuffle` / `pixel_unshuffle`
(functional).

torch additionally has `UpsamplingNearest2d`, `UpsamplingBilinear2d` (both
deprecated thin wrappers around `Upsample` — not worth porting).

### RNN

ferrotorch (8): `RNN`, `LSTM`, `GRU`, `RNNCell`, `LSTMCell`, `GRUCell`,
`RNNNonlinearity`, plus `rnn_utils::{PackedSequence, pack_padded_sequence,
pad_packed_sequence}`. ✅ matches torch.

### Transformer

ferrotorch: `Transformer`, `TransformerEncoder`, `TransformerDecoder`,
`TransformerEncoderLayer`, `TransformerDecoderLayer`, plus extras: `KVCache`,
`RoPEConvention`, `RoPEScaling`, `RotaryPositionEmbedding`, `SwiGLU`.

torch matches the 5 base classes; the RoPE / SwiGLU / KVCache extras are
ferrotorch-native (torch puts equivalents in `torch.nn.attention` /
`torch.nn.modules.transformer` only for vanilla pre-norm).

**Coverage: 100% + extras.**

### Attention

ferrotorch: `MultiheadAttention`, `flash_attention`, `standard_attention`,
`flex_attention` (with `BlockMask`, `causal_score_mod`, `alibi_score_mod`,
`relative_position_bias_score_mod`), `paged_attention`
(`KVPage`, `PagePool`, `PagedAttentionManager`, `PagedKVCache`).

torch: `MultiheadAttention`, `scaled_dot_product_attention`, `flex_attention`,
`bias.{LowerLeftCausalBias, etc.}`.

**Coverage: 100% + paged_attention is a ferrotorch original** (no torch
counterpart).

### Init

ferrotorch (14): `constant`, `zeros`, `ones`, `uniform`, `normal`,
`xavier_uniform`, `xavier_normal`, `kaiming_uniform`, `kaiming_normal`,
`trunc_normal_`, `orthogonal_`, `sparse_`, `dirac_`, `eye_`,
`NonLinearity` enum.

torch: + `calculate_gain(nonlinearity, param)` exposed as helper.

**Missing:** public `calculate_gain` helper. (Probably internal to
kaiming_*/xavier_* implementation — should be exposed.)

### Functional

ferrotorch (22): `linear`, `relu`, `sigmoid`, `tanh`, `gelu`, `gelu_with`,
`silu`, `softmax`, `log_softmax`, `leaky_relu`, `sum`, `mean`, `dropout`,
`mse_loss`, `cross_entropy`, `interpolate`, `grid_sample`, `affine_grid`,
`pixel_shuffle`, `pixel_unshuffle`, `unfold`, `fold`.

torch.nn.functional (91 functions). **Major gap (~70 missing functional forms).**

The biggest absent groups:
- All conv functional forms (`conv1d/2d/3d`, `conv_transpose*d`)
- All pool functional forms (`max_pool*d`, `avg_pool*d`, `adaptive_*_pool*d`,
  `lp_pool*d`, `fractional_max_pool*d`)
- All padding functional forms (`pad`, `reflection_pad*`, `replication_pad*`,
  `circular_pad*`)
- Most loss functional forms (`l1_loss`, `nll_loss`, `binary_cross_entropy`,
  `binary_cross_entropy_with_logits`, `kl_div`, `huber_loss`, `smooth_l1_loss`,
  `cosine_embedding_loss`, `hinge_embedding_loss`, `multi_margin_loss`,
  `multilabel_*`, `triplet_margin_loss`, `ctc_loss`, `gaussian_nll_loss`,
  `poisson_nll_loss`)
- `embedding`, `embedding_bag`
- `batch_norm`, `instance_norm`, `layer_norm`, `group_norm`, `rms_norm`,
  `local_response_norm`
- `multi_head_attention_forward`, `scaled_dot_product_attention`
- Most activation functional forms (`prelu`, `elu`, `selu`, `celu`, `glu`,
  `mish`, `softmin`, `softplus`, `softshrink`, `hardshrink`, `tanhshrink`,
  `hardsigmoid`, `hardtanh`, `hardswish`, `relu6`, `rrelu`, `threshold`,
  `softsign`, `logsigmoid`)
- `pairwise_distance`, `cosine_similarity`, `cdist`
- `one_hot`, `normalize`, `gumbel_softmax`

**This is the largest single API gap in the workspace.** Most module
implementations bypass functional and call core ops directly. For idiomatic
porting from torch and for `torch.nn.functional` users coming from Python,
the functional surface needs to be filled in (mostly thin wrappers).

### Utils

ferrotorch (2): `clip_grad_norm_`, `clip_grad_value_`.

torch.nn.utils. **Missing:**
- `weight_norm` / `remove_weight_norm`
- `spectral_norm` / `remove_spectral_norm`
- `parametrize` (constraint registry)
- `parameters_to_vector` / `vector_to_parameters`
- `prune.*` (most of pruning is in `ferrotorch-core::pruning` already)
- `fuse_conv_bn_eval`
- `rnn::PackedSequence` (already in `rnn_utils`)
- `stateless` (call_with_external_params — see `functional_call`/
  `stack_module_state` from `torch.func`)

### Quantization

ferrotorch (`qat.rs`): `ObserverType`, `QatConfig`, `QatModel`,
`QuantizedModel`, `prepare_qat`.

torch: `torch.nn.qat`, `torch.nn.quantizable`, `torch.nn.quantized` — three
layered modules with `Linear`/`Conv*` quant variants.

ferrotorch has **module-level QAT scaffolding** but lacks per-layer quant
variants (`nn.qat.Linear`, `nn.quantized.Linear`, `nn.quantized.Conv2d`).
For inference-only quantized graphs this is mostly fine via `quantize.rs` in
core; for a torch-native flow it's a gap.

### Sparse / adaptive

ferrotorch: `Embedding`, `EmbeddingBag`. (Sparse module is `torch.nn.modules.sparse` and only contains these two — covered.)

torch.nn.modules.adaptive: `AdaptiveLogSoftmaxWithLoss` —
**missing in ferrotorch.** Used for very large vocabulary efficient softmax;
matters for old-style language models.

### Extras (no torch counterpart)

- **LoRA**: `LoRALinear` — direct LoRA adapter for parameter-efficient
  fine-tuning. Torch has no built-in LoRA (you use PEFT externally).
- **Paged attention**: `KVPage`, `PagePool`, `PagedAttentionManager`,
  `PagedKVCache` — ferrotorch-paged work landing here.
- **Flex attention** with builder helpers `causal_score_mod`,
  `alibi_score_mod`, `relative_position_bias_score_mod`.
- **RoPE primitives** (`RoPEConvention`, `RoPEScaling`,
  `RotaryPositionEmbedding`) and `SwiGLU` as first-class modules.
- **`#[derive(Module)]`** macro auto-generates `parameters/named_parameters/
  train/eval/is_training`.

## Concrete recommendations

1. **Fill out `functional.rs`** to torch.nn.functional parity. Mostly thin
   wrappers around module impls. This is the biggest gap by surface area
   (~70 functions). Suggested phases:
   - Phase A: conv, pool, pad functional forms (10-15 fns each)
   - Phase B: all loss functional forms (15+ fns)
   - Phase C: norm functional forms (6 fns)
   - Phase D: remaining activations (~15 fns)
   - Phase E: `embedding`, `embedding_bag`, `multi_head_attention_forward`,
     `scaled_dot_product_attention`, `one_hot`, `normalize`, `gumbel_softmax`
2. **Add buffer concept to `Module` trait** — `buffers()` / `named_buffers()`
   / `register_buffer()`. Required for clean BatchNorm running stats and
   precomputed RoPE tables. Must round-trip through `state_dict`.
3. **Add submodule traversal** — `modules()` / `named_modules()` /
   `children()` / `named_children()` / `apply(fn)`. Currently must enumerate
   parameters, which loses module identity.
4. **Add hook registration** to `Module` trait (currently `HookedModule` is
   a wrapper-only opt-in; torch users expect `.register_forward_hook()`
   directly on any module).
5. **Add `zero_grad()` and `requires_grad_(bool)`** on `Module`.
6. **Add `SyncBatchNorm`** — needed for distributed training (depends on
   `ferrotorch-distributed`).
7. **Add `Bilinear`, `AdaptiveLogSoftmaxWithLoss`, `MaxUnpool1d/3d`,
   `LPPool3d`, `FractionalMaxPool3d`, `LazyConvTranspose*`** — fills the
   small remaining gaps in module classes.
8. **Add `weight_norm`, `spectral_norm`, `parametrize`,
   `parameters_to_vector`** in `utils.rs`.
9. **Expose `calculate_gain`** publicly in `init.rs`.
10. **Add `functional_call` and `stack_module_state`** here (depends on
    `Module` trait — better home than core).
11. **Wire `LazyBatchNorm*`/`LazyInstanceNorm*`** if lazy modules are valued
    (currently lazy linear/conv exist, lazy norm doesn't).

**Do not split ferrotorch-nn.** It cleanly maps to `torch.nn` as a single
crate. The derive macro splitting (`ferrotorch-nn-derive`) is a Rust-specific
constraint (proc-macro crates must be separate) and is correct as-is.

## ferrotorch-nn-derive notes

The derive macro auto-generates `parameters`, `parameters_mut`,
`named_parameters`, `train`, `eval`, `is_training`. Field attributes control
how each field is treated.

**Gap:** does not auto-generate `buffers()`, `named_buffers()`,
`modules()`, `named_modules()`, `state_dict`/`load_state_dict` (the latter
two come for free from the trait's default impls but only over `parameters`,
not `buffers`).

When buffers land on the trait, the derive must be extended to recognize a
`#[buffer]` field attribute and generate the matching `buffers/named_buffers`
methods.
