# Audit: `ferrotorch-llama` vs HF transformers Llama reference

There is no `pytorch/torch/llama` to compare against. The reference is
**HF transformers `LlamaForCausalLM`** + Meta's reference impl + the
ferrotorch-paged sister crate.

## Modules

| Module | Role |
|---|---|
| `config` | `LlamaConfig`, `LlamaActivation` — JSON-config parsing |
| `attention` | `LlamaAttention` — Grouped Query Attention + RoPE |
| `mlp` | `LlamaMLP` — SwiGLU gate/up/down projections |
| `layer` | `LlamaDecoderLayer` — pre-norm wrapper around attention + MLP |
| `model` | `LlamaModel`, `LlamaForCausalLM` — full stack with embeddings + lm_head |
| `gguf_remap` | `gguf_key_to_hf`, `gguf_to_hf_state_dict` — convert GGUF naming to HF naming |
| `grammar/` | Constrained decoding (JSON schema, DFA, GPU dispatch) |
| `gpu` (feature `cuda`) | `LlamaGpuInferencer`, `LlamaGpuLayer`, `ProfiledForwardResult` — fused GPU inference path |

## Coverage vs HF transformers

HF transformers `LlamaForCausalLM` exposes:
- Configuration (`LlamaConfig`)
- Model classes: `LlamaModel`, `LlamaForCausalLM`,
  `LlamaForSequenceClassification`, `LlamaForQuestionAnswering`,
  `LlamaForTokenClassification`
- Forward pass: input_ids, attention_mask, position_ids, past_key_values,
  inputs_embeds, labels, use_cache, output_attentions,
  output_hidden_states, return_dict, cache_position
- Generation: `.generate()` — greedy, beam, sampling, contrastive
  search, speculative decoding
- Tokenizer (`LlamaTokenizer`, `LlamaTokenizerFast`)
- Attention implementations: eager, sdpa, flash_attention_2, flex
- Quantization: bitsandbytes, gptq, awq, hqq, fp8

| Capability | ferrotorch-llama | HF transformers | Status |
|---|---|---|---|
| Config | `LlamaConfig` | ✅ | ✅ |
| LlamaModel | ✅ | ✅ | ✅ |
| LlamaForCausalLM | ✅ | ✅ | ✅ |
| Other heads (sequence/QA/token) | ❌ | ✅ | gap |
| RoPE (incl. NTK / dynamic / linear scaling) | basic via `RotaryPositionEmbedding` | ✅ all variants | partial — #515 NTK open |
| GQA (grouped-query attention) | ✅ | ✅ | ✅ |
| SwiGLU MLP | ✅ via `LlamaMLP` | ✅ | ✅ |
| Pre-norm RMSNorm | ✅ | ✅ | ✅ |
| KV cache | via `KVCache` from `ferrotorch-nn` | ✅ | ✅ |
| Flash Attention | yes via `flash_attention.rs` in core | ✅ FA2 | partial |
| SDPA / flex attention | yes | ✅ | ✅ |
| Paged attention | yes (sister crate `ferrotorch-paged`) | partial (vLLM-style external) | **superset** |
| Generation: greedy decode | likely yes | ✅ | unclear from listing |
| Generation: beam | unclear | ✅ | likely gap |
| Generation: sampling (top-k / top-p / temperature) | unclear | ✅ | likely gap |
| Generation: speculative decoding | partial via `grammar/` for constrained | partial | unclear |
| Constrained decoding (JSON schema, regex, DFA) | ✅ via `grammar/` | partial (ext: outlines, lm-format-enforcer) | **superset** — first-class in ferrotorch |
| HF state dict load (rename keys) | ✅ via `LlamaForCausalLM::load_hf_state_dict` | ✅ native | ✅ |
| GGUF state dict load | ✅ via `gguf_remap` | partial (HF supports GGUF in transformers via convert) | ✅ |
| safetensors load (sharded) | depends on `ferrotorch-serialize::load_safetensors_sharded` | ✅ | partial (#509) |
| bf16 weights on GPU | ✅ via `LlamaGpuInferencer` | ✅ | ✅ |
| Mixed precision training | depends on core autocast + optim grad scaler | ✅ | ✅ |
| Quantization: bitsandbytes (4/8-bit) | partial via core quantize + GGUF | ✅ | partial |
| Quantization: GPTQ | ❌ | ✅ | gap |
| Quantization: AWQ | ❌ | ✅ | gap |
| Quantization: HQQ | ❌ | ✅ | gap |
| Tokenizer | via `ferrotorch-tokenize` (HF tokenizers crate) | ✅ | ✅ |
| Chat template | gap (in tokenize) | ✅ | gap |

## Strengths over HF transformers

1. **Constrained decoding is built-in** — JSON schema + DFA + GPU
   dispatch in `grammar/`. HF needs external `outlines` /
   `lm-format-enforcer`.
2. **Paged attention** via the sister `ferrotorch-paged` crate gives a
   vLLM-style path natively.
3. **GGUF interop** is first-class — load llama.cpp weights directly,
   dequantize on GPU via `ferrotorch-cubecl::quant`.
4. **Single-binary deployment** — no Python runtime, no torch installation.
5. **bf16 GPU kernels are hand-tuned** — see `ferrotorch-gpu::bf16`
   (rmsnorm, rope_half, repeat_kv, etc.) which beats the HF-default
   "compute in fp32, cast to bf16 at boundaries" path.

## Gaps

1. **Generation API** — likely thin. Need explicit:
   - `generate(input_ids, max_new_tokens, do_sample, temperature, top_k,
     top_p, repetition_penalty, stop_tokens)`
   - Beam search (`num_beams`, `length_penalty`, `early_stopping`)
   - Speculative decoding (draft model + verification)
   - Streaming (token-by-token callback)
   Need to read `model.rs` to confirm what's there. Crate description
   doesn't mention `.generate()`.
2. **Other model heads** — `LlamaForSequenceClassification`, etc. Easy
   adds.
3. **Modern RoPE variants** — NTK-aware (#515), dynamic NTK, YaRN, linear
   scaling. Required for context-length extension.
4. **Quantization beyond GGUF**:
   - GPTQ — 4-bit weight-only with per-group quantization
   - AWQ — activation-aware weight quantization
   - HQQ — half-quadratic quantization
5. **Chat templating** — depends on `ferrotorch-tokenize` adding Jinja2
   support.
6. **Other Llama variants** — Llama 3.1 (128k context, NTK), Llama 3.2
   (vision, multimodal), Llama 3.3 (newer training). Likely the existing
   code with config tweaks supports them, needs validation.
7. **Multimodal Llama 3.2** — vision encoder, cross-attention layers.

## Sister crate: ferrotorch-paged

Lives outside this workspace at `~/ferrotorch-paged`. Hosts paged-
attention infrastructure (`PagedKVCache`, `PagePool`, `PagedAttentionManager`
in `ferrotorch-nn::paged_attention` are the public re-exports).

Many open issues (#520-#542 in the locks list) track this work.

## Recommendations

1. **Verify and document the generation API surface** — read `model.rs`,
   list what `LlamaForCausalLM::generate` (or equivalent) supports, fill
   gaps.
2. **Add NTK-aware RoPE scaling** (#515).
3. **Add GPTQ / AWQ / HQQ loaders** — at minimum, unpack their weight
   formats and integrate with `ferrotorch-core::quantize`.
4. **Add Llama variant tests** — verify Llama 2 (7B/13B), Llama 3.1
   (128k), Llama 3.3 round-trip cleanly.
5. **Wire chat templates** through `ferrotorch-tokenize` so
   `apply_chat_template(tokenizer, messages)` works.
6. **Document the paged-attention story** — make clear which crate owns
   what (paged primitives in `ferrotorch-nn`, paged store/swap in the
   sister crate, llama-specific glue here in `gpu.rs`).
7. **Add other heads** if anyone needs them.
8. **Multimodal Llama 3.2** is its own large workstream; defer unless
   targeted.

## Status

ferrotorch-llama is **the most novel crate in the workspace** —
it's where ferrotorch-paged, GGUF interop, GPU bf16 kernels, and
constrained decoding all converge into a working LLM serving stack. The
Llama 3 8B story (per `MEMORY.md`) is what landed first.

**Do not split.** Maps cleanly to a single model family. If more model
families are added (Mistral, Qwen, Gemma, Phi), each gets its own crate
parallel to this one.

## Related issues
- #515 — NTK-aware RoPE scaling
- #519 — Llama 3 8B on GPU end-to-end
- #513 — Llama 3 8B end-to-end smoke example
- #511 — bf16 hybrid GPU weight storage
- #509 — HF Hub auth + sharded download
- #520-542 — paged attention / ShadowLLM / paging infrastructure
- #544-552 — ShadowLLM training, ensembles, threshold sweeps
