//! GPU-resident Llama 3 inference.
//!
//! [`LlamaGpuInferencer`] owns the full model weights as
//! `CudaSlice<u16>` (bf16 bit layout) in VRAM and runs the forward
//! pass directly against the `ferrotorch-gpu` hand-written PTX bf16
//! kernels. This is the llama.cpp-style architecture: no generic
//! `Tensor<T>` dispatch, no CPU round-trips between layers, no
//! external toolchain. Every op in the forward pass is a CUDA
//! kernel launch against `CudaSlice<u16>` buffers.
//!
//! Enabled by the `cuda` feature on `ferrotorch-llama`, which pulls
//! in `ferrotorch-gpu` and `cudarc`.

#![cfg(feature = "cuda")]

use cudarc::driver::CudaSlice;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Tensor};
use ferrotorch_gpu::{
    GpuDevice, gpu_add_bf16, gpu_causal_mask_bf16, gpu_embedding_gather_bf16,
    gpu_matmul_bf16_bf16_nt, gpu_matmul_bf16_bf16_strided_batched,
    gpu_matmul_bf16_bf16_strided_batched_nt, gpu_mul_bf16, gpu_repeat_kv_bf16, gpu_rmsnorm_bf16,
    gpu_rope_half_bf16, gpu_silu_bf16, gpu_softmax_bf16, gpu_transpose_from_heads_bf16,
    gpu_transpose_to_heads_bf16,
};
use ferrotorch_nn::StateDict;
use half::bf16;

use crate::config::LlamaConfig;

/// All the weights that make up one Llama decoder layer, uploaded to GPU.
pub struct LlamaGpuLayer {
    pub input_norm: CudaSlice<u16>,      // [hidden]
    pub q_proj: CudaSlice<u16>,          // [hidden, hidden]
    pub k_proj: CudaSlice<u16>,          // [n_kv_heads * head_dim, hidden]
    pub v_proj: CudaSlice<u16>,          // [n_kv_heads * head_dim, hidden]
    pub o_proj: CudaSlice<u16>,          // [hidden, hidden]
    pub post_attn_norm: CudaSlice<u16>,  // [hidden]
    pub gate_proj: CudaSlice<u16>,       // [intermediate, hidden]
    pub up_proj: CudaSlice<u16>,         // [intermediate, hidden]
    pub down_proj: CudaSlice<u16>,       // [hidden, intermediate]
}

/// A GPU-resident Llama model ready for inference.
pub struct LlamaGpuInferencer {
    pub config: LlamaConfig,
    pub device: GpuDevice,
    pub embed_tokens: CudaSlice<u16>,
    pub layers: Vec<LlamaGpuLayer>,
    pub norm: CudaSlice<u16>,
    pub lm_head: CudaSlice<u16>,
    pub cos_cache: CudaSlice<u16>,
    pub sin_cache: CudaSlice<u16>,
}

impl LlamaGpuInferencer {
    /// Upload an HF-style `StateDict<bf16>` into GPU memory and build
    /// an inferencer ready to run. The `state` map is drained as each
    /// tensor is uploaded so the host copy can be freed incrementally.
    pub fn new(
        config: LlamaConfig,
        mut state: StateDict<bf16>,
        device: GpuDevice,
    ) -> FerrotorchResult<Self> {
        config.validate()?;

        let embed_tokens = upload_bf16_tensor(
            &mut state,
            "model.embed_tokens.weight",
            &[config.vocab_size, config.hidden_size],
            &device,
        )?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(upload_layer(&config, i, &mut state, &device)?);
        }

        let norm = upload_bf16_tensor(
            &mut state,
            "model.norm.weight",
            &[config.hidden_size],
            &device,
        )?;
        let lm_head = upload_bf16_tensor(
            &mut state,
            "lm_head.weight",
            &[config.vocab_size, config.hidden_size],
            &device,
        )?;

        let (cos_cache, sin_cache) = build_rope_caches(&config, &device)?;

        Ok(Self {
            config,
            device,
            embed_tokens,
            layers,
            norm,
            lm_head,
            cos_cache,
            sin_cache,
        })
    }

    /// Forward `ids` through the network and return the last-token
    /// logits as `Vec<f32>` of length `vocab_size`.
    pub fn forward_from_ids(&self, ids: &[u32]) -> FerrotorchResult<Vec<f32>> {
        if ids.is_empty() {
            return Err(FerrotorchError::InvalidArgument {
                message: "forward_from_ids: empty id list".into(),
            });
        }

        let cfg = &self.config;
        let dev = &self.device;
        let seq = ids.len();
        let hidden = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let n_kv = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();
        let group = cfg.kv_group_size();
        let ffn = cfg.intermediate_size;
        let vocab = cfg.vocab_size;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        if seq > cfg.max_position_embeddings {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "seq_len {seq} exceeds max_position_embeddings {}",
                    cfg.max_position_embeddings
                ),
            });
        }

        // Upload token ids.
        let ids_gpu = dev
            .stream()
            .clone_htod(ids)
            .map_err(map_driver_err)?;

        // Embedding lookup: [seq, hidden]
        let mut hidden_buf =
            gpu_embedding_gather_bf16(&self.embed_tokens, &ids_gpu, hidden, dev)
                .map_err(map_gpu_err)?;

        for (i, layer) in self.layers.iter().enumerate() {
            // ----- attention sub-block -----
            let h_norm = gpu_rmsnorm_bf16(
                &hidden_buf,
                &layer.input_norm,
                seq,
                hidden,
                cfg.rms_norm_eps as f32,
                dev,
            )
            .map_err(map_gpu_err)?;

            // Q,K,V projections — Linear is `x @ W^T`, with W stored
            // row-major as [out, in] in HF layout.
            let q = gpu_matmul_bf16_bf16_nt(&h_norm, &layer.q_proj, seq, hidden, hidden, dev)
                .map_err(map_gpu_err)?;
            let k = gpu_matmul_bf16_bf16_nt(
                &h_norm,
                &layer.k_proj,
                seq,
                hidden,
                n_kv * head_dim,
                dev,
            )
            .map_err(map_gpu_err)?;
            let v = gpu_matmul_bf16_bf16_nt(
                &h_norm,
                &layer.v_proj,
                seq,
                hidden,
                n_kv * head_dim,
                dev,
            )
            .map_err(map_gpu_err)?;

            // Reshape to per-head layout.
            let q_heads = gpu_transpose_to_heads_bf16(&q, n_heads, seq, head_dim, dev)
                .map_err(map_gpu_err)?;
            let k_heads = gpu_transpose_to_heads_bf16(&k, n_kv, seq, head_dim, dev)
                .map_err(map_gpu_err)?;
            let v_heads = gpu_transpose_to_heads_bf16(&v, n_kv, seq, head_dim, dev)
                .map_err(map_gpu_err)?;

            // RoPE on Q,K.
            let q_rot = gpu_rope_half_bf16(
                &q_heads,
                &self.cos_cache,
                &self.sin_cache,
                n_heads,
                seq,
                head_dim,
                0,
                dev,
            )
            .map_err(map_gpu_err)?;
            let k_rot = gpu_rope_half_bf16(
                &k_heads,
                &self.cos_cache,
                &self.sin_cache,
                n_kv,
                seq,
                head_dim,
                0,
                dev,
            )
            .map_err(map_gpu_err)?;

            // GQA broadcast KV heads to match Q head count.
            let k_full = gpu_repeat_kv_bf16(&k_rot, n_kv, group, seq, head_dim, dev)
                .map_err(map_gpu_err)?;
            let v_full = gpu_repeat_kv_bf16(&v_heads, n_kv, group, seq, head_dim, dev)
                .map_err(map_gpu_err)?;

            // Attention scores: scores[h] = Q_h @ K_h^T * scale, batched over heads.
            let mut scores = gpu_matmul_bf16_bf16_strided_batched_nt(
                &q_rot,
                &k_full,
                seq,
                head_dim,
                seq,
                n_heads,
                seq * head_dim,
                seq * head_dim,
                scale,
                dev,
            )
            .map_err(map_gpu_err)?;

            // Causal mask in-place.
            gpu_causal_mask_bf16(&mut scores, n_heads, seq, seq, dev)
                .map_err(map_gpu_err)?;

            // Softmax per row — reshape [n_heads, seq, seq] to [n_heads*seq, seq] for
            // the row-wise softmax kernel. This is a shape reinterpret, not a copy.
            let attn_weights = gpu_softmax_bf16(&scores, n_heads * seq, seq, dev)
                .map_err(map_gpu_err)?;

            // attn_out[h] = attn_weights[h] @ V_h
            let attn_out = gpu_matmul_bf16_bf16_strided_batched(
                &attn_weights,
                &v_full,
                seq,
                seq,
                head_dim,
                n_heads,
                seq * seq,
                seq * head_dim,
                1.0,
                dev,
            )
            .map_err(map_gpu_err)?;

            // Flatten heads back to [seq, n_heads * head_dim].
            let attn_flat =
                gpu_transpose_from_heads_bf16(&attn_out, n_heads, seq, head_dim, dev)
                    .map_err(map_gpu_err)?;

            // Output projection.
            let attn_proj =
                gpu_matmul_bf16_bf16_nt(&attn_flat, &layer.o_proj, seq, hidden, hidden, dev)
                    .map_err(map_gpu_err)?;

            // Residual.
            hidden_buf = gpu_add_bf16(&hidden_buf, &attn_proj, dev).map_err(map_gpu_err)?;

            // ----- MLP sub-block -----
            let h_norm2 = gpu_rmsnorm_bf16(
                &hidden_buf,
                &layer.post_attn_norm,
                seq,
                hidden,
                cfg.rms_norm_eps as f32,
                dev,
            )
            .map_err(map_gpu_err)?;

            let gate =
                gpu_matmul_bf16_bf16_nt(&h_norm2, &layer.gate_proj, seq, hidden, ffn, dev)
                    .map_err(map_gpu_err)?;
            let up =
                gpu_matmul_bf16_bf16_nt(&h_norm2, &layer.up_proj, seq, hidden, ffn, dev)
                    .map_err(map_gpu_err)?;

            let silu_gate = gpu_silu_bf16(&gate, dev).map_err(map_gpu_err)?;
            let gated = gpu_mul_bf16(&silu_gate, &up, dev).map_err(map_gpu_err)?;

            let down =
                gpu_matmul_bf16_bf16_nt(&gated, &layer.down_proj, seq, ffn, hidden, dev)
                    .map_err(map_gpu_err)?;

            hidden_buf = gpu_add_bf16(&hidden_buf, &down, dev).map_err(map_gpu_err)?;
            let _ = i; // silence unused warning; kept for future debug prints
        }

        // Final norm + LM head.
        let final_norm = gpu_rmsnorm_bf16(
            &hidden_buf,
            &self.norm,
            seq,
            hidden,
            cfg.rms_norm_eps as f32,
            dev,
        )
        .map_err(map_gpu_err)?;
        let logits =
            gpu_matmul_bf16_bf16_nt(&final_norm, &self.lm_head, seq, hidden, vocab, dev)
                .map_err(map_gpu_err)?;

        // Download just the last-token row of logits and convert bf16 -> f32.
        let logits_host: Vec<u16> = dev
            .stream()
            .clone_dtoh(&logits)
            .map_err(map_driver_err)?;
        let last_offset = (seq - 1) * vocab;
        let last_row = &logits_host[last_offset..last_offset + vocab];
        Ok(last_row
            .iter()
            .map(|&b| bf16::from_bits(b).to_f32())
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn upload_layer(
    cfg: &LlamaConfig,
    i: usize,
    state: &mut StateDict<bf16>,
    device: &GpuDevice,
) -> FerrotorchResult<LlamaGpuLayer> {
    let kv_dim = cfg.num_key_value_heads * cfg.head_dim();
    Ok(LlamaGpuLayer {
        input_norm: upload_bf16_tensor(
            state,
            &format!("model.layers.{i}.input_layernorm.weight"),
            &[cfg.hidden_size],
            device,
        )?,
        q_proj: upload_bf16_tensor(
            state,
            &format!("model.layers.{i}.self_attn.q_proj.weight"),
            &[cfg.hidden_size, cfg.hidden_size],
            device,
        )?,
        k_proj: upload_bf16_tensor(
            state,
            &format!("model.layers.{i}.self_attn.k_proj.weight"),
            &[kv_dim, cfg.hidden_size],
            device,
        )?,
        v_proj: upload_bf16_tensor(
            state,
            &format!("model.layers.{i}.self_attn.v_proj.weight"),
            &[kv_dim, cfg.hidden_size],
            device,
        )?,
        o_proj: upload_bf16_tensor(
            state,
            &format!("model.layers.{i}.self_attn.o_proj.weight"),
            &[cfg.hidden_size, cfg.hidden_size],
            device,
        )?,
        post_attn_norm: upload_bf16_tensor(
            state,
            &format!("model.layers.{i}.post_attention_layernorm.weight"),
            &[cfg.hidden_size],
            device,
        )?,
        gate_proj: upload_bf16_tensor(
            state,
            &format!("model.layers.{i}.mlp.gate_proj.weight"),
            &[cfg.intermediate_size, cfg.hidden_size],
            device,
        )?,
        up_proj: upload_bf16_tensor(
            state,
            &format!("model.layers.{i}.mlp.up_proj.weight"),
            &[cfg.intermediate_size, cfg.hidden_size],
            device,
        )?,
        down_proj: upload_bf16_tensor(
            state,
            &format!("model.layers.{i}.mlp.down_proj.weight"),
            &[cfg.hidden_size, cfg.intermediate_size],
            device,
        )?,
    })
}

/// Pop a tensor out of the `StateDict`, validate its shape, convert
/// its `Vec<bf16>` data to `Vec<u16>` bits, and upload to the device.
/// The tensor is removed from `state` so the host copy can be freed.
fn upload_bf16_tensor(
    state: &mut StateDict<bf16>,
    name: &str,
    expected_shape: &[usize],
    device: &GpuDevice,
) -> FerrotorchResult<CudaSlice<u16>> {
    let tensor = state.remove(name).ok_or_else(|| {
        FerrotorchError::InvalidArgument {
            message: format!("missing tensor in StateDict: \"{name}\""),
        }
    })?;
    if tensor.shape() != expected_shape {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "{name}: expected shape {expected_shape:?}, got {:?}",
                tensor.shape()
            ),
        });
    }
    let bits = bf16_tensor_as_bits(&tensor)?;
    device
        .stream()
        .clone_htod(&bits)
        .map_err(map_driver_err)
}

/// View a `Tensor<bf16>`'s underlying data as `Vec<u16>` bits.
/// Allocates a fresh Vec — correct and simple. Callers immediately
/// upload to GPU and drop this Vec, so the duplication is bounded
/// to one tensor at a time.
fn bf16_tensor_as_bits(t: &Tensor<bf16>) -> FerrotorchResult<Vec<u16>> {
    let data = t.data()?;
    Ok(data.iter().map(|v| v.to_bits()).collect())
}

/// Precompute cos/sin caches for Llama's half-rotation RoPE, upload
/// as bf16. Each cache is `[max_seq, head_dim / 2]` — the caller
/// slices into `[pos * half_dim + d]` per the kernel's indexing.
fn build_rope_caches(
    cfg: &LlamaConfig,
    device: &GpuDevice,
) -> FerrotorchResult<(CudaSlice<u16>, CudaSlice<u16>)> {
    let head_dim = cfg.head_dim();
    if head_dim == 0 || head_dim % 2 != 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "RoPE requires even head_dim".into(),
        });
    }
    let half = head_dim / 2;
    let max_seq = cfg.max_position_embeddings;
    let base = cfg.rope_theta;

    let inv_freq: Vec<f64> = (0..half)
        .map(|i| 1.0 / base.powf(2.0 * i as f64 / head_dim as f64))
        .collect();

    let mut cos_bits = Vec::with_capacity(max_seq * half);
    let mut sin_bits = Vec::with_capacity(max_seq * half);
    for pos in 0..max_seq {
        for d in 0..half {
            let angle = pos as f64 * inv_freq[d];
            cos_bits.push(bf16::from_f32(angle.cos() as f32).to_bits());
            sin_bits.push(bf16::from_f32(angle.sin() as f32).to_bits());
        }
    }

    let cos_cache = device
        .stream()
        .clone_htod(&cos_bits)
        .map_err(map_driver_err)?;
    let sin_cache = device
        .stream()
        .clone_htod(&sin_bits)
        .map_err(map_driver_err)?;
    Ok((cos_cache, sin_cache))
}

fn map_gpu_err(e: ferrotorch_gpu::GpuError) -> FerrotorchError {
    FerrotorchError::InvalidArgument {
        message: format!("gpu error: {e}"),
    }
}

fn map_driver_err(e: cudarc::driver::DriverError) -> FerrotorchError {
    FerrotorchError::InvalidArgument {
        message: format!("cuda driver error: {e}"),
    }
}

