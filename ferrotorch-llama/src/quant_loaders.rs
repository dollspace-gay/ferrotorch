//! Weight unpackers for HF-quantized LLM checkpoints. (#593)
//!
//! Supports the two formats most commonly shipped on the Hub:
//!
//! - **GPTQ** (`q4` / `q8`, group-wise asymmetric int quantization with
//!   per-group scales and zero-points, packed into i32 along the
//!   in_features axis). Reference: Frantar et al. 2023.
//! - **AWQ** (`q4`, group-wise scales but with a fixed *channel order*
//!   shuffle in the packed layout, originally introduced by MIT-Han-Lab).
//!
//! Both unpackers produce dequantized `f32` weight matrices that callers
//! can then load via the standard state-dict path. The unpackers do not
//! own a tokenizer or model — they operate purely on the four packed
//! tensors (`qweight`, `qzeros`, `scales`, optional `g_idx`) shipped
//! by the HF Transformers `auto_gptq` / `autoawq` libraries.
//!
//! HQQ unpacking is more involved (per-row half-precision scales + an
//! offset table) and is deferred — see follow-up issue.

use ferrotorch_core::{FerrotorchError, FerrotorchResult};

/// A 4-bit GPTQ-packed weight tile. Layout matches `auto_gptq` v0.7+.
///
/// Shapes (with `out_features = N` and `in_features = K`, group_size = G):
/// - `qweight`: `[K / 8, N]` packed i32 (8 int4 weights per i32, along K).
/// - `qzeros`:  `[K / G, N / 8]` packed i32 (8 int4 zeros per i32, along N).
/// - `scales`:  `[K / G, N]` `f32` per-group, per-out-channel scales.
///
/// `g_idx` is the optional permutation table for `act_order=True` GPTQ;
/// pass `None` when the checkpoint was saved with `act_order=False`
/// (the common case).
#[derive(Debug)]
pub struct GptqQ4 {
    pub qweight: Vec<i32>,
    pub qzeros: Vec<i32>,
    pub scales: Vec<f32>,
    pub g_idx: Option<Vec<i32>>,
    pub in_features: usize,
    pub out_features: usize,
    pub group_size: usize,
}

/// Dequantize a 4-bit GPTQ weight matrix to row-major `f32`.
///
/// Returns `[out_features, in_features]` row-major (matches torch's
/// `Linear.weight` shape, ready for the `linear` op).
///
/// # Errors
/// - In/out feature counts not divisible by 8 (packing constraint).
/// - in_features not divisible by group_size.
/// - Any of the packed buffers has the wrong length for the declared shape.
pub fn dequantize_gptq_q4(packed: &GptqQ4) -> FerrotorchResult<Vec<f32>> {
    let GptqQ4 {
        qweight,
        qzeros,
        scales,
        g_idx,
        in_features,
        out_features,
        group_size,
    } = packed;
    let in_features = *in_features;
    let out_features = *out_features;
    let group_size = *group_size;

    if out_features % 8 != 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "GPTQ q4: out_features ({out_features}) must be a multiple of 8"
            ),
        });
    }
    if in_features % 8 != 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "GPTQ q4: in_features ({in_features}) must be a multiple of 8"
            ),
        });
    }
    if in_features % group_size != 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "GPTQ q4: in_features ({in_features}) must be a multiple of group_size ({group_size})"
            ),
        });
    }
    let num_groups = in_features / group_size;
    let qweight_rows = in_features / 8;

    if qweight.len() != qweight_rows * out_features {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "GPTQ q4: qweight len {} != expected {}",
                qweight.len(),
                qweight_rows * out_features
            ),
        });
    }
    if qzeros.len() != num_groups * (out_features / 8) {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "GPTQ q4: qzeros len {} != expected {}",
                qzeros.len(),
                num_groups * (out_features / 8)
            ),
        });
    }
    if scales.len() != num_groups * out_features {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "GPTQ q4: scales len {} != expected {}",
                scales.len(),
                num_groups * out_features
            ),
        });
    }
    if let Some(g) = g_idx {
        if g.len() != in_features {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "GPTQ q4: g_idx len {} != in_features {in_features}",
                    g.len()
                ),
            });
        }
    }

    // Output: row-major [out_features, in_features].
    let mut out = vec![0.0f32; out_features * in_features];

    for k in 0..in_features {
        // Group index for this k. With act_order, g_idx[k] gives it.
        let group = match g_idx {
            Some(g) => g[k] as usize,
            None => k / group_size,
        };
        // Locate the int32 row + nibble for k.
        let qrow = k / 8;
        let nibble_idx = k % 8;

        for n in 0..out_features {
            // Extract the 4-bit weight.
            let packed_w = qweight[qrow * out_features + n] as u32;
            let q = ((packed_w >> (4 * nibble_idx)) & 0xF) as i32;

            // Extract the 4-bit zero for (group, n).
            let qzeros_row = group;
            let zero_col = n / 8;
            let zero_nib = n % 8;
            let packed_z = qzeros[qzeros_row * (out_features / 8) + zero_col] as u32;
            let z = ((packed_z >> (4 * zero_nib)) & 0xF) as i32;
            // GPTQ stores zero - 1; reconstruct true zero.
            let zero = z + 1;

            let scale = scales[group * out_features + n];
            let dequant = (q - zero) as f32 * scale;
            out[n * in_features + k] = dequant;
        }
    }
    Ok(out)
}

/// AWQ 4-bit packed layout. The major difference from GPTQ is the
/// per-int32 channel-order shuffle: AWQ packs 8 int4 weights per i32
/// from out_channels in a specific order so that runtime dequantize
/// kernels can load consecutive weights in cache-friendly stripes.
#[derive(Debug)]
pub struct AwqQ4 {
    pub qweight: Vec<i32>,
    pub qzeros: Vec<i32>,
    pub scales: Vec<f32>,
    pub in_features: usize,
    pub out_features: usize,
    pub group_size: usize,
}

/// AWQ's int32 → int4 channel-shuffle order (see autoawq/awq_inference_engine).
/// AWQ packs `[N0, N4, N1, N5, N2, N6, N3, N7]` instead of `[N0..N7]`.
const AWQ_PACK_ORDER: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];

/// Dequantize a 4-bit AWQ weight matrix to row-major `f32` of shape
/// `[out_features, in_features]`.
pub fn dequantize_awq_q4(packed: &AwqQ4) -> FerrotorchResult<Vec<f32>> {
    let AwqQ4 {
        qweight,
        qzeros,
        scales,
        in_features,
        out_features,
        group_size,
    } = packed;
    let in_features = *in_features;
    let out_features = *out_features;
    let group_size = *group_size;

    if out_features % 8 != 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "AWQ q4: out_features ({out_features}) must be a multiple of 8"
            ),
        });
    }
    if in_features % group_size != 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "AWQ q4: in_features ({in_features}) must be a multiple of group_size ({group_size})"
            ),
        });
    }
    let num_groups = in_features / group_size;
    let n_packed = out_features / 8;

    // AWQ qweight shape: [in_features, out_features / 8] int32.
    if qweight.len() != in_features * n_packed {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "AWQ q4: qweight len {} != expected {}",
                qweight.len(),
                in_features * n_packed
            ),
        });
    }
    if qzeros.len() != num_groups * n_packed {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "AWQ q4: qzeros len {} != expected {}",
                qzeros.len(),
                num_groups * n_packed
            ),
        });
    }
    if scales.len() != num_groups * out_features {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "AWQ q4: scales len {} != expected {}",
                scales.len(),
                num_groups * out_features
            ),
        });
    }

    let mut out = vec![0.0f32; out_features * in_features];

    for k in 0..in_features {
        let group = k / group_size;
        for n_block in 0..n_packed {
            let packed_w = qweight[k * n_packed + n_block] as u32;
            let packed_z = qzeros[group * n_packed + n_block] as u32;
            for shuffle_idx in 0..8 {
                let lane = AWQ_PACK_ORDER[shuffle_idx];
                let q = ((packed_w >> (4 * lane)) & 0xF) as i32;
                let z = ((packed_z >> (4 * lane)) & 0xF) as i32;
                let n = n_block * 8 + shuffle_idx;
                let scale = scales[group * out_features + n];
                let dequant = (q - z) as f32 * scale;
                out[n * in_features + k] = dequant;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pack 8 nibbles (low → high) into one i32. Helper for synthesizing
    /// test inputs.
    fn pack8(nibbles: [u32; 8]) -> i32 {
        let mut v: u32 = 0;
        for (i, n) in nibbles.iter().enumerate() {
            v |= (n & 0xF) << (4 * i);
        }
        v as i32
    }

    // -- GPTQ tests --------------------------------------------------------

    #[test]
    fn gptq_q4_dequantize_one_group_identity() {
        // out_features = 8, in_features = 8 (one group at group_size=8).
        // Pack qweight so that for every (k, n), the int4 value equals k.
        // qzeros all 1 → "true" zero = 2; scales all 1.0.
        // Expected dequantized w[n, k] = (k - 2) * 1.0 = k - 2.
        let in_features = 8;
        let out_features = 8;
        let group_size = 8;
        let qweight_rows = in_features / 8; // 1
        let mut qweight = vec![0i32; qweight_rows * out_features];
        for n in 0..out_features {
            // Pack k = 0..8 along the 8 nibbles for this output column.
            let nibbles: [u32; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
            qweight[n] = pack8(nibbles);
        }
        let num_groups = in_features / group_size; // 1
        // qzeros: [num_groups, out_features/8] = [1, 1]. Pack 8 ones.
        let qzeros = vec![pack8([1; 8]); num_groups * (out_features / 8)];
        let scales = vec![1.0f32; num_groups * out_features];

        let packed = GptqQ4 {
            qweight,
            qzeros,
            scales,
            g_idx: None,
            in_features,
            out_features,
            group_size,
        };
        let out = dequantize_gptq_q4(&packed).unwrap();
        // [out_features, in_features] = [8, 8]; expect every (n, k) = k - 2.
        for n in 0..out_features {
            for k in 0..in_features {
                let v = out[n * in_features + k];
                assert!(
                    (v - (k as f32 - 2.0)).abs() < 1e-6,
                    "GPTQ dequant ({n}, {k}) = {v}, expected {}",
                    k as f32 - 2.0
                );
            }
        }
    }

    #[test]
    fn gptq_q4_rejects_non_multiple_of_8_dims() {
        let p = GptqQ4 {
            qweight: vec![0i32; 1],
            qzeros: vec![0i32; 1],
            scales: vec![0.0f32; 1],
            g_idx: None,
            in_features: 9,
            out_features: 8,
            group_size: 8,
        };
        let err = dequantize_gptq_q4(&p).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }

    #[test]
    fn gptq_q4_rejects_misaligned_group() {
        let p = GptqQ4 {
            qweight: vec![0i32; 16],
            qzeros: vec![0i32; 16],
            scales: vec![0.0f32; 16],
            g_idx: None,
            in_features: 16,
            out_features: 8,
            group_size: 7, // 16 % 7 != 0
        };
        let err = dequantize_gptq_q4(&p).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }

    #[test]
    fn gptq_q4_rejects_g_idx_length_mismatch() {
        let p = GptqQ4 {
            qweight: vec![0i32; 8],
            qzeros: vec![0i32; 1],
            scales: vec![0.0f32; 8],
            g_idx: Some(vec![0; 7]), // wrong length
            in_features: 8,
            out_features: 8,
            group_size: 8,
        };
        let err = dequantize_gptq_q4(&p).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }

    #[test]
    fn gptq_q4_with_two_groups_uses_per_group_scale() {
        // 16-in-features split into 2 groups of 8. Group 0 scales are 1.0,
        // group 1 scales are 2.0. Same q values everywhere → group 1 should
        // produce 2× the magnitude of group 0.
        let in_features = 16;
        let out_features = 8;
        let group_size = 8;
        let mut qweight = vec![0i32; (in_features / 8) * out_features];
        let q_const = 5;
        for n in 0..out_features {
            for k_block in 0..(in_features / 8) {
                qweight[k_block * out_features + n] = pack8([q_const; 8]);
            }
        }
        let qzeros = vec![pack8([0; 8]); 2 * (out_features / 8)];
        let mut scales = vec![1.0f32; 2 * out_features];
        for s in scales.iter_mut().skip(out_features) {
            *s = 2.0;
        }

        let packed = GptqQ4 {
            qweight,
            qzeros,
            scales,
            g_idx: None,
            in_features,
            out_features,
            group_size,
        };
        let out = dequantize_gptq_q4(&packed).unwrap();
        // q=5, zero = 0 + 1 = 1 → (5 - 1) = 4. group 0 → 4 × 1 = 4; group 1 → 4 × 2 = 8.
        for n in 0..out_features {
            for k in 0..8 {
                assert!((out[n * in_features + k] - 4.0).abs() < 1e-6);
            }
            for k in 8..16 {
                assert!((out[n * in_features + k] - 8.0).abs() < 1e-6);
            }
        }
    }

    // -- AWQ tests ---------------------------------------------------------

    #[test]
    fn awq_q4_dequantize_uniform_inputs() {
        // For uniform q across all packed lanes, the AWQ shuffle is
        // observable only via which n the value lands in. Synthesize a
        // case where every weight unpacks to q = 7, zero = 3, scale = 1
        // → dequantized w = 4 everywhere.
        let in_features = 8;
        let out_features = 8;
        let group_size = 8;
        let n_packed = out_features / 8;
        let qweight = vec![pack8([7; 8]); in_features * n_packed];
        let qzeros = vec![pack8([3; 8]); 1 * n_packed];
        let scales = vec![1.0f32; 1 * out_features];

        let packed = AwqQ4 {
            qweight,
            qzeros,
            scales,
            in_features,
            out_features,
            group_size,
        };
        let out = dequantize_awq_q4(&packed).unwrap();
        for v in &out {
            assert!((v - 4.0).abs() < 1e-6);
        }
    }

    #[test]
    fn awq_q4_shuffle_order_is_distinct_from_gptq() {
        // Pack distinct q values per lane (0..8) and verify the AWQ
        // unpacker emits them in the expected channel order
        // [0, 4, 1, 5, 2, 6, 3, 7]. Use a single in_features=1, OF=8,
        // group=1 case so we can read off out_channels directly.
        let in_features = 1;
        let out_features = 8;
        let group_size = 1;
        let n_packed = out_features / 8;
        // Pack the 8 lanes with values 0..8 (low → high in the i32).
        let qweight = vec![pack8([0, 1, 2, 3, 4, 5, 6, 7]); in_features * n_packed];
        // zero = 0 in every lane.
        let qzeros = vec![pack8([0; 8]); 1 * n_packed];
        let scales = vec![1.0f32; 1 * out_features];

        let packed = AwqQ4 {
            qweight,
            qzeros,
            scales,
            in_features,
            out_features,
            group_size,
        };
        let out = dequantize_awq_q4(&packed).unwrap();
        // out[n, 0] = q-from-lane-AWQ_PACK_ORDER[shuffle_idx_for_n]
        // shuffle index 0 → lane 0 → n=0 gets q=0
        // shuffle index 1 → lane 4 → n=1 gets q=4
        // shuffle index 2 → lane 1 → n=2 gets q=1
        // shuffle index 3 → lane 5 → n=3 gets q=5
        // shuffle index 4 → lane 2 → n=4 gets q=2
        // shuffle index 5 → lane 6 → n=5 gets q=6
        // shuffle index 6 → lane 3 → n=6 gets q=3
        // shuffle index 7 → lane 7 → n=7 gets q=7
        let expected = [0.0, 4.0, 1.0, 5.0, 2.0, 6.0, 3.0, 7.0];
        for (n, want) in expected.iter().enumerate() {
            assert!(
                (out[n * in_features] - want).abs() < 1e-6,
                "AWQ unpack n={n}: got {}, want {}",
                out[n * in_features],
                want
            );
        }
    }

    #[test]
    fn awq_q4_rejects_non_multiple_of_8_out_features() {
        let p = AwqQ4 {
            qweight: vec![0i32; 1],
            qzeros: vec![0i32; 1],
            scales: vec![0.0f32; 1],
            in_features: 8,
            out_features: 5,
            group_size: 8,
        };
        assert!(matches!(
            dequantize_awq_q4(&p).unwrap_err(),
            FerrotorchError::InvalidArgument { .. }
        ));
    }

    #[test]
    fn awq_q4_rejects_misaligned_group() {
        let p = AwqQ4 {
            qweight: vec![0i32; 16],
            qzeros: vec![0i32; 8],
            scales: vec![0.0f32; 8],
            in_features: 16,
            out_features: 8,
            group_size: 7,
        };
        assert!(matches!(
            dequantize_awq_q4(&p).unwrap_err(),
            FerrotorchError::InvalidArgument { .. }
        ));
    }
}
