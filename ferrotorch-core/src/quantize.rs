//! Post-training quantization (PTQ) for ferrotorch tensors.
//!
//! Provides symmetric and asymmetric quantization to INT8, INT4, and UINT8,
//! with per-tensor or per-channel granularity. Designed for inference-time
//! model compression — quantize once after training, then run forward passes
//! with reduced memory and (on supported hardware) faster matmul.

use std::collections::HashMap;

use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Granularity of quantization parameters (scale / zero_point).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantScheme {
    /// One scale and zero_point for the entire tensor.
    PerTensor,
    /// One scale and zero_point per slice along the given axis.
    PerChannel(usize),
}

/// Target integer dtype for quantized storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantDtype {
    /// Signed 8-bit: [-128, 127].
    Int8,
    /// Signed 4-bit: [-8, 7].  Stored packed in `i8` values.
    Int4,
    /// Unsigned 8-bit: [0, 255].
    Uint8,
}

impl QuantDtype {
    /// Minimum representable value.
    #[inline]
    fn qmin(self) -> i32 {
        match self {
            QuantDtype::Int8 => -128,
            QuantDtype::Int4 => -8,
            QuantDtype::Uint8 => 0,
        }
    }

    /// Maximum representable value.
    #[inline]
    fn qmax(self) -> i32 {
        match self {
            QuantDtype::Int8 => 127,
            QuantDtype::Int4 => 7,
            QuantDtype::Uint8 => 255,
        }
    }
}

// ---------------------------------------------------------------------------
// QuantizedTensor
// ---------------------------------------------------------------------------

/// A tensor stored in quantized (integer) representation.
///
/// The real value is recovered by `x = (q - zero_point) * scale`.
///
/// `scale` and `zero_point` are vectors whose length equals:
/// * 1 for `PerTensor`
/// * `shape[axis]` for `PerChannel(axis)`
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Quantized values stored as `i8` regardless of logical dtype.
    /// For `Uint8`, the stored `i8` is reinterpreted as `u8` via
    /// wrapping cast; for `Int4` only the low 4 bits are significant.
    data: Vec<i8>,
    /// Per-tensor or per-channel scales.
    scale: Vec<f32>,
    /// Per-tensor or per-channel zero points (in quantized domain).
    zero_point: Vec<i32>,
    /// Original tensor shape.
    shape: Vec<usize>,
    /// Quantization granularity.
    scheme: QuantScheme,
    /// Target quantized dtype.
    dtype: QuantDtype,
}

impl QuantizedTensor {
    /// Number of elements.
    #[inline]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Borrow the shape.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Borrow the quantized data.
    #[inline]
    pub fn data(&self) -> &[i8] {
        &self.data
    }

    /// Borrow the scale vector.
    #[inline]
    pub fn scale(&self) -> &[f32] {
        &self.scale
    }

    /// Borrow the zero-point vector.
    #[inline]
    pub fn zero_point(&self) -> &[i32] {
        &self.zero_point
    }

    /// The quantization scheme used.
    #[inline]
    pub fn scheme(&self) -> QuantScheme {
        self.scheme
    }

    /// The quantized dtype.
    #[inline]
    pub fn qdtype(&self) -> QuantDtype {
        self.dtype
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute scale and zero_point for a given (min, max) range and target dtype.
///
/// Uses the standard asymmetric affine quantization formula:
///   scale = (max - min) / (qmax - qmin)
///   zero_point = round(qmin - min / scale)
///
/// The range is always expanded to include zero so that `0.0` maps exactly
/// to an integer quantized value (important for zero-padding and ReLU outputs).
/// When min == max the range would collapse to zero, so this expansion also
/// prevents division-by-zero.
fn compute_scale_zp(min_val: f32, max_val: f32, dtype: QuantDtype) -> (f32, i32) {
    let qmin = dtype.qmin();
    let qmax = dtype.qmax();

    // Ensure the range includes zero (standard PyTorch behaviour).
    let min_val = min_val.min(0.0);
    let max_val = max_val.max(0.0);

    // After including zero the range is at least max(|min|, |max|) > 0,
    // but guard against the degenerate all-zeros case.
    let range = (max_val - min_val).max(f32::EPSILON);
    let scale = range / (qmax - qmin) as f32;

    // zero_point is intentionally NOT clamped to [qmin, qmax]. It is stored
    // as i32 and may lie outside the quantized integer range. This is correct
    // for asymmetric affine quantization — clamping the zero_point distorts
    // the mapping when the float range doesn't straddle zero.
    let zp = (qmin as f32 - min_val / scale).round() as i32;

    (scale, zp)
}

/// Clamp and round a float to the quantized integer range.
///
/// Returns the result as `i8`. For `Uint8` the caller passes `qmin=0`,
/// `qmax=255`; the clamped i32 is cast to `u8` first then transmuted to `i8`
/// so that values 128..=255 are preserved through the bit pattern.
#[inline]
fn quantize_val(x: f32, scale: f32, zp: i32, qmin: i32, qmax: i32, is_unsigned: bool) -> i8 {
    let q = (x / scale + zp as f32).round() as i32;
    let clamped = q.clamp(qmin, qmax);
    if is_unsigned {
        (clamped as u8) as i8
    } else {
        clamped as i8
    }
}

/// Recover the i32 quantized value from the stored `i8`, accounting for
/// unsigned dtypes where the bit pattern represents a `u8`.
#[inline]
fn stored_to_i32(val: i8, is_unsigned: bool) -> i32 {
    if is_unsigned {
        (val as u8) as i32
    } else {
        val as i32
    }
}

/// Map a linear flat index to per-channel parameters.
///
/// For a tensor of shape `[d0, d1, ..., dn]` with channel axis `axis`,
/// returns the channel index for the element at `flat_index`.
#[inline]
fn channel_index(flat_index: usize, shape: &[usize], axis: usize) -> usize {
    // stride of the channel axis = product of dims after axis.
    let stride: usize = shape[axis + 1..].iter().product();
    (flat_index / stride) % shape[axis]
}

// ---------------------------------------------------------------------------
// Quantize
// ---------------------------------------------------------------------------

/// Quantize a floating-point tensor.
///
/// # Per-tensor
///
/// Computes a single (scale, zero_point) pair from the global min/max.
///
/// # Per-channel
///
/// Computes one (scale, zero_point) per slice along the given axis. This is
/// common for weight tensors where each output channel has its own range.
pub fn quantize<T: Float>(
    tensor: &Tensor<T>,
    scheme: QuantScheme,
    dtype: QuantDtype,
) -> FerrotorchResult<QuantizedTensor> {
    let data = tensor.data()?;
    let shape = tensor.shape().to_vec();
    let numel = tensor.numel();
    let qmin = dtype.qmin();
    let qmax = dtype.qmax();

    let is_unsigned = dtype == QuantDtype::Uint8;

    match scheme {
        QuantScheme::PerTensor => {
            // Global min/max.
            let mut min_val = f32::INFINITY;
            let mut max_val = f32::NEG_INFINITY;
            for &v in data {
                let f = v.to_f32().unwrap();
                if f < min_val {
                    min_val = f;
                }
                if f > max_val {
                    max_val = f;
                }
            }

            let (scale, zp) = compute_scale_zp(min_val, max_val, dtype);

            let qdata: Vec<i8> = data
                .iter()
                .map(|&v| {
                    quantize_val(v.to_f32().unwrap(), scale, zp, qmin, qmax, is_unsigned)
                })
                .collect();

            Ok(QuantizedTensor {
                data: qdata,
                scale: vec![scale],
                zero_point: vec![zp],
                shape,
                scheme,
                dtype,
            })
        }

        QuantScheme::PerChannel(axis) => {
            if axis >= shape.len() {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "PerChannel axis {axis} out of range for {}-d tensor",
                        shape.len()
                    ),
                });
            }

            let num_channels = shape[axis];
            let mut mins = vec![f32::INFINITY; num_channels];
            let mut maxs = vec![f32::NEG_INFINITY; num_channels];

            for (i, &v) in data.iter().enumerate() {
                let ch = channel_index(i, &shape, axis);
                let f = v.to_f32().unwrap();
                if f < mins[ch] {
                    mins[ch] = f;
                }
                if f > maxs[ch] {
                    maxs[ch] = f;
                }
            }

            let params: Vec<(f32, i32)> = mins
                .iter()
                .zip(maxs.iter())
                .map(|(&mn, &mx)| compute_scale_zp(mn, mx, dtype))
                .collect();

            let scales: Vec<f32> = params.iter().map(|&(s, _)| s).collect();
            let zps: Vec<i32> = params.iter().map(|&(_, z)| z).collect();

            let mut qdata = Vec::with_capacity(numel);
            for (i, &v) in data.iter().enumerate() {
                let ch = channel_index(i, &shape, axis);
                qdata.push(quantize_val(
                    v.to_f32().unwrap(),
                    scales[ch],
                    zps[ch],
                    qmin,
                    qmax,
                    is_unsigned,
                ));
            }

            Ok(QuantizedTensor {
                data: qdata,
                scale: scales,
                zero_point: zps,
                shape,
                scheme,
                dtype,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Dequantize
// ---------------------------------------------------------------------------

/// Dequantize back to a floating-point tensor.
///
/// Applies the inverse mapping: `x = (q - zero_point) * scale`.
pub fn dequantize<T: Float>(qtensor: &QuantizedTensor) -> FerrotorchResult<Tensor<T>> {
    let numel = qtensor.numel();
    let mut result = Vec::with_capacity(numel);
    let is_unsigned = qtensor.dtype == QuantDtype::Uint8;

    match qtensor.scheme {
        QuantScheme::PerTensor => {
            let scale = qtensor.scale[0];
            let zp = qtensor.zero_point[0];
            for &q in &qtensor.data {
                let val = (stored_to_i32(q, is_unsigned) - zp) as f32 * scale;
                result.push(T::from(val).unwrap());
            }
        }
        QuantScheme::PerChannel(axis) => {
            for (i, &q) in qtensor.data.iter().enumerate() {
                let ch = channel_index(i, &qtensor.shape, axis);
                let val = (stored_to_i32(q, is_unsigned) - qtensor.zero_point[ch]) as f32
                    * qtensor.scale[ch];
                result.push(T::from(val).unwrap());
            }
        }
    }

    Tensor::from_storage(TensorStorage::cpu(result), qtensor.shape.clone(), false)
}

// ---------------------------------------------------------------------------
// Quantized matmul
// ---------------------------------------------------------------------------

/// Multiply two quantized 2-D matrices and return a quantized result.
///
/// Strategy: accumulate in `i32` to avoid overflow, then rescale to the output
/// quantized domain. This avoids a full dequantize-matmul-requantize round-trip
/// while remaining numerically correct for INT8.
///
/// Both inputs must be 2-D, with compatible inner dimensions (standard matmul
/// rules: `[M, K] x [K, N] -> [M, N]`).
pub fn quantized_matmul(
    a: &QuantizedTensor,
    b: &QuantizedTensor,
) -> FerrotorchResult<QuantizedTensor> {
    // Validate shapes.
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "quantized_matmul requires 2-D tensors, got shapes {:?} and {:?}",
                a.shape, b.shape
            ),
        });
    }

    let m = a.shape[0];
    let k = a.shape[1];
    let k2 = b.shape[0];
    let n = b.shape[1];

    if k != k2 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "quantized_matmul inner dimensions mismatch: [{m}, {k}] x [{k2}, {n}]"
            ),
        });
    }

    // Both inputs must be PerTensor for the fast path.
    if a.scale.len() != 1 || b.scale.len() != 1 {
        return Err(FerrotorchError::InvalidArgument {
            message: "quantized_matmul currently requires PerTensor-quantized inputs".into(),
        });
    }

    let a_scale = a.scale[0];
    let a_zp = a.zero_point[0];
    let b_scale = b.scale[0];
    let b_zp = b.zero_point[0];

    let a_unsigned = a.dtype == QuantDtype::Uint8;
    let b_unsigned = b.dtype == QuantDtype::Uint8;

    // Accumulate in i32.
    let mut acc = vec![0i32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0i32;
            for p in 0..k {
                let qa = stored_to_i32(a.data[i * k + p], a_unsigned) - a_zp;
                let qb = stored_to_i32(b.data[p * n + j], b_unsigned) - b_zp;
                sum += qa * qb;
            }
            acc[i * n + j] = sum;
        }
    }

    // The real-valued result element is: acc[i,j] * a_scale * b_scale.
    // Requantize: pick INT8 output with its own scale/zp.
    let combined_scale = a_scale * b_scale;

    // Find the real-valued min/max of the output.
    let mut out_min = f32::INFINITY;
    let mut out_max = f32::NEG_INFINITY;
    for &a_val in &acc {
        let real = a_val as f32 * combined_scale;
        if real < out_min {
            out_min = real;
        }
        if real > out_max {
            out_max = real;
        }
    }

    let out_dtype = QuantDtype::Int8;
    let (out_scale, out_zp) = compute_scale_zp(out_min, out_max, out_dtype);
    let qmin = out_dtype.qmin();
    let qmax = out_dtype.qmax();

    let qdata: Vec<i8> = acc
        .iter()
        .map(|&a_val| {
            let real = a_val as f32 * combined_scale;
            quantize_val(real, out_scale, out_zp, qmin, qmax, false)
        })
        .collect();

    Ok(QuantizedTensor {
        data: qdata,
        scale: vec![out_scale],
        zero_point: vec![out_zp],
        shape: vec![m, n],
        scheme: QuantScheme::PerTensor,
        dtype: out_dtype,
    })
}

// ---------------------------------------------------------------------------
// Module-level quantization utility
// ---------------------------------------------------------------------------

/// Quantize every weight tensor in a module, returning a name -> QuantizedTensor
/// map suitable for serialization or quantized inference.
///
/// This accepts any type implementing the `Module` trait from `ferrotorch-nn`.
/// Because `ferrotorch-core` does not depend on `ferrotorch-nn`, we accept a
/// generic iterator of named tensors instead.
pub fn quantize_named_tensors<T: Float>(
    named_tensors: impl IntoIterator<Item = (String, Tensor<T>)>,
    scheme: QuantScheme,
    dtype: QuantDtype,
) -> FerrotorchResult<HashMap<String, QuantizedTensor>> {
    let mut result = HashMap::new();
    for (name, tensor) in named_tensors {
        let qtensor = quantize(&tensor, scheme, dtype)?;
        result.insert(name, qtensor);
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a tensor from f32 data.
    fn make_tensor(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        crate::from_slice(data, shape).unwrap()
    }

    // ----- Round-trip quantize/dequantize -----

    #[test]
    fn test_per_tensor_int8_roundtrip() {
        let data: Vec<f32> = (-10..=10).map(|x| x as f32 * 0.5).collect();
        let t = make_tensor(&data, &[data.len()]);
        let qt = quantize(&t, QuantScheme::PerTensor, QuantDtype::Int8).unwrap();
        let rt: Tensor<f32> = dequantize(&qt).unwrap();

        assert_eq!(rt.shape(), t.shape());
        let orig = t.data().unwrap();
        let recovered = rt.data().unwrap();
        for (i, (&o, &r)) in orig.iter().zip(recovered.iter()).enumerate() {
            let err = (o - r).abs();
            // INT8 over [-5, 5]: step ≈ 10/255 ≈ 0.04, max error ≈ half step ≈ 0.02
            assert!(
                err < 0.05,
                "element {i}: original={o}, recovered={r}, error={err}"
            );
        }
    }

    #[test]
    fn test_per_tensor_uint8_roundtrip() {
        let data: Vec<f32> = (0..=20).map(|x| x as f32 * 0.1).collect();
        let t = make_tensor(&data, &[data.len()]);
        let qt = quantize(&t, QuantScheme::PerTensor, QuantDtype::Uint8).unwrap();
        let rt: Tensor<f32> = dequantize(&qt).unwrap();

        let orig = t.data().unwrap();
        let recovered = rt.data().unwrap();
        for (i, (&o, &r)) in orig.iter().zip(recovered.iter()).enumerate() {
            let err = (o - r).abs();
            // UINT8 over [0, 2]: step ≈ 2/255 ≈ 0.008
            assert!(
                err < 0.02,
                "element {i}: original={o}, recovered={r}, error={err}"
            );
        }
    }

    #[test]
    fn test_per_tensor_int4_roundtrip() {
        // INT4 has only 16 levels, so larger quantization error is expected.
        let data: Vec<f32> = (-8..=7).map(|x| x as f32).collect();
        let t = make_tensor(&data, &[data.len()]);
        let qt = quantize(&t, QuantScheme::PerTensor, QuantDtype::Int4).unwrap();
        let rt: Tensor<f32> = dequantize(&qt).unwrap();

        let orig = t.data().unwrap();
        let recovered = rt.data().unwrap();
        for (i, (&o, &r)) in orig.iter().zip(recovered.iter()).enumerate() {
            let err = (o - r).abs();
            // INT4 over [-8, 7]: step = 15/15 = 1.0, max error ≈ 0.5
            assert!(
                err < 1.01,
                "element {i}: original={o}, recovered={r}, error={err}"
            );
        }
    }

    // ----- Per-channel -----

    #[test]
    fn test_per_channel_int8_roundtrip() {
        // Shape [3, 4]: 3 channels along axis 0, each with different ranges.
        #[rustfmt::skip]
        let data: Vec<f32> = vec![
            // channel 0: range [0, 3]
            0.0, 1.0, 2.0, 3.0,
            // channel 1: range [-10, 10]
            -10.0, -5.0, 5.0, 10.0,
            // channel 2: range [100, 200]
            100.0, 130.0, 170.0, 200.0,
        ];
        let t = make_tensor(&data, &[3, 4]);
        let qt = quantize(&t, QuantScheme::PerChannel(0), QuantDtype::Int8).unwrap();
        let rt: Tensor<f32> = dequantize(&qt).unwrap();

        assert_eq!(qt.scale.len(), 3);
        assert_eq!(qt.zero_point.len(), 3);

        let orig = t.data().unwrap();
        let recovered = rt.data().unwrap();
        for (i, (&o, &r)) in orig.iter().zip(recovered.iter()).enumerate() {
            let err = (o - r).abs();
            // Each channel has its own scale, so error is relative to the
            // channel's range. Worst case channel 2: 100/255 ≈ 0.39.
            assert!(
                err < 0.5,
                "element {i}: original={o}, recovered={r}, error={err}"
            );
        }
    }

    #[test]
    fn test_per_channel_axis_out_of_bounds() {
        let t = make_tensor(&[1.0, 2.0, 3.0], &[3]);
        let result = quantize(&t, QuantScheme::PerChannel(5), QuantDtype::Int8);
        assert!(result.is_err());
    }

    // ----- Quantized matmul -----

    #[test]
    fn test_quantized_matmul_identity() {
        // A * I should ≈ A after quantize -> matmul -> dequantize.
        let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let a = make_tensor(&a_data, &[2, 2]);
        let eye = make_tensor(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);

        let qa = quantize(&a, QuantScheme::PerTensor, QuantDtype::Int8).unwrap();
        let qi = quantize(&eye, QuantScheme::PerTensor, QuantDtype::Int8).unwrap();
        let qc = quantized_matmul(&qa, &qi).unwrap();
        let c: Tensor<f32> = dequantize(&qc).unwrap();

        assert_eq!(c.shape(), &[2, 2]);
        let c_data = c.data().unwrap();
        for (i, (&expected, &got)) in a_data.iter().zip(c_data.iter()).enumerate() {
            let err = (expected - got).abs();
            assert!(
                err < 0.5,
                "element {i}: expected={expected}, got={got}, error={err}"
            );
        }
    }

    #[test]
    fn test_quantized_matmul_correctness() {
        // [2,3] x [3,2] -> [2,2]
        // A = [[1, 2, 3],
        //      [4, 5, 6]]
        // B = [[7,  8],
        //      [9, 10],
        //      [11, 12]]
        // A @ B = [[ 58,  64],
        //          [139, 154]]
        let a = make_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = make_tensor(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);

        let qa = quantize(&a, QuantScheme::PerTensor, QuantDtype::Int8).unwrap();
        let qb = quantize(&b, QuantScheme::PerTensor, QuantDtype::Int8).unwrap();
        let qc = quantized_matmul(&qa, &qb).unwrap();
        let c: Tensor<f32> = dequantize(&qc).unwrap();

        let expected = [58.0f32, 64.0, 139.0, 154.0];
        let c_data = c.data().unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        for (i, (&e, &g)) in expected.iter().zip(c_data.iter()).enumerate() {
            let err = (e - g).abs();
            // Quantization introduces some error; for small integers in INT8
            // the error should be small relative to the values.
            assert!(
                err < 3.0,
                "element {i}: expected={e}, got={g}, error={err}"
            );
        }
    }

    #[test]
    fn test_quantized_matmul_shape_mismatch() {
        let a = make_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let qa = quantize(&a, QuantScheme::PerTensor, QuantDtype::Int8).unwrap();
        let qb = quantize(&b, QuantScheme::PerTensor, QuantDtype::Int8).unwrap();
        let result = quantized_matmul(&qa, &qb);
        assert!(result.is_err());
    }

    #[test]
    fn test_quantized_matmul_non_2d() {
        let a = make_tensor(&[1.0, 2.0, 3.0], &[3]);
        let b = make_tensor(&[4.0, 5.0, 6.0], &[3]);

        let qa = quantize(&a, QuantScheme::PerTensor, QuantDtype::Int8).unwrap();
        let qb = quantize(&b, QuantScheme::PerTensor, QuantDtype::Int8).unwrap();
        let result = quantized_matmul(&qa, &qb);
        assert!(result.is_err());
    }

    // ----- Module quantization utility -----

    #[test]
    fn test_quantize_named_tensors() {
        let w1 = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let w2 = make_tensor(&[-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], &[3, 2]);

        let named = vec![
            ("layer.weight".to_string(), w1),
            ("layer2.weight".to_string(), w2),
        ];

        let qmap =
            quantize_named_tensors(named, QuantScheme::PerTensor, QuantDtype::Int8).unwrap();

        assert_eq!(qmap.len(), 2);
        assert!(qmap.contains_key("layer.weight"));
        assert!(qmap.contains_key("layer2.weight"));
        assert_eq!(qmap["layer.weight"].shape(), &[2, 2]);
        assert_eq!(qmap["layer2.weight"].shape(), &[3, 2]);
    }

    // ----- Constant values / edge cases -----

    #[test]
    fn test_quantize_constant_tensor() {
        // All values identical — scale should not be zero.
        let t = make_tensor(&[5.0, 5.0, 5.0, 5.0], &[4]);
        let qt = quantize(&t, QuantScheme::PerTensor, QuantDtype::Int8).unwrap();
        let rt: Tensor<f32> = dequantize(&qt).unwrap();

        let recovered = rt.data().unwrap();
        for &r in recovered {
            assert!(
                (r - 5.0).abs() < 0.1,
                "constant tensor dequantized to {r}, expected 5.0"
            );
        }
    }

    #[test]
    fn test_quantize_single_element() {
        let t = make_tensor(&[42.0], &[1]);
        let qt = quantize(&t, QuantScheme::PerTensor, QuantDtype::Int8).unwrap();
        let rt: Tensor<f32> = dequantize(&qt).unwrap();
        assert!((rt.data().unwrap()[0] - 42.0).abs() < 0.5);
    }

    #[test]
    fn test_per_channel_int4() {
        // 2 channels, 3 elements each.
        let data = vec![0.0, 1.0, 2.0, -4.0, 0.0, 4.0];
        let t = make_tensor(&data, &[2, 3]);
        let qt = quantize(&t, QuantScheme::PerChannel(0), QuantDtype::Int4).unwrap();

        assert_eq!(qt.scale.len(), 2);
        assert_eq!(qt.zero_point.len(), 2);

        let rt: Tensor<f32> = dequantize(&qt).unwrap();
        let orig = t.data().unwrap();
        let recovered = rt.data().unwrap();
        for (i, (&o, &r)) in orig.iter().zip(recovered.iter()).enumerate() {
            let err = (o - r).abs();
            // INT4 has coarse resolution, but channel-level ranges are small.
            assert!(
                err < 1.0,
                "element {i}: original={o}, recovered={r}, error={err}"
            );
        }
    }

    #[test]
    fn test_dequantize_f64() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let t = crate::from_slice(&data, &[4]).unwrap();
        let qt = quantize(&t, QuantScheme::PerTensor, QuantDtype::Int8).unwrap();
        let rt: Tensor<f64> = dequantize(&qt).unwrap();

        assert_eq!(rt.shape(), &[4]);
        let recovered = rt.data().unwrap();
        for (i, &r) in recovered.iter().enumerate() {
            let expected = data[i] as f64;
            let err = (expected - r).abs();
            assert!(
                err < 0.05,
                "element {i}: expected={expected}, recovered={r}, error={err}"
            );
        }
    }

    #[test]
    fn test_quantized_tensor_accessors() {
        let t = make_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let qt = quantize(&t, QuantScheme::PerTensor, QuantDtype::Int8).unwrap();

        assert_eq!(qt.numel(), 6);
        assert_eq!(qt.shape(), &[2, 3]);
        assert_eq!(qt.data().len(), 6);
        assert_eq!(qt.scale().len(), 1);
        assert_eq!(qt.zero_point().len(), 1);
        assert_eq!(qt.scheme(), QuantScheme::PerTensor);
        assert_eq!(qt.qdtype(), QuantDtype::Int8);
    }
}
