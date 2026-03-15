//! Elementwise tensor operations.
//!
//! These operate on tensor data slices directly using `num_traits::Float`
//! methods. SIMD-accelerated paths via ferray-ufunc kernels can be added
//! as an optimization without changing the API.

use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::shape::broadcast_shapes;
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

/// Apply a unary function elementwise, producing a new tensor.
///
/// This is the generic workhorse for `exp`, `log`, `sin`, etc.
/// The `grad_fn` is attached by the calling grad_fn module, not here.
pub fn unary_map<T: Float>(input: &Tensor<T>, f: impl Fn(T) -> T) -> FerrotorchResult<Tensor<T>> {
    let data = input.data()?;
    let result: Vec<T> = data.iter().map(|&x| f(x)).collect();
    Tensor::from_storage(TensorStorage::cpu(result), input.shape().to_vec(), false)
}

/// Apply a binary function elementwise on two tensors with broadcasting.
pub fn binary_map<T: Float>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    f: impl Fn(T, T) -> T,
) -> FerrotorchResult<Tensor<T>> {
    // Same-shape fast path.
    if a.shape() == b.shape() {
        let a_data = a.data()?;
        let b_data = b.data()?;
        let result: Vec<T> = a_data
            .iter()
            .zip(b_data.iter())
            .map(|(&x, &y)| f(x, y))
            .collect();
        return Tensor::from_storage(TensorStorage::cpu(result), a.shape().to_vec(), false);
    }

    // Broadcasting path.
    let out_shape = broadcast_shapes(a.shape(), b.shape())?;
    let out_numel: usize = out_shape.iter().product();
    let mut result = Vec::with_capacity(out_numel);

    let a_data = a.data()?;
    let b_data = b.data()?;

    for i in 0..out_numel {
        let a_idx = broadcast_index(i, &out_shape, a.shape());
        let b_idx = broadcast_index(i, &out_shape, b.shape());
        result.push(f(a_data[a_idx], b_data[b_idx]));
    }

    Tensor::from_storage(TensorStorage::cpu(result), out_shape, false)
}

/// Apply a binary function between a tensor and a scalar.
pub fn scalar_map<T: Float>(
    input: &Tensor<T>,
    scalar: T,
    f: impl Fn(T, T) -> T,
) -> FerrotorchResult<Tensor<T>> {
    let data = input.data()?;
    let result: Vec<T> = data.iter().map(|&x| f(x, scalar)).collect();
    Tensor::from_storage(TensorStorage::cpu(result), input.shape().to_vec(), false)
}

/// Map a flat linear index in the output shape to a flat index in an input
/// shape, handling broadcasting (size-1 dimensions map to index 0).
fn broadcast_index(flat_idx: usize, out_shape: &[usize], in_shape: &[usize]) -> usize {
    let out_ndim = out_shape.len();
    let in_ndim = in_shape.len();
    let mut idx = 0;
    let mut in_stride = 1;
    let mut out_stride = 1;

    for i in 0..in_ndim {
        let out_axis = out_ndim - 1 - i;
        let in_axis = in_ndim - 1 - i;

        let out_dim = out_shape[out_axis];
        let in_dim = in_shape[in_axis];

        let out_coord = (flat_idx / out_stride) % out_dim;
        let in_coord = if in_dim == 1 { 0 } else { out_coord };

        idx += in_coord * in_stride;
        in_stride *= in_dim;
        out_stride *= out_dim;
    }

    idx
}

// --- Reduction operations ---

/// Sum all elements of a tensor, returning a scalar tensor.
pub fn sum<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let data = input.data()?;
    let total = data.iter().copied().fold(<T as num_traits::Zero>::zero(), |a, b| a + b);
    Tensor::from_storage(TensorStorage::cpu(vec![total]), vec![], false)
}

/// Sum along a given axis, reducing that dimension.
pub fn sum_axis<T: Float>(input: &Tensor<T>, axis: usize) -> FerrotorchResult<Tensor<T>> {
    let shape = input.shape();
    if axis >= shape.len() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("axis {} out of bounds for tensor with {} dims", axis, shape.len()),
        });
    }

    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape.remove(axis);

    let data = input.data()?;

    let out_numel: usize = out_shape.iter().product();
    let mut result = vec![<T as num_traits::Zero>::zero(); out_numel.max(1)];

    for i in 0..input.numel() {
        // Decompose flat index into per-axis coordinates.
        let mut coords = vec![0usize; shape.len()];
        let mut rem = i;
        for d in (0..shape.len()).rev() {
            coords[d] = rem % shape[d];
            rem /= shape[d];
        }
        // Compute output flat index by skipping the reduced axis.
        let mut oi = 0;
        let mut os = 1;
        for d in (0..shape.len()).rev() {
            if d != axis {
                oi += coords[d] * os;
                os *= shape[d];
            }
        }
        result[oi] = result[oi] + data[i];
    }

    if out_shape.is_empty() {
        // Reduced to scalar.
        Tensor::from_storage(TensorStorage::cpu(result), vec![], false)
    } else {
        Tensor::from_storage(TensorStorage::cpu(result), out_shape, false)
    }
}

/// Mean of all elements, returning a scalar.
pub fn mean<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let data = input.data()?;
    let n = T::from(data.len()).unwrap();
    let total = data.iter().copied().fold(<T as num_traits::Zero>::zero(), |a, b| a + b);
    Tensor::from_storage(TensorStorage::cpu(vec![total / n]), vec![], false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    #[test]
    fn test_unary_map() {
        let a = t(&[1.0, 4.0, 9.0], &[3]);
        let b = unary_map(&a, |x| x.sqrt()).unwrap();
        let d = b.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-6);
        assert!((d[1] - 2.0).abs() < 1e-6);
        assert!((d[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_binary_map_same_shape() {
        let a = t(&[1.0, 2.0, 3.0], &[3]);
        let b = t(&[4.0, 5.0, 6.0], &[3]);
        let c = binary_map(&a, &b, |x, y| x + y).unwrap();
        assert_eq!(c.data().unwrap(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_binary_map_broadcast() {
        let a = t(&[1.0, 2.0, 3.0], &[3]);
        let b = t(&[10.0], &[1]);
        let c = binary_map(&a, &b, |x, y| x + y).unwrap();
        assert_eq!(c.shape(), &[3]);
        assert_eq!(c.data().unwrap(), &[11.0, 12.0, 13.0]);
    }

    #[test]
    fn test_binary_map_broadcast_2d() {
        // [2,3] + [1,3] -> [2,3]
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = t(&[10.0, 20.0, 30.0], &[1, 3]);
        let c = binary_map(&a, &b, |x, y| x + y).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(c.data().unwrap(), &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn test_scalar_map() {
        let a = t(&[2.0, 4.0, 6.0], &[3]);
        let b = scalar_map(&a, 2.0, |x, s| x * s).unwrap();
        assert_eq!(b.data().unwrap(), &[4.0, 8.0, 12.0]);
    }

    #[test]
    fn test_sum() {
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let s = sum(&a).unwrap();
        assert!(s.is_scalar());
        assert!((s.item().unwrap() - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_sum_axis() {
        // [[1, 2, 3], [4, 5, 6]] sum along axis 0 -> [5, 7, 9]
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let s = sum_axis(&a, 0).unwrap();
        assert_eq!(s.shape(), &[3]);
        let d = s.data().unwrap();
        assert!((d[0] - 5.0).abs() < 1e-6);
        assert!((d[1] - 7.0).abs() < 1e-6);
        assert!((d[2] - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_sum_axis_1() {
        // [[1, 2, 3], [4, 5, 6]] sum along axis 1 -> [6, 15]
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let s = sum_axis(&a, 1).unwrap();
        assert_eq!(s.shape(), &[2]);
        let d = s.data().unwrap();
        assert!((d[0] - 6.0).abs() < 1e-6);
        assert!((d[1] - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_mean() {
        let a = t(&[2.0, 4.0, 6.0, 8.0], &[4]);
        let m = mean(&a).unwrap();
        assert!((m.item().unwrap() - 5.0).abs() < 1e-6);
    }
}
