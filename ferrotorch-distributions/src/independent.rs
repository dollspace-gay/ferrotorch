//! Independent distribution wrapper.
//!
//! Reinterprets the rightmost `reinterpreted_batch_ndims` of a base
//! distribution's batch dimensions as event dimensions. The semantics:
//!
//! - `sample` / `rsample`: identical to the base distribution — same shape,
//!   same values.
//! - `log_prob`: the base log_prob is summed over the reinterpreted dims.
//!   This is the natural log_prob of a multivariate distribution formed
//!   by treating the reinterpreted dims as independent.
//! - `entropy`: similarly summed over the reinterpreted dims.
//!
//! Mirrors `torch.distributions.Independent`.
//!
//! # Why
//!
//! `Independent` is the standard way to turn a `Normal(loc=[B,K], scale=[B,K])`
//! (which yields a `[B,K]`-shaped log_prob) into a multivariate-style
//! distribution whose log_prob has shape `[B]`. It is also a building
//! block for variational autoencoders where the latent distribution is a
//! diagonal Gaussian over the K latent dims.

use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::tensor::Tensor;

use crate::Distribution;

/// Wraps a base distribution and reinterprets the rightmost
/// `reinterpreted_batch_ndims` of its batch shape as event dimensions.
pub struct Independent<T: Float, D: Distribution<T>> {
    base: D,
    reinterpreted_batch_ndims: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float, D: Distribution<T>> Independent<T, D> {
    /// Wrap a base distribution, treating the rightmost `n` batch dims
    /// as event dims.
    ///
    /// # Errors
    ///
    /// Returns an error if `reinterpreted_batch_ndims == 0` (in which
    /// case there is nothing to reinterpret — use the base directly).
    pub fn new(base: D, reinterpreted_batch_ndims: usize) -> FerrotorchResult<Self> {
        if reinterpreted_batch_ndims == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message:
                    "Independent: reinterpreted_batch_ndims must be > 0; use the base distribution directly"
                        .into(),
            });
        }
        Ok(Self {
            base,
            reinterpreted_batch_ndims,
            _phantom: std::marker::PhantomData,
        })
    }

    /// The wrapped base distribution.
    pub fn base(&self) -> &D {
        &self.base
    }

    /// The number of batch dims being reinterpreted as event dims.
    pub fn reinterpreted_batch_ndims(&self) -> usize {
        self.reinterpreted_batch_ndims
    }
}

impl<T: Float, D: Distribution<T>> Distribution<T> for Independent<T, D> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        // Independent does not change the shape of samples — only how the
        // tail dimensions are interpreted for log_prob/entropy.
        self.base.sample(shape)
    }

    fn rsample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        self.base.rsample(shape)
    }

    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let base_lp = self.base.log_prob(value)?;
        sum_rightmost(&base_lp, self.reinterpreted_batch_ndims)
    }

    fn entropy(&self) -> FerrotorchResult<Tensor<T>> {
        let base_h = self.base.entropy()?;
        sum_rightmost(&base_h, self.reinterpreted_batch_ndims)
    }
}

/// Sum a tensor along its rightmost `n` dims, returning a tensor whose
/// shape has `n` fewer dims. Stays on the input device.
fn sum_rightmost<T: Float>(t: &Tensor<T>, n: usize) -> FerrotorchResult<Tensor<T>> {
    let shape = t.shape();
    if n > shape.len() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "Independent: cannot sum {} rightmost dims of a {}-D tensor",
                n,
                shape.len()
            ),
        });
    }
    if n == 0 {
        return Ok(t.clone());
    }
    // Reduce along each rightmost dim from the right; sum_dim removes the
    // dim when keepdim=false. We start from the rightmost so dim indices
    // remain valid.
    let mut out = t.clone();
    for _ in 0..n {
        let last_dim = (out.ndim() - 1) as i64;
        out = ferrotorch_core::grad_fns::reduction::sum_dim(&out, last_dim, false)?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Normal;
    use ferrotorch_core::storage::TensorStorage;

    fn cpu_tensor(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    #[test]
    fn test_independent_zero_ndims_errors() {
        let loc = cpu_tensor(&[0.0, 0.0], &[2]);
        let scale = cpu_tensor(&[1.0, 1.0], &[2]);
        let n = Normal::new(loc, scale).unwrap();
        assert!(Independent::new(n, 0).is_err());
    }

    #[test]
    fn test_independent_log_prob_sums_event_dims() {
        // 2-D Normal: loc, scale shape [2]. log_prob of a 2-element value
        // is shape [2] for the base, shape [] (scalar) when wrapped in
        // Independent(reinterpreted_batch_ndims=1).
        let loc = cpu_tensor(&[0.0, 0.0], &[2]);
        let scale = cpu_tensor(&[1.0, 1.0], &[2]);
        let normal = Normal::new(loc.clone(), scale.clone()).unwrap();
        let value = cpu_tensor(&[0.5, -0.3], &[2]);
        let base_lp = normal.log_prob(&value).unwrap();
        assert_eq!(base_lp.shape(), &[2]);
        let base_data = base_lp.data().unwrap();
        let expected_sum = base_data[0] + base_data[1];

        let normal2 = Normal::new(loc, scale).unwrap();
        let ind = Independent::new(normal2, 1).unwrap();
        let ind_lp = ind.log_prob(&value).unwrap();
        // After summing 1 rightmost dim, the [2] -> [] (scalar).
        assert_eq!(ind_lp.shape(), [] as [usize; 0]);
        let val = ind_lp.item().unwrap();
        assert!(
            (val - expected_sum).abs() < 1e-5,
            "expected {expected_sum}, got {val}"
        );
    }

    #[test]
    fn test_independent_entropy_sums_event_dims() {
        let loc = cpu_tensor(&[0.0, 0.0, 0.0], &[3]);
        let scale = cpu_tensor(&[1.0, 2.0, 0.5], &[3]);
        let base_normal = Normal::new(loc.clone(), scale.clone()).unwrap();
        let base_h = base_normal.entropy().unwrap();
        let base_h_data = base_h.data().unwrap();
        let expected_sum: f32 = base_h_data.iter().sum();

        let normal2 = Normal::new(loc, scale).unwrap();
        let ind = Independent::new(normal2, 1).unwrap();
        let ind_h = ind.entropy().unwrap();
        assert_eq!(ind_h.shape(), [] as [usize; 0]);
        let val = ind_h.item().unwrap();
        assert!(
            (val - expected_sum).abs() < 1e-5,
            "expected {expected_sum}, got {val}"
        );
    }

    #[test]
    fn test_independent_sample_shape_unchanged() {
        let loc = cpu_tensor(&[0.0, 0.0], &[2]);
        let scale = cpu_tensor(&[1.0, 1.0], &[2]);
        let normal = Normal::new(loc, scale).unwrap();
        let ind = Independent::new(normal, 1).unwrap();
        let s = ind.sample(&[5, 2]).unwrap();
        assert_eq!(s.shape(), &[5, 2]);
    }
}
