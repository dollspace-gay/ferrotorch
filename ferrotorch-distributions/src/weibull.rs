//! Weibull distribution.
//!
//! `Weibull(scale, concentration)` — a two-parameter continuous distribution
//! commonly used in reliability engineering and survival analysis.

use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

use crate::Distribution;

/// Weibull distribution parameterized by `scale` (lambda) and
/// `concentration` (k, also called shape parameter).
///
/// PDF: `f(x) = (k/lambda) * (x/lambda)^(k-1) * exp(-(x/lambda)^k)` for x >= 0.
///
/// Sampling uses inverse CDF: `x = scale * (-log(1 - u))^(1/concentration)`.
pub struct Weibull<T: Float> {
    scale: Tensor<T>,
    concentration: Tensor<T>,
}

impl<T: Float> Weibull<T> {
    pub fn new(scale: Tensor<T>, concentration: Tensor<T>) -> FerrotorchResult<Self> {
        if scale.shape() != concentration.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Weibull: scale shape {:?} != concentration shape {:?}",
                    scale.shape(), concentration.shape()
                ),
            });
        }
        Ok(Self { scale, concentration })
    }

    pub fn scale(&self) -> &Tensor<T> { &self.scale }
    pub fn concentration(&self) -> &Tensor<T> { &self.concentration }
}

impl<T: Float> Distribution<T> for Weibull<T> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        let u = creation::rand::<T>(shape)?;
        let u_data = u.data()?;
        let s_data = self.scale.data()?;
        let k_data = self.concentration.data()?;
        let numel = u_data.len();
        let one = <T as num_traits::One>::one();

        let mut out = Vec::with_capacity(numel);
        for i in 0..numel {
            let si = if s_data.len() == 1 { 0 } else { i % s_data.len() };
            let ki = if k_data.len() == 1 { 0 } else { i % k_data.len() };
            // x = scale * (-log(1-u))^(1/k)
            let log_term = (one - u_data[i]).max(T::from(1e-30).unwrap()).ln();
            let val = s_data[si] * (<T as num_traits::Zero>::zero() - log_term).powf(one / k_data[ki]);
            out.push(val);
        }

        Tensor::from_storage(TensorStorage::cpu(out), shape.to_vec(), false)
    }

    fn rsample(&self, _shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message: "Weibull: rsample not yet implemented (requires inverse CDF backward)".into(),
        })
    }

    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let v = value.data()?;
        let s = self.scale.data()?;
        let k = self.concentration.data()?;
        let numel = v.len();
        let zero = <T as num_traits::Zero>::zero();

        let mut out = Vec::with_capacity(numel);
        for i in 0..numel {
            let si = if s.len() == 1 { 0 } else { i % s.len() };
            let ki = if k.len() == 1 { 0 } else { i % k.len() };
            if v[i] < zero {
                out.push(T::neg_infinity());
            } else {
                // log_prob = log(k/lambda) + (k-1)*log(x/lambda) - (x/lambda)^k
                let x_over_l = v[i] / s[si];
                let lp = (k[ki] / s[si]).ln()
                    + (k[ki] - <T as num_traits::One>::one()) * x_over_l.ln()
                    - x_over_l.powf(k[ki]);
                out.push(lp);
            }
        }

        Tensor::from_storage(TensorStorage::cpu(out), value.shape().to_vec(), false)
    }

    fn entropy(&self) -> FerrotorchResult<Tensor<T>> {
        // H = euler_gamma * (1 - 1/k) + log(lambda/k) + 1
        let s = self.scale.data()?;
        let k = self.concentration.data()?;
        let euler = T::from(0.5772156649015329).unwrap();
        let one = <T as num_traits::One>::one();

        let mut out = Vec::with_capacity(s.len());
        for i in 0..s.len() {
            let h = euler * (one - one / k[i]) + (s[i] / k[i]).ln() + one;
            out.push(h);
        }

        Tensor::from_storage(TensorStorage::cpu(out), self.scale.shape().to_vec(), false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar(v: f64) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(vec![v]), vec![1], false).unwrap()
    }

    #[test]
    fn test_weibull_sample_shape() {
        let d = Weibull::new(scalar(1.0), scalar(1.5)).unwrap();
        let s = d.sample(&[100]).unwrap();
        assert_eq!(s.shape(), &[100]);
        // All samples should be non-negative.
        for &v in s.data().unwrap() {
            assert!(v >= 0.0, "Weibull sample should be >= 0, got {v}");
        }
    }

    #[test]
    fn test_weibull_log_prob_negative() {
        let d = Weibull::new(scalar(1.0), scalar(2.0)).unwrap();
        let v = Tensor::from_storage(TensorStorage::cpu(vec![-1.0]), vec![1], false).unwrap();
        let lp = d.log_prob(&v).unwrap();
        assert!(lp.data().unwrap()[0].is_infinite() && lp.data().unwrap()[0] < 0.0);
    }

    #[test]
    fn test_weibull_entropy() {
        let d = Weibull::new(scalar(1.0), scalar(1.0)).unwrap();
        let h = d.entropy().unwrap();
        // For k=1, lambda=1: H = euler*(1-1/1) + ln(1/1) + 1 = 0 + 0 + 1 = 1.0
        assert!((h.data().unwrap()[0] - 1.0).abs() < 0.01);
    }
}
