//! Kumaraswamy distribution.
//!
//! `Kumaraswamy(a, b)` — a two-parameter distribution on [0, 1], similar to
//! Beta but with a closed-form CDF and simpler sampling.

use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

use crate::Distribution;

/// Kumaraswamy distribution parameterized by concentration parameters `a` > 0
/// and `b` > 0.
///
/// PDF: `f(x) = a * b * x^(a-1) * (1 - x^a)^(b-1)` for `x in [0, 1]`.
///
/// CDF: `F(x) = 1 - (1 - x^a)^b`.
///
/// Sampling via inverse CDF: `x = (1 - (1-u)^(1/b))^(1/a)`.
pub struct Kumaraswamy<T: Float> {
    a: Tensor<T>,
    b: Tensor<T>,
}

impl<T: Float> Kumaraswamy<T> {
    pub fn new(a: Tensor<T>, b: Tensor<T>) -> FerrotorchResult<Self> {
        if a.shape() != b.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Kumaraswamy: a shape {:?} != b shape {:?}",
                    a.shape(), b.shape()
                ),
            });
        }
        Ok(Self { a, b })
    }

    pub fn a(&self) -> &Tensor<T> { &self.a }
    pub fn b(&self) -> &Tensor<T> { &self.b }
}

impl<T: Float> Distribution<T> for Kumaraswamy<T> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        let u = creation::rand::<T>(shape)?;
        let u_data = u.data()?;
        let a_data = self.a.data()?;
        let b_data = self.b.data()?;
        let numel = u_data.len();
        let one = <T as num_traits::One>::one();

        let mut out = Vec::with_capacity(numel);
        for i in 0..numel {
            let ai = if a_data.len() == 1 { 0 } else { i % a_data.len() };
            let bi = if b_data.len() == 1 { 0 } else { i % b_data.len() };
            // x = (1 - (1-u)^(1/b))^(1/a)
            let inner = (one - u_data[i]).powf(one / b_data[bi]);
            let val = (one - inner).max(T::from(1e-30).unwrap()).powf(one / a_data[ai]);
            out.push(val);
        }

        Tensor::from_storage(TensorStorage::cpu(out), shape.to_vec(), false)
    }

    fn rsample(&self, _shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message: "Kumaraswamy: rsample not yet implemented".into(),
        })
    }

    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let v = value.data()?;
        let a = self.a.data()?;
        let b = self.b.data()?;
        let numel = v.len();
        let one = <T as num_traits::One>::one();
        let zero = <T as num_traits::Zero>::zero();

        let mut out = Vec::with_capacity(numel);
        for i in 0..numel {
            let ai = if a.len() == 1 { 0 } else { i % a.len() };
            let bi = if b.len() == 1 { 0 } else { i % b.len() };
            if v[i] <= zero || v[i] >= one {
                out.push(T::neg_infinity());
            } else {
                // log_prob = log(a) + log(b) + (a-1)*log(x) + (b-1)*log(1 - x^a)
                let lp = a[ai].ln() + b[bi].ln()
                    + (a[ai] - one) * v[i].ln()
                    + (b[bi] - one) * (one - v[i].powf(a[ai])).ln();
                out.push(lp);
            }
        }

        Tensor::from_storage(TensorStorage::cpu(out), value.shape().to_vec(), false)
    }

    fn entropy(&self) -> FerrotorchResult<Tensor<T>> {
        // H = (1 - 1/b) + (1 - 1/a) * H_b - log(a) - log(b)
        // where H_b = digamma(b+1) + euler_gamma (harmonic number approximation)
        // Simplified: H ≈ (1-1/a)*(digamma(b+1)+euler) + (1-1/b) - ln(a*b)
        let a = self.a.data()?;
        let b = self.b.data()?;
        let one = <T as num_traits::One>::one();
        let euler = T::from(0.5772156649015329).unwrap();

        let mut out = Vec::with_capacity(a.len());
        for i in 0..a.len() {
            // Approximate digamma(b+1) ≈ ln(b) + 1/(2*b) for large b
            let bf = num_traits::ToPrimitive::to_f64(&b[i]).unwrap();
            let digamma_b1 = T::from(if bf > 5.0 {
                bf.ln() + 0.5 / bf
            } else {
                // For small b, use the recurrence: digamma(x+1) = digamma(x) + 1/x
                // digamma(1) = -euler_gamma
                let mut val = -0.5772156649015329;
                let mut x = 1.0;
                while x < bf + 1.0 {
                    val += 1.0 / x;
                    x += 1.0;
                }
                val
            }).unwrap();

            let h = (one - one / a[i]) * (digamma_b1 + euler)
                + (one - one / b[i])
                - a[i].ln() - b[i].ln();
            out.push(h);
        }

        Tensor::from_storage(TensorStorage::cpu(out), self.a.shape().to_vec(), false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar(v: f64) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(vec![v]), vec![1], false).unwrap()
    }

    #[test]
    fn test_kumaraswamy_sample_range() {
        let d = Kumaraswamy::new(scalar(2.0), scalar(5.0)).unwrap();
        let s = d.sample(&[500]).unwrap();
        for &v in s.data().unwrap() {
            assert!(v > 0.0 && v < 1.0, "Kumaraswamy sample should be in (0,1), got {v}");
        }
    }

    #[test]
    fn test_kumaraswamy_log_prob_boundary() {
        let d = Kumaraswamy::new(scalar(1.0), scalar(1.0)).unwrap();
        let at_zero = Tensor::from_storage(TensorStorage::cpu(vec![0.0]), vec![1], false).unwrap();
        let at_one = Tensor::from_storage(TensorStorage::cpu(vec![1.0]), vec![1], false).unwrap();
        assert!(d.log_prob(&at_zero).unwrap().data().unwrap()[0].is_infinite());
        assert!(d.log_prob(&at_one).unwrap().data().unwrap()[0].is_infinite());
    }

    #[test]
    fn test_kumaraswamy_uniform_case() {
        // a=1, b=1 should be uniform — log_prob should be 0 everywhere in (0,1)
        let d = Kumaraswamy::new(scalar(1.0), scalar(1.0)).unwrap();
        let v = Tensor::from_storage(TensorStorage::cpu(vec![0.5]), vec![1], false).unwrap();
        let lp = d.log_prob(&v).unwrap().data().unwrap()[0];
        assert!((lp - 0.0).abs() < 1e-6, "a=1,b=1 should be uniform, log_prob={lp}");
    }
}
