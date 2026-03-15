//! Weight initialization functions.
//!
//! All functions operate on `Parameter<T>` in-place, matching PyTorch's
//! `nn.init` module. Each layer's constructor applies the appropriate
//! default initialization.

use ferrotorch_core::{Float, FerrotorchError, FerrotorchResult, Tensor, TensorStorage};

use crate::parameter::Parameter;

/// Non-linearity type for computing the correct gain in Kaiming init.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NonLinearity {
    Linear,
    Sigmoid,
    Tanh,
    ReLU,
    LeakyReLU(f64),
}

impl NonLinearity {
    /// Recommended gain for this non-linearity.
    pub fn gain(&self) -> f64 {
        match self {
            NonLinearity::Linear | NonLinearity::Sigmoid => 1.0,
            NonLinearity::Tanh => 5.0 / 3.0,
            NonLinearity::ReLU => (2.0f64).sqrt(),
            NonLinearity::LeakyReLU(neg_slope) => (2.0 / (1.0 + neg_slope * neg_slope)).sqrt(),
        }
    }
}

/// Compute fan_in and fan_out for a parameter tensor.
///
/// - 1D: fan_in = fan_out = shape[0]
/// - 2D: fan_in = shape[1], fan_out = shape[0]
/// - 3D+: fan_in = shape[1] * product(shape[2..]), fan_out = shape[0] * product(shape[2..])
fn compute_fans(shape: &[usize]) -> FerrotorchResult<(usize, usize)> {
    match shape.len() {
        0 => Err(FerrotorchError::InvalidArgument {
            message: "cannot compute fan for scalar tensor".into(),
        }),
        1 => Ok((shape[0], shape[0])),
        2 => Ok((shape[1], shape[0])),
        _ => {
            let receptive_field: usize = shape[2..].iter().product();
            Ok((shape[1] * receptive_field, shape[0] * receptive_field))
        }
    }
}

/// Fill parameter with a constant value.
pub fn constant<T: Float>(param: &mut Parameter<T>, value: T) -> FerrotorchResult<()> {
    let data = vec![value; param.numel()];
    *param = Parameter::new(Tensor::from_storage(
        TensorStorage::cpu(data),
        param.shape().to_vec(),
        true,
    )?);
    Ok(())
}

/// Fill parameter with zeros.
pub fn zeros<T: Float>(param: &mut Parameter<T>) -> FerrotorchResult<()> {
    constant(param, <T as num_traits::Zero>::zero())
}

/// Fill parameter with ones.
pub fn ones<T: Float>(param: &mut Parameter<T>) -> FerrotorchResult<()> {
    constant(param, <T as num_traits::One>::one())
}

/// Fill parameter with values from U(low, high).
pub fn uniform<T: Float>(param: &mut Parameter<T>, low: f64, high: f64) -> FerrotorchResult<()> {
    let numel = param.numel();
    let data: Vec<T> = simple_uniform(numel, low, high);
    *param = Parameter::new(Tensor::from_storage(
        TensorStorage::cpu(data),
        param.shape().to_vec(),
        true,
    )?);
    Ok(())
}

/// Fill parameter with values from N(mean, std).
pub fn normal<T: Float>(param: &mut Parameter<T>, mean: f64, std: f64) -> FerrotorchResult<()> {
    let numel = param.numel();
    let data: Vec<T> = simple_normal(numel, mean, std);
    *param = Parameter::new(Tensor::from_storage(
        TensorStorage::cpu(data),
        param.shape().to_vec(),
        true,
    )?);
    Ok(())
}

/// Xavier uniform initialization (Glorot).
///
/// Fills with values from U(-limit, limit) where limit = sqrt(6 / (fan_in + fan_out)).
pub fn xavier_uniform<T: Float>(param: &mut Parameter<T>) -> FerrotorchResult<()> {
    let (fan_in, fan_out) = compute_fans(param.shape())?;
    let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
    uniform(param, -limit, limit)
}

/// Xavier normal initialization (Glorot).
///
/// Fills with values from N(0, std) where std = sqrt(2 / (fan_in + fan_out)).
pub fn xavier_normal<T: Float>(param: &mut Parameter<T>) -> FerrotorchResult<()> {
    let (fan_in, fan_out) = compute_fans(param.shape())?;
    let std = (2.0 / (fan_in + fan_out) as f64).sqrt();
    normal(param, 0.0, std)
}

/// Kaiming uniform initialization (He).
///
/// Fills with values from U(-limit, limit) where limit = gain * sqrt(3 / fan_in).
pub fn kaiming_uniform<T: Float>(
    param: &mut Parameter<T>,
    nonlinearity: NonLinearity,
) -> FerrotorchResult<()> {
    let (fan_in, _) = compute_fans(param.shape())?;
    let gain = nonlinearity.gain();
    let std = gain / (fan_in as f64).sqrt();
    let limit = (3.0f64).sqrt() * std;
    uniform(param, -limit, limit)
}

/// Kaiming normal initialization (He).
///
/// Fills with values from N(0, std) where std = gain / sqrt(fan_in).
pub fn kaiming_normal<T: Float>(
    param: &mut Parameter<T>,
    nonlinearity: NonLinearity,
) -> FerrotorchResult<()> {
    let (fan_in, _) = compute_fans(param.shape())?;
    let gain = nonlinearity.gain();
    let std = gain / (fan_in as f64).sqrt();
    normal(param, 0.0, std)
}

// --- Internal PRNG helpers ---

fn simple_uniform<T: Float>(n: usize, low: f64, high: f64) -> Vec<T> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::SystemTime;

    let mut hasher = DefaultHasher::new();
    SystemTime::now().hash(&mut hasher);
    std::thread::current().id().hash(&mut hasher);
    let mut state = hasher.finish();
    if state == 0 {
        state = 0xdeadbeefcafe;
    }

    let range = high - low;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = (state as f64) / (u64::MAX as f64);
            T::from(low + u * range).unwrap()
        })
        .collect()
}

fn simple_normal<T: Float>(n: usize, mean: f64, std: f64) -> Vec<T> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::SystemTime;

    let mut hasher = DefaultHasher::new();
    SystemTime::now().hash(&mut hasher);
    std::thread::current().id().hash(&mut hasher);
    let mut state = hasher.finish();
    if state == 0 {
        state = 0xdeadbeefcafe;
    }

    let mut next_uniform = || -> f64 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state as f64) / (u64::MAX as f64)).max(1e-300)
    };

    let mut data = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        let u1 = next_uniform();
        let u2 = next_uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        data.push(T::from(mean + std * r * theta.cos()).unwrap());
        if i + 1 < n {
            data.push(T::from(mean + std * r * theta.sin()).unwrap());
        }
        i += 2;
    }
    data.truncate(n);
    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros_init() {
        let mut p = Parameter::<f32>::ones(&[3, 4]).unwrap();
        zeros(&mut p).unwrap();
        assert!(p.data().unwrap().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_ones_init() {
        let mut p = Parameter::<f32>::zeros(&[2, 3]).unwrap();
        ones(&mut p).unwrap();
        assert!(p.data().unwrap().iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_constant_init() {
        let mut p = Parameter::<f32>::zeros(&[5]).unwrap();
        constant(&mut p, 3.14).unwrap();
        assert!(p.data().unwrap().iter().all(|&x| (x - 3.14).abs() < 1e-5));
    }

    #[test]
    fn test_uniform_init_bounds() {
        let mut p = Parameter::<f32>::zeros(&[10000]).unwrap();
        uniform(&mut p, -1.0, 1.0).unwrap();
        let data = p.data().unwrap();
        assert!(data.iter().all(|&x| x >= -1.0 && x <= 1.0));
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert!(mean.abs() < 0.1);
    }

    #[test]
    fn test_normal_init_stats() {
        let mut p = Parameter::<f32>::zeros(&[10000]).unwrap();
        normal(&mut p, 0.0, 1.0).unwrap();
        let data = p.data().unwrap();
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        let var: f32 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        assert!(mean.abs() < 0.1, "mean = {mean}");
        assert!((var - 1.0).abs() < 0.2, "var = {var}");
    }

    #[test]
    fn test_xavier_uniform_stats() {
        let mut p = Parameter::<f32>::zeros(&[256, 128]).unwrap();
        xavier_uniform(&mut p).unwrap();
        let data = p.data().unwrap();
        let limit = (6.0 / (128.0 + 256.0) as f64).sqrt() as f32;
        assert!(data.iter().all(|&x| x.abs() <= limit + 0.01));
    }

    #[test]
    fn test_xavier_normal_stats() {
        let mut p = Parameter::<f32>::zeros(&[256, 128]).unwrap();
        xavier_normal(&mut p).unwrap();
        let data = p.data().unwrap();
        let expected_std = (2.0 / (128.0 + 256.0) as f64).sqrt() as f32;
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        let var: f32 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        assert!(mean.abs() < 0.05, "mean = {mean}");
        assert!(
            (var.sqrt() - expected_std).abs() < expected_std * 0.15,
            "std = {}, expected = {expected_std}",
            var.sqrt()
        );
    }

    #[test]
    fn test_kaiming_uniform_relu() {
        let mut p = Parameter::<f32>::zeros(&[64, 128]).unwrap();
        kaiming_uniform(&mut p, NonLinearity::ReLU).unwrap();
        let data = p.data().unwrap();
        let gain = (2.0f64).sqrt();
        let std = gain / (128.0f64).sqrt();
        let limit = (3.0f64).sqrt() * std;
        assert!(data.iter().all(|&x| (x as f64).abs() <= limit + 0.01));
    }

    #[test]
    fn test_kaiming_normal_relu() {
        let mut p = Parameter::<f32>::zeros(&[64, 128]).unwrap();
        kaiming_normal(&mut p, NonLinearity::ReLU).unwrap();
        let data = p.data().unwrap();
        let expected_std = (2.0f64).sqrt() / (128.0f64).sqrt();
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        let var: f32 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        assert!(mean.abs() < 0.1, "mean = {mean}");
        assert!(
            ((var.sqrt() as f64) - expected_std).abs() < expected_std * 0.2,
            "std = {}, expected = {expected_std}",
            var.sqrt()
        );
    }

    #[test]
    fn test_compute_fans_2d() {
        let (fi, fo) = compute_fans(&[64, 128]).unwrap();
        assert_eq!(fi, 128);
        assert_eq!(fo, 64);
    }

    #[test]
    fn test_compute_fans_4d() {
        let (fi, fo) = compute_fans(&[32, 16, 3, 3]).unwrap();
        assert_eq!(fi, 16 * 9);
        assert_eq!(fo, 32 * 9);
    }

    #[test]
    fn test_nonlinearity_gain() {
        assert!((NonLinearity::ReLU.gain() - (2.0f64).sqrt()).abs() < 1e-10);
        assert!((NonLinearity::Linear.gain() - 1.0).abs() < 1e-10);
        assert!((NonLinearity::Tanh.gain() - 5.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_init_preserves_requires_grad() {
        let mut p = Parameter::<f32>::zeros(&[5]).unwrap();
        xavier_uniform(&mut p).unwrap();
        assert!(p.requires_grad());
    }
}
