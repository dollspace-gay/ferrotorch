//! Vision datasets: MNIST, CIFAR-10, CIFAR-100.
//!
//! Each dataset implements [`ferrotorch_data::Dataset`] and provides a
//! `synthetic()` constructor for pipeline testing without real data files.

pub mod cifar;
pub mod mnist;

pub use cifar::{Cifar10, Cifar100, CifarSample};
pub use mnist::{Mnist, MnistSample, Split};
