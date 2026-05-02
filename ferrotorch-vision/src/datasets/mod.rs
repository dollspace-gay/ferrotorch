//! Vision datasets: MNIST, CIFAR-10, CIFAR-100, ImageFolder, DatasetFolder.
//!
//! Each dataset implements [`ferrotorch_data::Dataset`] and provides a
//! `synthetic()` or `from_dir()` constructor.

pub mod cifar;
pub mod folder;
pub mod mnist;

pub use cifar::{Cifar10, Cifar100, CifarSample};
pub use folder::{DatasetFolder, FolderSample, IMG_EXTENSIONS, ImageFolder, ImageSample};
pub use mnist::{Mnist, MnistSample, Split};
