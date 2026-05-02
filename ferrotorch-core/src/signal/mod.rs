//! Signal-processing utilities.
//!
//! Mirrors `torch.signal.*`. Currently exposes the [`windows`] submodule;
//! future work may add filter design, convolution helpers, and other
//! `scipy.signal`-shaped primitives.

pub mod windows;

pub use windows::{
    bartlett, blackman, cosine, exponential, gaussian, general_cosine, general_hamming, hamming,
    hann, hanning, kaiser, nuttall, parzen, taylor, tukey,
};
