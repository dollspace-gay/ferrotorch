/// Re-export ferray's type system for tensor element types.
///
/// ferray's `Element` trait is sealed — only types ferray knows about can be
/// tensor elements. This module re-exports those types and adds convenience
/// traits for float-specific operations needed by the autograd engine.
pub use ferray_core::{DType, Element};

/// Marker trait for float element types that support autograd.
///
/// This is the bound used by `Tensor<T>` operations that require
/// differentiable arithmetic (add, mul, matmul, activations, etc.).
/// Integer and boolean tensors exist but cannot participate in gradient
/// computation.
pub trait Float: Element + num_traits::Float + std::ops::AddAssign {}

impl Float for f32 {}
impl Float for f64 {}
