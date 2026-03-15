pub mod autograd;
pub mod creation;
pub mod device;
mod display;
pub mod dtype;
pub mod error;
pub mod grad_fns;
pub mod ops;
mod inplace;
mod methods;
mod ops_trait;
pub mod quantize;
pub mod shape;
pub mod storage;
pub mod tensor;

// Public re-exports for ergonomic use.
pub use autograd::{autocast, autocast_dtype, is_autocast_enabled, AutocastDtype, backward, is_grad_enabled, no_grad};
pub use creation::{
    arange, eye, from_slice, from_vec, full, linspace, ones, rand, randn, scalar, tensor, zeros,
};
pub use device::Device;
pub use dtype::{DType, Element, Float};
pub use error::{FerrotorchError, FerrotorchResult};
pub use shape::{broadcast_shapes, normalize_axis};
pub use quantize::{
    dequantize, quantize, quantize_named_tensors, quantized_matmul, QuantDtype, QuantScheme,
    QuantizedTensor,
};
pub use storage::{StorageBuffer, TensorStorage};
pub use tensor::{GradFn, Tensor, TensorId};
