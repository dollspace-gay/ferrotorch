use crate::device::Device;
use crate::dtype::Element;

/// The underlying data buffer for a tensor, tagged with its device.
///
/// Owns the data directly (`Vec<T>` for CPU) rather than wrapping ferray's
/// `Array`. This gives freedom to add GPU backends as new `StorageBuffer`
/// variants without refactoring the `Tensor` type.
#[derive(Debug)]
pub struct TensorStorage<T: Element> {
    pub(crate) data: StorageBuffer<T>,
    pub(crate) device: Device,
}

/// Device-specific data buffer.
#[derive(Debug)]
pub enum StorageBuffer<T: Element> {
    /// CPU heap-allocated data.
    Cpu(Vec<T>),
    // Future: Cuda(CudaBuffer<T>),
    // Future: Metal(MetalBuffer<T>),
}

impl<T: Element> TensorStorage<T> {
    /// Create a new CPU storage from a `Vec<T>`.
    pub fn cpu(data: Vec<T>) -> Self {
        Self {
            data: StorageBuffer::Cpu(data),
            device: Device::Cpu,
        }
    }

    /// The device this storage resides on.
    #[inline]
    pub fn device(&self) -> Device {
        self.device
    }

    /// Total number of elements in the buffer.
    pub fn len(&self) -> usize {
        match &self.data {
            StorageBuffer::Cpu(v) => v.len(),
        }
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrow the data as a slice. Only available for CPU storage.
    pub fn as_slice(&self) -> &[T] {
        match &self.data {
            StorageBuffer::Cpu(v) => v.as_slice(),
        }
    }

    /// Borrow the data as a mutable slice. Only available for CPU storage.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        match &mut self.data {
            StorageBuffer::Cpu(v) => v.as_mut_slice(),
        }
    }
}

impl<T: Element> Clone for TensorStorage<T> {
    fn clone(&self) -> Self {
        Self {
            data: match &self.data {
                StorageBuffer::Cpu(v) => StorageBuffer::Cpu(v.clone()),
            },
            device: self.device,
        }
    }
}
