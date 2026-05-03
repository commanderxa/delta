#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;

use crate::{Tensor, device::Device, op::Op, storage::Storage};

#[derive(Debug, PartialEq)]
/// # TensorImpl
///
/// ### This type holds the data of the tensor, and is an actual data.
///
/// `Tensor` holds a reference to it, allowing to use the same data.\
/// See the documentation for `Tensor`.
pub(crate) struct TensorImpl {
    // stored data
    // it is a single vector that is viewed regarding the shape
    pub data: Storage,
    // gradients
    pub grad: Option<Storage>,
    // vector of the tensors that were used to produce this tensor
    pub _prev: Vec<Tensor>,
    // operation that was used to produce this tensor
    pub _op: Option<Op>,
}

#[allow(dead_code)]
impl TensorImpl {
    /// Create a new instance of the TensorImpl.
    pub fn new(shape: &[usize]) -> Self {
        let len = shape.iter().product();
        Self {
            data: Storage::CPU {
                data: vec![0.0; len],
            },
            grad: Some(Storage::CPU {
                data: vec![0.0; len],
            }),
            _prev: vec![],
            _op: None,
        }
    }

    /// Sets gradients to 0
    pub fn zero_grad(&mut self) {
        let grad = vec![0.0; self.data.len()];
        self.grad = Some(Storage::CPU { data: grad });
    }

    /// Sets gradients to `None`
    pub fn grad_none(&mut self) {
        self.grad = None;
    }

    pub fn set_grad(&mut self, grad: Vec<f64>) {
        self.grad = Some(Storage::CPU { data: grad });
    }

    /// Creates a new instance of the TensorImpl from a Vector.
    pub fn from_f64(data: Vec<f64>) -> Self {
        let grad = vec![0.0; data.len()];
        // Self::fill_grad(&mut grad);
        Self {
            data: Storage::CPU { data: data },
            grad: Some(Storage::CPU { data: grad }),
            _prev: vec![],
            _op: None,
        }
    }

    /// Creates a new instance of the TensorImpl produced by any `Op`.
    pub fn from_op(data: Vec<f64>, prev: Vec<Tensor>, op: Op, device: Device) -> Self {
        let data = match device {
            Device::CPU => Storage::CPU { data: data },
            #[cfg(feature = "cuda")]
            Device::CUDA => Storage::CUDA {
                data: Storage::to_cuda_slice(&data),
            },
        };

        let _grad = vec![0.0; data.len()];
        let grad = match device {
            Device::CPU => Storage::CPU { data: _grad },
            #[cfg(feature = "cuda")]
            Device::CUDA => Storage::CUDA {
                data: Storage::to_cuda_slice(&_grad),
            },
        };

        Self {
            data: data,
            grad: Some(grad),
            _prev: prev,
            _op: Some(op),
        }
    }

    #[cfg(feature = "cuda")]
    pub fn from_cuda(data: CudaSlice<f64>, prev: Vec<Tensor>, op: Option<Op>) -> Self {
        let data = Storage::CUDA { data: data };

        let _grad = vec![0.0; data.len()];
        let grad = Storage::CUDA {
            data: Storage::to_cuda_slice(&_grad),
        };

        Self {
            data: data,
            grad: Some(grad),
            _prev: prev,
            _op: op,
        }
    }

    pub fn device(&self) -> Device {
        self.data.device()
    }
}
