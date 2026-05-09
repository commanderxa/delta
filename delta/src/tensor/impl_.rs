#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;

use crate::{DType, Storage, Tensor, device::Device, op::Op, tensor::element::TensorElement};

#[derive(Debug)]
/// # TensorImpl
///
/// ### This type holds the data of the tensor, and is an actual data.
///
/// `Tensor` holds a reference to it, allowing to use the same data.\
/// See the documentation for `Tensor`.
pub(crate) struct TensorImpl<T: TensorElement> {
    // stored data
    // it is a single vector that is viewed regarding the shape
    pub data: Storage<T>,
    // gradients
    pub grad: Option<Tensor<T>>,
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
    // vector of the tensors that were used to produce this tensor
    pub _prev: Vec<Tensor<T>>,
    // operation that was used to produce this tensor
    pub _op: Option<Op<T>>,
}

#[allow(dead_code)]
impl<T: TensorElement> TensorImpl<T> {
    /// Create a new instance of the TensorImpl.
    pub fn new(shape: &[usize]) -> Self {
        let len = shape.iter().product();
        Self {
            data: Storage::CPU(vec![T::zero(); len]),
            grad: Some(crate::tensor(&vec![T::zero(); len], shape)),
            shape: shape.to_vec(),
            stride: Self::compute_stride(shape),
            _prev: vec![],
            _op: None,
        }
    }

    pub(crate) fn compute_stride(shape: &[usize]) -> Vec<usize> {
        let mut stride = vec![1; shape.len()];
        // compute stride
        for i in (0..shape.len() - 1).rev() {
            stride[i] = shape[i + 1] * stride[i + 1];
        }
        stride
    }

    /// Sets gradients to 0
    pub fn zero_grad(&mut self) {
        let len = self.shape.iter().product();
        self.grad = Some(crate::tensor(&vec![T::zero(); len], &self.shape));
    }

    /// Sets gradients to `None`
    pub fn grad_none(&mut self) {
        self.grad = None;
    }

    pub fn set_grad(&mut self, grad: Vec<T>) {
        self.grad = Some(crate::tensor(&grad, &self.shape));
    }

    /// Creates a new instance of the TensorImpl from a Vector.
    pub fn from_slice(data: &[T], shape: &[usize]) -> Self
    where
        T: TensorElement,
    {
        Self {
            data: Storage::from_slice(data, Device::CPU),
            grad: None,
            shape: shape.to_vec(),
            stride: Self::compute_stride(shape),
            _prev: vec![],
            _op: None,
        }
    }

    /// Creates a new instance of the TensorImpl produced by any `Op`.
    pub fn from_op(
        data: Vec<T>,
        shape: &[usize],
        prev: Vec<Tensor<T>>,
        op: Op<T>,
        device: Device,
    ) -> Self {
        let data = match device {
            Device::CPU => Storage::CPU(data),
            #[cfg(feature = "cuda")]
            Device::CUDA => Storage::CUDA(Storage::to_cuda_slice(&data)),
        };

        let _grad = vec![T::zero(); data.len()];
        let grad = match device {
            Device::CPU => crate::create_grad(shape),
            #[cfg(feature = "cuda")]
            Device::CUDA => crate::create_grad(shape).cuda(),
        };

        Self {
            data: data,
            grad: Some(grad),
            shape: shape.to_vec(),
            stride: Self::compute_stride(shape),
            _prev: prev,
            _op: Some(op),
        }
    }

    #[cfg(feature = "cuda")]
    pub fn from_cuda(
        data: CudaSlice<T>,
        shape: &[usize],
        prev: Vec<Tensor<T>>,
        op: Option<Op<T>>,
    ) -> Self {
        let data = Storage::CUDA(data);

        let grad = crate::create_grad(shape).cuda();

        Self {
            data: data,
            grad: Some(grad),
            shape: shape.to_vec(),
            stride: Self::compute_stride(shape),
            _prev: prev,
            _op: op,
        }
    }

    pub fn device(&self) -> Device {
        self.data.device()
    }

    pub fn dtype(&self) -> DType {
        self.data.dtype()
    }
}

impl<T: TensorElement> PartialEq for TensorImpl<T> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data && self.shape == other.shape
    }
}
