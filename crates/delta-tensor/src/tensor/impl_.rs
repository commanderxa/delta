#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;
use half::{bf16, f16};

use crate::{DType, Op, Storage, Tensor, device::Device, f8, tensor::repr::TensorRepr};

#[derive(Clone, Debug)]
/// # TensorImpl
///
/// ### This type holds the data of the tensor, and is an actual data.
///
/// `Tensor` holds a reference to it, allowing to use the same data.\
/// See the documentation for `Tensor`.
pub struct TensorImpl {
    // stored data
    // it is a single vector that is viewed regarding the shape
    pub data: Storage,
    // gradients
    pub grad: Option<Tensor>,
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
    pub offset: usize,
    // vector of the tensors that were used to produce this tensor
    pub prev: Vec<Tensor>,
    // operation that was used to produce this tensor
    pub op: Option<Op>,
}

#[allow(dead_code)]
impl TensorImpl {
    /// Create a new instance of the TensorImpl.
    pub fn new<T: TensorRepr>(shape: &[usize], device: Device) -> Self {
        let len = shape.iter().product();
        Self {
            data: Storage::from_slice(&vec![T::zero(); len], device),
            grad: Some(crate::tensor(&vec![T::zero(); len], shape, device)),
            shape: shape.to_vec(),
            stride: Self::compute_stride(shape),
            offset: 0,
            prev: vec![],
            op: None,
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
    pub fn zero_grad<T: TensorRepr>(&mut self) {
        let len = self.shape.iter().product();
        self.grad = Some(crate::tensor(
            &vec![T::zero(); len],
            &self.shape,
            self.device(),
        ));
    }

    /// Sets gradients to `None`
    pub fn grad_none(&mut self) {
        self.grad = None;
    }

    pub fn set_grad<T: TensorRepr>(&mut self, grad: Vec<T>) {
        self.grad = Some(crate::tensor(&grad, &self.shape, self.device()));
    }

    /// Creates a new instance of the TensorImpl from a Vector.
    pub fn from_slice<T: TensorRepr>(data: &[T], shape: &[usize], device: Device) -> Self {
        Self {
            data: Storage::from_slice(data, device),
            grad: None,
            shape: shape.to_vec(),
            stride: Self::compute_stride(shape),
            offset: 0,
            prev: vec![],
            op: None,
        }
    }

    pub fn from_storage(storage: Storage, shape: &[usize]) -> Self {
        Self {
            data: storage,
            grad: None,
            shape: shape.to_vec(),
            stride: Self::compute_stride(shape),
            offset: 0,
            prev: vec![],
            op: None,
        }
    }

    /// Creates a new instance of the TensorImpl produced by any `Op`.
    pub fn from_op<T: TensorRepr>(
        data: Vec<T>,
        shape: &[usize],
        prev: Vec<Tensor>,
        op: Op,
        device: Device,
    ) -> Self {
        let data = match device {
            Device::CPU => Storage::CPU(T::into_cpu_storage(&data)),
            #[cfg(feature = "cuda")]
            Device::CUDA => Storage::CUDA(T::into_cuda_storage(&data)),
        };

        let _grad = vec![T::zero(); data.len()];
        let grad = match device {
            Device::CPU => crate::create_grad::<T>(shape, Device::CPU),
            #[cfg(feature = "cuda")]
            Device::CUDA => crate::create_grad::<T>(shape, Device::CUDA),
        };

        Self {
            data: data,
            grad: Some(grad),
            shape: shape.to_vec(),
            stride: Self::compute_stride(shape),
            offset: 0,
            prev: prev,
            op: Some(op),
        }
    }

    pub fn from_op_and_storage(
        storage: Storage,
        shape: &[usize],
        prev: Vec<Tensor>,
        op: Op,
    ) -> Self {
        let grad = match storage.device() {
            Device::CPU => match storage.dtype() {
                DType::Float8 => crate::create_grad::<f8>(shape, Device::CPU),
                DType::Float16 => crate::create_grad::<f16>(shape, Device::CPU),
                DType::BFloat16 => crate::create_grad::<bf16>(shape, Device::CPU),
                DType::Float32 => crate::create_grad::<f32>(shape, Device::CPU),
                DType::Float64 => crate::create_grad::<f64>(shape, Device::CPU),
                DType::Int8 => crate::create_grad::<i8>(shape, Device::CPU),
                DType::Int16 => crate::create_grad::<i16>(shape, Device::CPU),
                DType::Int32 => crate::create_grad::<i32>(shape, Device::CPU),
                DType::Int64 => crate::create_grad::<i64>(shape, Device::CPU),
                DType::Bool => todo!(),
            },
            #[cfg(feature = "cuda")]
            Device::CUDA => match storage.dtype() {
                DType::Float8 => crate::create_grad::<f8>(shape, Device::CUDA),
                DType::Float16 => crate::create_grad::<f16>(shape, Device::CUDA),
                DType::BFloat16 => crate::create_grad::<bf16>(shape, Device::CUDA),
                DType::Float32 => crate::create_grad::<f32>(shape, Device::CUDA),
                DType::Float64 => crate::create_grad::<f64>(shape, Device::CUDA),
                DType::Int8 => crate::create_grad::<i8>(shape, Device::CUDA),
                DType::Int16 => crate::create_grad::<i16>(shape, Device::CUDA),
                DType::Int32 => crate::create_grad::<i32>(shape, Device::CUDA),
                DType::Int64 => crate::create_grad::<i64>(shape, Device::CUDA),
                DType::Bool => todo!(),
            },
        };

        Self {
            data: storage,
            grad: Some(grad),
            shape: shape.to_vec(),
            stride: Self::compute_stride(shape),
            offset: 0,
            prev: prev,
            op: Some(op),
        }
    }

    #[cfg(feature = "cuda")]
    pub fn from_cuda<T: TensorRepr>(
        data: CudaSlice<T>,
        shape: &[usize],
        prev: Vec<Tensor>,
        op: Option<Op>,
    ) -> Self {
        let data = Storage::CUDA(T::into_cuda_storage_from_cuda(data));

        let grad = crate::create_grad::<T>(shape, Device::CUDA).cuda();

        Self {
            data: data,
            grad: Some(grad),
            shape: shape.to_vec(),
            stride: Self::compute_stride(shape),
            offset: 0,
            prev: prev,
            op: op,
        }
    }

    pub fn cpu(&self) -> Self {
        match self.device() {
            Device::CPU => self.clone(),
            #[cfg(feature = "cuda")]
            Device::CUDA => Self::from_storage(self.data.to_cpu(), &self.shape),
        }
    }

    #[cfg(feature = "cuda")]
    pub fn cuda(&self) -> Self {
        match self.device() {
            Device::CPU => Self::from_storage(self.data.to_cuda(), &self.shape),
            Device::CUDA => self.clone(),
        }
    }

    pub fn device(&self) -> Device {
        self.data.device()
    }

    pub fn dtype(&self) -> DType {
        self.data.dtype()
    }
}

impl PartialEq for TensorImpl {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data && self.shape == other.shape
    }
}
