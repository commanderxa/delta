mod binary_ops_kernel;
pub mod cast;
pub mod dtype;
pub(crate) mod impl_;
mod slice_index;
#[macro_use]
pub mod init;
pub mod operations;
#[macro_use]
pub(crate) mod promote;
pub(crate) mod repr;
pub(crate) mod storage;
pub(crate) mod storage_impl;

use std::{cell::RefCell, fmt::Display, ops::Range, rc::Rc};

use half::{bf16, f16};
use num_traits::Float;

use crate::{
    DType, Storage,
    backward::Backward,
    device::Device,
    f8,
    op::Op,
    tensor::{
        cast::Cast,
        impl_::TensorImpl,
        repr::{FloatTensorRepr, TensorRepr},
        slice_index::{SliceIndex, SliceIndexArg, SliceIndexEnum},
        storage_impl::CPUStorage,
    },
};

type TensorImplRef = Rc<RefCell<TensorImpl>>;

#[derive(Clone, Debug)]
/// # Tensor
///
/// ### Holds the reference to the inner data inside.
///
/// See the documentation for `TensorImpl`.
pub struct Tensor {
    pub(crate) inner: TensorImplRef,
}

impl Tensor {
    /// Creates a new instance of a `Tensor`.
    pub fn new(inner: TensorImpl) -> Self {
        Self {
            inner: Rc::new(RefCell::new(inner)),
        }
    }

    /// Returns the shape of the tensor as vector.
    pub fn shape(&self) -> Vec<usize> {
        self.inner.borrow().shape.clone()
    }

    /// Returns the total length of the vector. It obtains the length by taking
    /// the product of the shape of the tensor.
    ///
    /// E.g. if the shape of the tensor is (3, 2, 3), then the length is
    /// (3 * 2 * 3) => 18.
    pub fn length(&self) -> usize {
        self.shape().iter().product()
    }

    pub fn storage(&self) -> Storage {
        self.inner.borrow().data.clone()
    }

    pub fn dtype(&self) -> DType {
        self.inner.borrow().dtype()
    }

    pub fn device(&self) -> Device {
        self.inner.borrow().device()
    }

    pub(crate) fn offset(&self) -> usize {
        self.inner.borrow().offset
    }

    /// Fills the given empty tensor with a values with an inputted range. It
    /// also sets the gradients to 0.
    ///
    /// E.g. if the range is (-1.0..1.0) then each value in the tensor will be
    /// between -1 and 1.
    pub(crate) fn fill_tensor<T: FloatTensorRepr>(tensor: &mut TensorImpl, range: Range<T>) {
        let len = tensor.data.len();
        let values: Vec<T> = (0..len).map(|_| T::random_range(range.clone())).collect();
        tensor.data.replace_data(&values);
        tensor.grad = Some(crate::create_grad::<T>(&tensor.shape, tensor.device()));
    }

    /// Returns an owned copy of tensor strides
    pub fn stride(&self) -> Vec<usize> {
        self.inner.borrow().stride.clone()
    }

    /// Returns the data of the tensor.
    pub fn data<T: TensorRepr>(&self) -> Vec<T> {
        let storage = self.storage();
        let storage = storage.as_cpu();
        let stride = self.stride();
        let shape = self.shape();
        let offset = self.offset();
        let mut mask = vec![0; shape.len()];
        let mut data = vec![T::zero(); self.length()];
        // iterate over storage data
        for d in data.iter_mut() {
            // compute index of past position of data
            *d = storage[offset + stride.iter().zip(&mask).map(|(a, b)| a * b).sum::<usize>()];
            // iterate over shape
            for j in (0..shape.len()).rev() {
                // skip the properly filled dims
                if shape[j] - 1 == mask[j] {
                    continue;
                }
                // increment the necessary mask dim
                mask[j] += 1;
                // set to 0 all prevous shape dims
                for k in ((j + 1)..shape.len()).rev() {
                    mask[k] = 0;
                }
                break;
            }
        }
        data
    }

    /// Defines the `Tensor` behavior.
    ///
    /// The tensor property `requires_grad` is `true` by default, which means
    /// that the `Tensor` has a gradient, but this gradient might be sent to
    /// `None` if is not necessary.
    pub fn requires_grad(self, value: bool) -> Self {
        if value {
            let grad = match self.device() {
                Device::CPU => self.init_grad(),
                #[cfg(feature = "cuda")]
                Device::CUDA => self.init_grad().cuda(),
            };
            self.inner.borrow_mut().grad = Some(grad);
        } else {
            self.inner.borrow_mut().grad = None;
        }
        self
    }

    pub(crate) fn init_grad(&self) -> Tensor {
        let shape = self.shape();
        let len = shape.iter().product();
        match self.dtype() {
            DType::Float8 => crate::tensor(&vec![f8::zero(); len], &shape, self.device()),
            DType::Float16 => crate::tensor(&vec![f16::zero(); len], &shape, self.device()),
            DType::BFloat16 => crate::tensor(&vec![bf16::zero(); len], &shape, self.device()),
            DType::Float32 => crate::tensor(&vec![f32::zero(); len], &shape, self.device()),
            DType::Float64 => crate::tensor(&vec![f64::zero(); len], &shape, self.device()),
            DType::Int8 => panic!("Only floating point tensors can require gradients."),
            DType::Int16 => panic!("Only floating point tensors can require gradients."),
            DType::Int32 => panic!("Only floating point tensors can require gradients."),
            DType::Int64 => panic!("Only floating point tensors can require gradients."),
            DType::Bool => panic!("Only floating point tensors can require gradients."),
        }
    }

    pub fn sum(&self, dim: Option<usize>, keepdim: bool) -> Tensor {
        let inner_data = match self.dtype() {
            DType::Float8 => todo!(),
            DType::Float16 => todo!(),
            DType::BFloat16 => todo!(),
            DType::Float32 => match dim {
                None => {
                    let value = self
                        .data()
                        .iter()
                        .fold(f32::zero(), |acc, x: &f32| acc + *x);
                    let shape = if keepdim {
                        vec![1; self.shape().len().max(1)]
                    } else {
                        vec![1]
                    };
                    TensorImpl::from_op(
                        vec![value],
                        &shape,
                        vec![self.clone()],
                        Op::Sum { dim: None, keepdim },
                        self.device(),
                    )
                }
                Some(dim) => {
                    assert!(dim < self.shape().len(), "sum: dim out of range");

                    let input = self.data::<f32>();
                    let outer: usize = self.shape()[..dim].iter().product();
                    let reduce: usize = self.shape()[dim];
                    let inner: usize = self.shape()[dim + 1..].iter().product();

                    let mut out = vec![f32::zero(); outer * inner];

                    for o in 0..outer {
                        for i in 0..inner {
                            let mut acc = f32::zero();
                            for r in 0..reduce {
                                let idx = o * reduce * inner + r * inner + i;
                                acc = acc + input[idx];
                            }
                            out[o * inner + i] = acc;
                        }
                    }

                    let mut out_shape = self.shape().clone();
                    if keepdim {
                        out_shape[dim] = 1;
                    } else {
                        out_shape.remove(dim);
                        if out_shape.is_empty() {
                            out_shape.push(1);
                        }
                    }

                    TensorImpl::from_op(
                        out,
                        &out_shape,
                        vec![self.clone()],
                        Op::Sum {
                            dim: Some(dim),
                            keepdim,
                        },
                        self.device(),
                    )
                }
            },
            DType::Float64 => todo!(),
            DType::Int8 => todo!(),
            DType::Int16 => todo!(),
            DType::Int32 => todo!(),
            DType::Int64 => todo!(),
            DType::Bool => todo!(),
        };
        Tensor::new(inner_data)
    }

    pub fn mean(&self, dim: Option<usize>, keepdim: bool) -> Tensor {
        match dim {
            None => {
                let n = self.length();
                assert!(n > 0, "mean of empty tensor is undefined");
                let value = self
                    .data()
                    .iter()
                    .fold(<f32 as Cast<f32>>::cast(0.), |acc, x: &f32| acc + *x)
                    / <i64 as Cast<f32>>::cast(n as i64);
                let shape = if keepdim {
                    vec![1; self.shape().len().max(1)]
                } else {
                    vec![1]
                };
                let inner = TensorImpl::from_op(
                    vec![value],
                    &shape,
                    vec![self.clone()],
                    Op::Mean {
                        dim: None,
                        keepdim,
                        count: n,
                    },
                    self.device(),
                );
                Tensor::new(inner)
            }
            Some(dim) => {
                assert!(dim < self.shape().len(), "mean: dim out of range");

                let input: Vec<f32> = self.data();
                let outer: usize = self.shape()[..dim].iter().product();
                let reduce: usize = self.shape()[dim];
                let inner: usize = self.shape()[dim + 1..].iter().product();

                let mut out = vec![Cast::cast(0.); outer * inner];

                for o in 0..outer {
                    for i in 0..inner {
                        let mut acc = <f32 as Cast<f32>>::cast(0.);
                        for r in 0..reduce {
                            let idx = o * reduce * inner + r * inner + i;
                            acc = acc + input[idx];
                        }
                        out[o * inner + i] = acc / <i64 as Cast<f32>>::cast(reduce as i64);
                    }
                }

                let mut out_shape = self.shape().clone();
                if keepdim {
                    out_shape[dim] = 1;
                } else {
                    out_shape.remove(dim);
                    if out_shape.is_empty() {
                        out_shape.push(1);
                    }
                }

                let inner_data = TensorImpl::from_op(
                    out,
                    &out_shape,
                    vec![self.clone()],
                    Op::Mean {
                        dim: Some(dim),
                        keepdim,
                        count: reduce,
                    },
                    self.device(),
                );
                Tensor::new(inner_data)
            }
        }
    }

    /// Transpose
    ///
    /// This method transposes the tensor, it changes the shape.
    /// The rows become the columns and vice versa.
    ///
    /// E.g. if the shape was (2, 3) it will make (3, 2).
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Self {
        let t = self.clone();
        // transpose tensor of 2 and more dimensions
        if t.shape().len() >= 2 {
            t.inner.borrow_mut().shape.swap(dim0, dim1);
            t.inner.borrow_mut().stride.swap(dim0, dim1);
        }
        t
    }

    /// Expects input to be <= 2D and transposes 0 and 1 dims.
    ///
    /// 0D and 1D tensors are returned without any transpose performed
    pub fn t(&self) -> Self {
        self.transpose(0, 1)
    }

    /// Returns the tensor of the new shape.
    ///
    /// Tensor of shape (2, 3) might be viewed as (3, 2), (6, 1), (1, 6).
    /// Tensor can be viewed as any shape, only if the length of this shape is
    /// the same as the length of the previous shape.
    pub fn view(&self, shape: &[usize]) -> Self {
        assert_eq!(
            self.length(),
            shape.iter().product(),
            "Length of the new shape: {} does not match the length of the old one: {}",
            shape.iter().product::<usize>(),
            self.length()
        );
        let t = self.clone();
        if self.stride().iter().product::<usize>() == 0 {
            panic!(
                "view size is not compatible with size and stride of input tensor. Use .reshape(...) instead"
            );
        }
        let mut stride = vec![1; shape.len()];
        // compute stride
        for i in (0..shape.len() - 1).rev() {
            stride[i] = shape[i + 1] * stride[i + 1];
        }
        t.inner.borrow_mut().stride = stride;
        t.inner.borrow_mut().shape = shape.to_vec();
        t
    }

    /// Reshapes the tensor.
    ///
    /// Tensor of shape (2, 3) might be reshaped to (3, 2), (6, 1), (1, 6).
    /// Tensor can be reshaped into any shape, only if the length of this shape
    /// is the same as the length of the previous shape.
    pub fn reshape(&self, shape: &[usize]) -> Self {
        assert_eq!(
            self.length(),
            shape.iter().product(),
            "Length of the new shape: {} does not match the length of the old one: {}",
            shape.iter().product::<usize>(),
            self.length()
        );
        // if stride indicate that tensor wasn't expanded, then just `view` it
        if self.stride().iter().product::<usize>() > 0 {
            return self.view(shape);
        }
        let mut mask = vec![0; self.shape().len()];
        match self.dtype() {
            DType::Float8 => todo!(),
            DType::Float16 => todo!(),
            DType::BFloat16 => todo!(),
            DType::Float32 => {
                let mut data = vec![f32::zero(); self.length()];
                // iterate over storage data
                for d in data.iter_mut() {
                    // compute index of past position of data
                    *d = self.data()[self
                        .stride()
                        .iter()
                        .zip(&mask)
                        .map(|(a, b)| a * b)
                        .sum::<usize>()];
                    // iterate over shape
                    for j in (0..self.shape().len()).rev() {
                        // skip the properly filled dims
                        if self.shape()[j] - 1 == mask[j] {
                            continue;
                        }
                        // increment the necessary mask dim
                        mask[j] += 1;
                        // set to 0 all prevous shape dims
                        for k in ((j + 1)..self.shape().len()).rev() {
                            mask[k] = 0;
                        }
                        break;
                    }
                }
                let mut stride = vec![1; shape.len()];
                // compute stride
                for i in (0..shape.len() - 1).rev() {
                    stride[i] = shape[i + 1] * stride[i + 1];
                }
                let t = self.clone();
                t.inner.borrow_mut().data.replace_data(&data);
                t.inner.borrow_mut().shape = shape.to_vec();
                t.inner.borrow_mut().stride = stride;
                t
            }
            DType::Float64 => todo!(),
            DType::Int8 => todo!(),
            DType::Int16 => todo!(),
            DType::Int32 => todo!(),
            DType::Int64 => todo!(),
            DType::Bool => todo!(),
        }
    }

    /// Inserts a dimension of size 1 at a specified location in shape.
    pub fn unsqueeze(&self, dim: usize) -> Self {
        assert!(
            dim <= self.shape().len(),
            "Dimension out of range (expected range of [0, {}])",
            self.shape().len()
        );
        let t = self.clone();
        t.shape().insert(dim, 1);
        let mut replica = 1;
        if dim < self.shape().len() {
            replica = t.stride()[dim];
        }
        t.inner.borrow_mut().stride.insert(dim, replica);
        t
    }

    /// Returns a tensor with all specified dimensions of shape of size 1 removed.
    ///
    /// * If `dim` is empty it performs removal across the whole shape.
    /// * If `dim` contains dimensions, then it only considers them.
    ///
    /// All the specified dimensions that are more than 1 it leaves as it is.
    pub fn squeeze(&self, dim: &[usize]) -> Self {
        let t = self.clone();
        if dim.is_empty() {
            for d in (0..t.shape().len()).rev() {
                if t.shape()[d] == 1 {
                    t.inner.borrow_mut().shape.remove(d);
                    t.inner.borrow_mut().stride.remove(d);
                }
            }
        } else {
            for d in (0..dim.len()).rev() {
                if t.shape()[d] == 1 {
                    t.inner.borrow_mut().shape.remove(d);
                    t.inner.borrow_mut().stride.remove(d);
                }
            }
        }
        t
    }

    /// Exapnds tesnor along its dimensions.
    ///
    /// Takes new shape.
    pub fn expand(&self, new_shape: &[usize]) -> Self {
        assert!(
            self.shape().len() <= new_shape.len(),
            "The number of sizes provided ({:?}) must be equal or greater than the number of sizes in the tensor ({:?})",
            self.shape().len(),
            new_shape.len()
        );
        let t = self.clone();
        let mut _old_shape = self.shape();
        // check if batch dims have to be added in th front
        let dims_to_add = new_shape.len() - _old_shape.len();
        let mut old_shape: Vec<usize> = vec![1; dims_to_add];
        // push neccessary front batch dims
        for _ in 0..dims_to_add {
            t.inner.borrow_mut().stride.insert(0, t.stride()[0]);
        }
        // append the rest of the shape
        old_shape.append(&mut _old_shape);
        // check if sizes are consistent
        for i in (0..new_shape.len()).rev() {
            assert!(
                old_shape[i] == new_shape[i] || (old_shape[i] == 1),
                "The expanded size of the tensor ({}) must match the existing size ({}) at dimension ({})",
                new_shape[i],
                old_shape[i],
                i,
            );
            // set expanded dim strides to 0
            if old_shape[i] == 1 && new_shape[i] > 1 {
                t.inner.borrow_mut().stride[i] = 0;
            }
        }

        // change tensor properties
        t.inner.borrow_mut().shape = new_shape.to_vec();
        t
    }

    /// Exponents each value of the `Tensor`.
    ///
    /// `exp(x)` => `e^(x)`.
    pub fn exp(&self) -> Tensor {
        let inner = match self.dtype() {
            DType::Float8 => {
                let data = self.data::<f8>().iter().map(|x| x.exp()).collect();
                TensorImpl::from_op(
                    data,
                    &self.shape(),
                    vec![self.clone()],
                    Op::Exp(self.clone()),
                    self.device(),
                )
            }
            DType::Float16 => {
                let data = self.data::<f16>().iter().map(|x| x.exp()).collect();
                TensorImpl::from_op(
                    data,
                    &self.shape(),
                    vec![self.clone()],
                    Op::Exp(self.clone()),
                    self.device(),
                )
            }
            DType::BFloat16 => {
                let data = self.data::<bf16>().iter().map(|x| x.exp()).collect();
                TensorImpl::from_op(
                    data,
                    &self.shape(),
                    vec![self.clone()],
                    Op::Exp(self.clone()),
                    self.device(),
                )
            }
            DType::Float32 => {
                let data = self.data::<f32>().iter().map(|x| x.exp()).collect();
                TensorImpl::from_op(
                    data,
                    &self.shape(),
                    vec![self.clone()],
                    Op::Exp(self.clone()),
                    self.device(),
                )
            }
            DType::Float64 => {
                let data = self.data::<f64>().iter().map(|x| x.exp()).collect();
                TensorImpl::from_op(
                    data,
                    &self.shape(),
                    vec![self.clone()],
                    Op::Exp(self.clone()),
                    self.device(),
                )
            }
            DType::Int8 => todo!(),
            DType::Int16 => todo!(),
            DType::Int32 => todo!(),
            DType::Int64 => todo!(),
            DType::Bool => todo!(),
        };
        Tensor::new(inner)
    }

    pub fn contiguous(&self) -> Self {
        // already contiguous -> no work needed
        if self.is_contiguous() {
            return self.clone();
        }

        let old_data = self.cpu().storage();
        let old_data = old_data.as_cpu();
        let shape = self.shape();
        let strides = self.stride();
        let offset = self.offset();

        let total: usize = shape.iter().product();
        let inner = match self.dtype() {
            DType::Float8 => todo!(),
            DType::Float16 => todo!(),
            DType::BFloat16 => todo!(),
            DType::Float32 => {
                let mut new_data = vec![f32::zero(); total];

                for flat_idx in 0..total {
                    // flat index -> nd index in current shape
                    let mut nd_idx = vec![0usize; shape.len()];
                    let mut remainder = flat_idx;
                    for d in (0..shape.len()).rev() {
                        nd_idx[d] = remainder % shape[d];
                        remainder /= shape[d];
                    }

                    // nd index -> flat index in source data via strides
                    let src_idx: usize = nd_idx
                        .iter()
                        .zip(strides.iter())
                        .map(|(&i, &s)| i * s)
                        .sum();

                    new_data[flat_idx] = old_data[offset + src_idx];
                }

                // compute new contiguous row-major strides
                let mut new_strides = vec![1usize; shape.len()];
                for d in (0..shape.len() - 1).rev() {
                    new_strides[d] = new_strides[d + 1] * shape[d + 1];
                }

                let device = self.device();

                match device {
                    Device::CPU => TensorImpl::from_slice(&new_data, &shape, device),
                    #[cfg(feature = "cuda")]
                    Device::CUDA => {
                        use crate::cuda::array_to_cuda_slice;

                        let _inner = &self.inner.borrow();
                        TensorImpl::from_cuda(
                            array_to_cuda_slice(&new_data),
                            &shape,
                            _inner.prev.clone(),
                            _inner.op.clone(),
                        )
                    }
                }
            }
            DType::Float64 => todo!(),
            DType::Int8 => todo!(),
            DType::Int16 => todo!(),
            DType::Int32 => todo!(),
            DType::Int64 => todo!(),
            DType::Bool => todo!(),
        };

        Self {
            inner: Rc::new(RefCell::new(inner)),
        }
    }

    pub(crate) fn is_contiguous(&self) -> bool {
        let mut expected = 1usize;
        if self.inner.borrow().offset > 0 {
            return false;
        }
        for d in (0..self.shape().len()).rev() {
            if self.stride()[d] != expected {
                return false;
            }
            expected *= self.shape()[d];
        }
        true
    }

    /// Add a `Vec` value to the gradient inside the `TensorImpl`.
    pub(crate) fn add_to_grad(&self, tensor: Tensor) {
        let mut t: std::cell::RefMut<'_, TensorImpl> = self.inner.borrow_mut();
        // t.grad = Some(Storage::from_slice(data, self.device.clone()));
        t.grad = Some(t.grad.clone().unwrap() + tensor);
    }

    /// Returns the gradient vector.
    pub fn grad(&self) -> Option<Tensor> {
        self.inner.borrow().grad.clone()
    }

    pub fn zero_grad<T: FloatTensorRepr>(&self) {
        self.inner.borrow_mut().zero_grad::<T>();
    }

    /// Replace current data inside the tensor with new `data`
    pub fn set_data<T: TensorRepr>(&self, data: Vec<T>) {
        self.inner.borrow_mut().data.replace_data(&data);
    }

    /// Multicast operation
    ///
    /// It ensures that the lower dimensions of the two tensors are the same if
    /// they are, then it performs the given operation elementwise, using the
    /// lower dimensional tensor as a convolution window.
    ///
    /// Accepts:
    /// * a: Tensor
    /// * b: Tensor
    /// * op: operation `Op`, permitted operations are `Add` and `Mul`
    ///
    /// Returns Tensor
    pub(crate) fn multicast_op(a: Tensor, b: Tensor, op: Op) -> Self {
        let mut a = a;
        let mut b = b;
        // check whether to expand any of variables
        if a.shape() != b.shape() {
            // if `a` tensor is bigger => expand `b`
            // else expand `a`
            if a.length() > b.length() {
                b = b.expand(&a.shape());
            } else {
                a = a.expand(&b.shape());
            }
        }

        let device = a.device();

        promote_tensors!(&mut a, &mut b);

        let inner = match a.dtype() {
            DType::Float8 => todo!(),
            DType::Float16 => todo!(),
            DType::BFloat16 => todo!(),
            DType::Float32 => {
                let mut mask = vec![0; a.shape().len()];
                let mut data = vec![Cast::cast(0.); a.length()];
                // iterate over storage data
                for d in data.iter_mut() {
                    // compute index of past position of data
                    let a_i = a.data::<f32>()[a
                        .stride()
                        .iter()
                        .zip(&mask)
                        .map(|(a, b)| a * b)
                        .sum::<usize>()];
                    let b_i: f32 = b.data()[b
                        .stride()
                        .iter()
                        .zip(&mask)
                        .map(|(a, b)| a * b)
                        .sum::<usize>()];
                    // write the result for particular element based on the operation
                    *d = match op {
                        Op::Add => a_i + b_i,
                        Op::Sub => a_i - b_i,
                        Op::Mul => a_i * b_i,
                        _ => unreachable!(),
                    };
                    for j in (0..a.shape().len()).rev() {
                        if a.shape()[j] - 1 == mask[j] {
                            continue;
                        }
                        mask[j] += 1;
                        for k in ((j + 1)..a.shape().len()).rev() {
                            mask[k] = 0;
                        }
                        break;
                    }
                }
                TensorImpl::from_op(data, &a.shape(), vec![a, b], op, device)
            }
            DType::Float64 => {
                let mut mask = vec![0; a.shape().len()];
                let mut data = vec![Cast::cast(0.); a.length()];
                // iterate over storage data
                for d in data.iter_mut() {
                    // compute index of past position of data
                    let a_i = a.data::<f64>()[a
                        .stride()
                        .iter()
                        .zip(&mask)
                        .map(|(a, b)| a * b)
                        .sum::<usize>()];
                    let b_i: f64 = b.data()[b
                        .stride()
                        .iter()
                        .zip(&mask)
                        .map(|(a, b)| a * b)
                        .sum::<usize>()];
                    // write the result for particular element based on the operation
                    *d = match op {
                        Op::Add => a_i + b_i,
                        Op::Sub => a_i - b_i,
                        Op::Mul => a_i * b_i,
                        _ => unreachable!(),
                    };
                    for j in (0..a.shape().len()).rev() {
                        if a.shape()[j] - 1 == mask[j] {
                            continue;
                        }
                        mask[j] += 1;
                        for k in ((j + 1)..a.shape().len()).rev() {
                            mask[k] = 0;
                        }
                        break;
                    }
                }
                TensorImpl::from_op(data, &a.shape(), vec![a, b], op, device)
            }
            DType::Int8 => todo!(),
            DType::Int16 => todo!(),
            DType::Int32 => todo!(),
            DType::Int64 => todo!(),
            DType::Bool => todo!(),
        };

        Self::new(inner)
    }

    #[cfg(feature = "cuda")]
    pub fn cuda(&self) -> Tensor {
        match self.device() {
            Device::CPU => Tensor {
                inner: Rc::new(RefCell::new(self.inner.borrow().cuda())),
            },
            Device::CUDA => self.clone(),
        }
    }

    pub fn cpu(&self) -> Tensor {
        match self.device() {
            Device::CPU => self.clone(),
            #[cfg(feature = "cuda")]
            Device::CUDA => Tensor {
                inner: Rc::new(RefCell::new(self.inner.borrow().cpu())),
            },
        }
    }

    /// Concatenates a slice of tensors along the given dimension.
    /// All tensors must have the same shape except in the `dim` axis.
    pub fn cat(tensors: &[Tensor], dim: isize) -> Self {
        assert!(!tensors.is_empty(), "cat: need at least one tensor");
        assert!(dim >= -1, "cat: `dim` cannot be negative integer");

        let ndim = tensors[0].shape().len();

        let dim: usize = if dim == -1 {
            tensors[0].shape().len() - 1
        } else {
            dim as usize
        };

        let device = tensors[0].device();
        for t in tensors {
            assert_eq!(t.device(), device)
        }

        assert!(
            dim < ndim,
            "cat: dim {} out of range for {}-D tensor",
            dim,
            ndim
        );

        // Validate all shapes match except on `dim`
        for t in tensors.iter().skip(1) {
            assert_eq!(
                t.shape().len(),
                ndim,
                "cat: all tensors must have same number of dims"
            );
            for d in 0..ndim {
                if d != dim {
                    assert_eq!(
                        t.shape()[d],
                        tensors[0].shape()[d],
                        "cat: shape mismatch on dim {}",
                        d
                    );
                }
            }
        }

        // Compute output shape
        let mut out_shape = tensors[0].shape().clone();
        out_shape[dim] = tensors.iter().map(|t| t.shape()[dim]).sum();

        // Collect contiguous data from each tensor
        let mut data: Vec<f32> = Vec::with_capacity(out_shape.iter().product());

        // Iterate over all positions except `dim`, then gather slices
        let outer: usize = out_shape[..dim].iter().product();
        let inner: usize = out_shape[dim + 1..].iter().product();

        for o in 0..outer {
            for t in tensors {
                let slice_len = t.shape()[dim] * inner;
                let offset = o * slice_len;
                let items = t.data();
                data.extend_from_slice(&items[offset..offset + slice_len]);
            }
        }

        let inner_data = TensorImpl::from_slice(&data, &out_shape, device);
        Tensor::new(inner_data)
    }

    pub fn cast(self, dtype: DType) -> Tensor {
        let shape = self.shape();
        let new_storage = match dtype {
            DType::Float8 => self.storage().cast_to::<f8>(),
            DType::Float16 => self.storage().cast_to::<f16>(),
            DType::BFloat16 => self.storage().cast_to::<bf16>(),
            DType::Float32 => self.storage().cast_to::<f32>(),
            DType::Float64 => self.storage().cast_to::<f64>(),
            DType::Int8 => self.storage().cast_to::<i8>(),
            DType::Int16 => self.storage().cast_to::<i16>(),
            DType::Int32 => self.storage().cast_to::<i32>(),
            DType::Int64 => self.storage().cast_to::<i64>(),
            DType::Bool => self.storage().cast_to::<bool>(),
        };
        let inner = TensorImpl::from_storage(new_storage, &shape);
        Tensor::new(inner)
    }

    pub fn cast_(&mut self, dtype: DType) {
        let new_storage = match dtype {
            DType::Float8 => self.storage().cast_to::<f8>(),
            DType::Float16 => self.storage().cast_to::<f16>(),
            DType::BFloat16 => self.storage().cast_to::<bf16>(),
            DType::Float32 => self.storage().cast_to::<f32>(),
            DType::Float64 => self.storage().cast_to::<f64>(),
            DType::Int8 => self.storage().cast_to::<i8>(),
            DType::Int16 => self.storage().cast_to::<i16>(),
            DType::Int32 => self.storage().cast_to::<i32>(),
            DType::Int64 => self.storage().cast_to::<i64>(),
            DType::Bool => self.storage().cast_to::<bool>(),
        };
        self.inner.borrow_mut().data = new_storage;
    }

    /// Backward
    ///
    /// Computes the gradients of all the tensors that have been interacting and
    /// have `requires_grad` set to `true`.
    pub fn backward(&self) {
        assert!(
            self.length() == 1,
            "grad can be implicitly created only for scalar outputs"
        );

        match self.dtype() {
            DType::Float8 => self.add_to_grad(crate::tensor(&[f8::one()], &[1], self.device())),
            DType::Float16 => self.add_to_grad(crate::tensor(&[f16::one()], &[1], self.device())),
            DType::BFloat16 => self.add_to_grad(crate::tensor(&[bf16::one()], &[1], self.device())),
            DType::Float32 => self.add_to_grad(crate::tensor(&[f32::one()], &[1], self.device())),
            DType::Float64 => self.add_to_grad(crate::tensor(&[f64::one()], &[1], self.device())),
            DType::Int8 => panic!("Only floating point tensors can require gradients."),
            DType::Int16 => panic!("Only floating point tensors can require gradients."),
            DType::Int32 => panic!("Only floating point tensors can require gradients."),
            DType::Int64 => panic!("Only floating point tensors can require gradients."),
            DType::Bool => panic!("Only floating point tensors can require gradients."),
        }
        self._backward()
    }

    /// Backward private
    ///
    /// Evokes Backward function in all components of the computational graph
    fn _backward(&self) {
        let t = self.inner.borrow();
        if t.grad.is_some() && t.op.is_some() {
            t.op.as_ref().unwrap().backward(&self);
            if !t.prev.is_empty() {
                for prev in t.prev.clone() {
                    prev._backward()
                }
            }
        }
    }

    /// Powers the `Tensor`
    ///
    /// Accepts `n` integer in which the `Tensor` will be powered.
    ///
    /// For backpropagation it stores the `n` inside the `Op::Pow(n)`.
    pub fn pow(&self, n: i32) -> Self {
        let new_storage = match self.storage() {
            Storage::CPU(s) => Storage::CPU(match s {
                CPUStorage::F32(v) => CPUStorage::F32(v.iter().map(|x| x.powi(n)).collect()),
                CPUStorage::F64(v) => CPUStorage::F64(v.iter().map(|x| x.powi(n)).collect()),
                CPUStorage::F16(v) => CPUStorage::F16(
                    v.iter()
                        .map(|x| f16::from_f32(x.to_f32().powi(n)))
                        .collect(),
                ),
                CPUStorage::BF16(v) => CPUStorage::BF16(
                    v.iter()
                        .map(|x| bf16::from_f32(x.to_f32().powi(n)))
                        .collect(),
                ),
                _ => panic!("log() not supported for integer dtypes"),
            }),
            #[cfg(feature = "cuda")]
            Storage::CUDA(_) => todo!(),
        };
        let inner = TensorImpl::from_op_and_storage(
            new_storage,
            &self.shape(),
            vec![self.clone()],
            Op::Pow(n),
        );
        Self::new(inner)
    }

    pub fn log(&self) -> Self {
        let new_storage = match self.storage() {
            Storage::CPU(s) => Storage::CPU(match s {
                CPUStorage::F32(v) => CPUStorage::F32(v.iter().map(|x| x.ln()).collect()),
                CPUStorage::F64(v) => CPUStorage::F64(v.iter().map(|x| x.ln()).collect()),
                CPUStorage::F16(v) => {
                    CPUStorage::F16(v.iter().map(|x| f16::from_f32(x.to_f32().ln())).collect())
                }
                CPUStorage::BF16(v) => {
                    CPUStorage::BF16(v.iter().map(|x| bf16::from_f32(x.to_f32().ln())).collect())
                }
                _ => panic!("log() not supported for integer dtypes"),
            }),
            #[cfg(feature = "cuda")]
            Storage::CUDA(_) => todo!(),
        };
        let inner = TensorImpl::from_op_and_storage(
            new_storage,
            &self.shape(),
            vec![self.clone()],
            Op::Exp(self.clone()),
        );
        Self::new(inner)
    }

    /// Converts the tensor to a `String`, so that it can be printed.
    fn tensor_to_str(&self, tensor_str: String, level: usize, range: Range<usize>) -> String {
        // TODO: for different data types
        let mut width = 1;
        width = match self.dtype() {
            DType::Float8 => {
                for i in self.data::<f8>() {
                    let s = (i.floor().to_f32() as i64).to_string();
                    if s.len() > width {
                        width = s.len();
                    }
                }
                width += 5;
                width
            }
            DType::Float16 => {
                for i in self.data::<f16>() {
                    let s = (i.floor().to_f32() as i64).to_string();
                    if s.len() > width {
                        width = s.len();
                    }
                }
                width += 5;
                width
            }
            DType::BFloat16 => {
                for i in self.data::<bf16>() {
                    let s = (i.floor().to_f32() as i64).to_string();
                    if s.len() > width {
                        width = s.len();
                    }
                }
                width += 5;
                width
            }
            DType::Float32 => {
                for i in self.data::<f32>() {
                    let s = (i.floor() as i64).to_string();
                    if s.len() > width {
                        width = s.len();
                    }
                }
                width += 5;
                width
            }
            DType::Float64 => {
                for i in self.data::<f64>() {
                    let s = (i.floor() as i64).to_string();
                    if s.len() > width {
                        width = s.len();
                    }
                }
                width += 5;
                width
            }
            DType::Int8 => {
                for i in self.data::<i8>() {
                    let s = i.to_string();
                    if s.len() > width {
                        width = s.len();
                    }
                }
                width += 1;
                width
            }
            DType::Int16 => {
                for i in self.data::<i16>() {
                    let s = i.to_string();
                    if s.len() > width {
                        width = s.len();
                    }
                }
                width += 1;
                width
            }
            DType::Int32 => {
                for i in self.data::<i32>() {
                    let s = i.to_string();
                    if s.len() > width {
                        width = s.len();
                    }
                }
                width += 1;
                width
            }
            DType::Int64 => {
                for i in self.data::<i64>() {
                    let s = i.to_string();
                    if s.len() > width {
                        width = s.len();
                    }
                }
                width += 1;
                width
            }
            DType::Bool => {
                for i in self.data::<bool>() {
                    let s = i.to_string();
                    if s.len() > width {
                        width = s.len();
                    }
                }
                width += 1;
                width
            }
        };
        self._tensor_to_str(tensor_str, level, range, width)
    }

    /// Inner mechanics of converting `Tensor` into string
    fn _tensor_to_str(
        &self,
        tensor_str: String,
        level: usize,
        range: Range<usize>,
        width: usize,
    ) -> String {
        match self.dtype() {
            DType::Float8 => self._tensor_to_str_float(tensor_str, level, range, width),
            DType::Float16 => self._tensor_to_str_float(tensor_str, level, range, width),
            DType::BFloat16 => self._tensor_to_str_float(tensor_str, level, range, width),
            DType::Float32 => self._tensor_to_str_float(tensor_str, level, range, width),
            DType::Float64 => self._tensor_to_str_float(tensor_str, level, range, width),
            DType::Int8 => self._tensor_to_str_int(tensor_str, level, range, width),
            DType::Int16 => self._tensor_to_str_int(tensor_str, level, range, width),
            DType::Int32 => self._tensor_to_str_int(tensor_str, level, range, width),
            DType::Int64 => self._tensor_to_str_int(tensor_str, level, range, width),
            DType::Bool => self._tensor_to_str_bool(tensor_str, level, range, width),
        }
    }

    fn _tensor_to_str_float(
        &self,
        tensor_str: String,
        level: usize,
        range: Range<usize>,
        width: usize,
    ) -> String {
        let len: usize = range.end - range.start;
        // the current dimension from the shape
        let dim = self.shape()[level];
        // convolution to iterate over the data
        let conv = len / dim;
        // the length of shape vector
        let shape_size = self.shape().len();
        let item = self.data::<f32>();
        // denote the start of the dimension
        let mut result = String::from("[");
        // iterate over the dimension => print a vector
        for i in (range.start..range.end).step_by(conv) {
            // if the dimension is the last one
            let mut spaces: usize = 0;
            if shape_size - 1 == level {
                let s = (item[i].floor() as i64).to_string();
                if s.len() < width {
                    spaces = width - (s.len() + 5);
                }
                for _ in 0..spaces {
                    result.push(' ');
                }
                let mut num = format!("{:.4}", item[i]);
                if i < self.shape()[level] - 1 {
                    num.push_str(", ");
                }
                result.push_str(num.as_str());
            }
            // if the iteration is over the last 2 dimensions => print a matrix
            else if shape_size - 2 == level {
                result.push('[');
                for j in 0..self.shape()[level + 1] {
                    let s = (item[i + j].floor() as i64).to_string();
                    if s.len() < width {
                        spaces = width - (s.len() + 5);
                    }
                    for _ in 0..spaces {
                        result.push(' ');
                    }
                    let mut num = format!("{:.4}", item[i + j]);
                    if j < self.shape()[level + 1] - 1 {
                        num.push_str(", ");
                    }
                    result.push_str(num.as_str());
                }
                // close the matrix and add indents for the following row (if exists)
                if i != range.end - conv {
                    result.push_str("],\n\t");
                    let space = String::from(" ").repeat(shape_size - 2);
                    result.push_str(space.as_str());
                } else {
                    result.push(']');
                }
            } else {
                // else, fall further into the next dimensions
                result.push_str(
                    self._tensor_to_str(tensor_str.clone(), level + 1, i..(i + conv), width)
                        .as_str(),
                );
                // make indents for following tensor (if exists)
                if i != range.end - conv {
                    result.push(',');
                    for _ in 0..(shape_size - (level + 3)) {
                        result.push('\n');
                    }
                    result.push_str("\n\n\t");
                    let space = String::from(" ").repeat(level);
                    result.push_str(space.as_str());
                }
            }
        }
        // denote the end of the dimension
        result.push(']');
        result
    }

    fn _tensor_to_str_int(
        &self,
        tensor_str: String,
        level: usize,
        range: Range<usize>,
        width: usize,
    ) -> String {
        let len: usize = range.end - range.start;
        // the current dimension from the shape
        let dim = self.shape()[level];
        // convolution to iterate over the data
        let conv = len / dim;
        // the length of shape vector
        let shape_size = self.shape().len();
        let item = self.data::<i32>();
        // denote the start of the dimension
        let mut result = String::from("[");
        // iterate over the dimension => print a vector
        for i in (range.start..range.end).step_by(conv) {
            // if the dimension is the last one
            let mut spaces: usize = 0;
            if shape_size - 1 == level {
                let s = (item[i]).to_string();
                if s.len() < width {
                    spaces = width - (s.len() + 1);
                }
                for _ in 0..spaces {
                    result.push(' ');
                }
                let mut num = format!("{}", item[i]);
                if i < self.shape()[level] - 1 {
                    num.push_str(", ");
                }
                result.push_str(num.as_str());
            }
            // if the iteration is over the last 2 dimensions => print a matrix
            else if shape_size - 2 == level {
                result.push('[');
                for j in 0..self.shape()[level + 1] {
                    let s = (item[i + j] as i64).to_string();
                    if s.len() < width {
                        spaces = width - (s.len() + 1);
                    }
                    for _ in 0..spaces {
                        result.push(' ');
                    }
                    let mut num = format!("{}", item[i + j]);
                    if j < self.shape()[level + 1] - 1 {
                        num.push_str(", ");
                    }
                    result.push_str(num.as_str());
                }
                // close the matrix and add indents for the following row (if exists)
                if i != range.end - conv {
                    result.push_str("],\n\t");
                    let space = String::from(" ").repeat(shape_size - 2);
                    result.push_str(space.as_str());
                } else {
                    result.push(']');
                }
            } else {
                // else, fall further into the next dimensions
                result.push_str(
                    self._tensor_to_str(tensor_str.clone(), level + 1, i..(i + conv), width)
                        .as_str(),
                );
                // make indents for following tensor (if exists)
                if i != range.end - conv {
                    result.push(',');
                    for _ in 0..(shape_size - (level + 3)) {
                        result.push('\n');
                    }
                    result.push_str("\n\n\t");
                    let space = String::from(" ").repeat(level);
                    result.push_str(space.as_str());
                }
            }
        }
        // denote the end of the dimension
        result.push(']');
        result
    }

    fn _tensor_to_str_bool(
        &self,
        tensor_str: String,
        level: usize,
        range: Range<usize>,
        width: usize,
    ) -> String {
        todo!()
    }

    pub fn slice<I>(&self, slices: I) -> Tensor
    where
        I: SliceIndexArg,
    {
        let slice_enums = slices.as_slice_arg(); // returns Vec<SliceIndexEnum>
        let shape = self.shape();

        let slice_specs: Vec<SliceIndex> = slice_enums
            .into_iter()
            .enumerate()
            .map(|(dim, s)| match s {
                SliceIndexEnum::Full => SliceIndex {
                    start: 0,
                    end: shape[dim] as isize,
                    step: 1,
                },
                SliceIndexEnum::Range(s) => {
                    let end = if s.end < 0 {
                        shape[dim] as isize - s.end
                    } else {
                        s.end
                    };
                    SliceIndex {
                        start: s.start,
                        end: end,
                        step: 1,
                    }
                }
                SliceIndexEnum::Index(i) => {
                    let real = SliceIndex::map_negative(i, shape[dim] as isize);
                    SliceIndex {
                        start: real,
                        end: real + 1,
                        step: 1,
                    }
                }
            })
            .collect();
        let strides = self.stride();
        assert_eq!(slice_specs.len(), shape.len());

        let mut new_shape = Vec::with_capacity(shape.len());
        let mut new_strides = Vec::with_capacity(shape.len());
        let mut new_offset = 0;

        for (dim, &SliceIndex { start, end, step }) in slice_specs.iter().enumerate() {
            let dim_len = shape[dim] as isize;

            let real_start = SliceIndex::map_negative(start, dim_len);
            let real_end = SliceIndex::map_negative(end, dim_len);

            let num_elems = ((real_end - real_start + step - 1) / step) as usize;
            let _stride = strides[dim] * step as usize;

            new_shape.push(num_elems);
            new_strides.push(_stride);
            new_offset += real_start as usize * strides[dim];
        }

        let result = self.clone();
        result.inner.borrow_mut().shape = new_shape;
        result.inner.borrow_mut().stride = new_strides;
        result.inner.borrow_mut().offset = new_offset;
        result
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data = String::new();
        let shape = self.shape();
        let _t = self.cpu();
        let data = _t.tensor_to_str(data, 0, 0..shape.iter().product());
        let device = self.device();
        let dtype = self.dtype();
        let res = format!("tensor({data}, device={device}, dtype={dtype})");
        write!(f, "{res}")
    }
}
