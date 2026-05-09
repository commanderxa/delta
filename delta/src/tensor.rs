mod binary_ops_kernel;
pub(crate) mod cast;
pub mod dtype;
pub(crate) mod element;
pub(crate) mod impl_;
pub mod init;
pub mod operations;
pub(crate) mod promote_primitives;
pub(crate) mod storage;

use std::{cell::RefCell, f64::consts::E, fmt::Display, ops::Range, rc::Rc};

use crate::{
    DType, Storage,
    backward::Backward,
    device::Device,
    op::Op,
    tensor::{
        cast::Cast,
        element::{TensorElement, TensorFloat},
        impl_::TensorImpl,
    },
};

type TensorImplRef<T> = Rc<RefCell<TensorImpl<T>>>;

#[derive(Clone, Debug)]
/// # Tensor
///
/// ### Holds the reference to the inner data inside.
///
/// See the documentation for `TensorImpl`.
pub struct Tensor<T: TensorElement> {
    pub(crate) inner: TensorImplRef<T>,
    pub device: Device,
}

impl<T: TensorElement> Tensor<T> {
    /// Creates a new instance of a `Tensor`.
    pub(crate) fn new(inner: TensorImpl<T>) -> Self {
        let device = inner.device();
        Self {
            inner: Rc::new(RefCell::new(inner)),
            device: device,
        }
    }

    /// Returns the shape of the tensor as vector.
    pub(crate) fn shape(&self) -> Vec<usize> {
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

    pub(crate) fn storage(&self) -> Storage<T> {
        self.inner.borrow().data.clone()
    }

    pub fn dtype(&self) -> DType {
        self.inner.borrow().dtype()
    }

    /// Fills the given empty tensor with a values with an inputted range. It
    /// also sets the gradients to 0.
    ///
    /// E.g. if the range is (-1.0..1.0) then each value in the tensor will be
    /// between -1 and 1.
    pub(crate) fn fill_tensor(tensor: &mut TensorImpl<T>, range: Range<T>)
    where
        T: TensorFloat,
    {
        let len = tensor.data.len();
        let values: Vec<T> = (0..len).map(|_| T::random_range(range.clone())).collect();
        tensor.data.replace_data(values);
        tensor.grad = Some(crate::create_grad(&tensor.shape));
    }

    /// Returns an owned copy of tensor strides
    pub(crate) fn stride(&self) -> Vec<usize> {
        self.inner.borrow().stride.clone()
    }

    /// Returns the data of the tensor.
    pub fn data(&self) -> Vec<T> {
        let storage = self.storage().to_vec();
        let stride = self.stride();
        let shape = self.shape();
        let mut mask = vec![0; shape.len()];
        let mut data = vec![T::zero(); self.length()];
        // iterate over storage data
        for d in data.iter_mut() {
            // compute index of past position of data
            *d = storage[stride.iter().zip(&mask).map(|(a, b)| a * b).sum::<usize>()];
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
            let grad = match self.device {
                Device::CPU => crate::create_grad(&self.shape()),
                #[cfg(feature = "cuda")]
                Device::CUDA => crate::create_grad(&self.shape()).cuda(),
            };
            self.inner.borrow_mut().grad = Some(grad);
        } else {
            self.inner.borrow_mut().grad = None;
        }
        self
    }

    pub fn sum(&self, dim: Option<usize>, keepdim: bool) -> Tensor<T> {
        match dim {
            None => {
                let value: T = self.data().iter().fold(T::zero(), |acc, x| acc + *x);
                let shape = if keepdim {
                    vec![1; self.shape().len().max(1)]
                } else {
                    vec![1]
                };
                let inner = TensorImpl::from_op(
                    vec![value],
                    &shape,
                    vec![self.clone()],
                    Op::Sum { dim: None, keepdim },
                    self.device,
                );
                Tensor::new(inner)
            }
            Some(dim) => {
                assert!(dim < self.shape().len(), "sum: dim out of range");

                let input = self.data();
                let outer: usize = self.shape()[..dim].iter().product();
                let reduce: usize = self.shape()[dim];
                let inner: usize = self.shape()[dim + 1..].iter().product();

                let mut out = vec![T::zero(); outer * inner];

                for o in 0..outer {
                    for i in 0..inner {
                        let mut acc = T::zero();
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

                let inner_data = TensorImpl::from_op(
                    out,
                    &out_shape,
                    vec![self.clone()],
                    Op::Sum {
                        dim: Some(dim),
                        keepdim,
                    },
                    self.device,
                );
                Tensor::new(inner_data)
            }
        }
    }

    pub fn mean(&self, dim: Option<usize>, keepdim: bool) -> Tensor<T> {
        match dim {
            None => {
                let n = self.length();
                assert!(n > 0, "mean of empty tensor is undefined");
                let value: T =
                    self.data().iter().fold(T::zero(), |acc, x| acc + *x) / T::from(n).unwrap();
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
                    self.device,
                );
                Tensor::new(inner)
            }
            Some(dim) => {
                assert!(dim < self.shape().len(), "mean: dim out of range");

                let input = self.data();
                let outer: usize = self.shape()[..dim].iter().product();
                let reduce: usize = self.shape()[dim];
                let inner: usize = self.shape()[dim + 1..].iter().product();

                let mut out = vec![T::zero(); outer * inner];

                for o in 0..outer {
                    for i in 0..inner {
                        let mut acc = T::zero();
                        for r in 0..reduce {
                            let idx = o * reduce * inner + r * inner + i;
                            acc = acc + input[idx];
                        }
                        out[o * inner + i] = acc / T::from(reduce).unwrap();
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
                    self.device,
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
        let mut data = vec![T::zero(); self.length()];
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
        t.inner.borrow_mut().data.replace_data(data);
        t.inner.borrow_mut().shape = shape.to_vec();
        t.inner.borrow_mut().stride = stride;
        t
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
    pub fn exp<U: TensorFloat>(&self) -> Tensor<U>
    where
        T: Cast<U>,
    {
        let mut data = self.data();
        let mut data: Vec<U> = self
            .data()
            .into_iter()
            .map(|x| {
                let u: U = x.cast();
                u.exp()
            })
            .collect();
        let inner = TensorImpl::from_op(
            data,
            &self.shape(),
            vec![self.clone()],
            Op::Exp(self.clone()),
            self.device,
        );
        Tensor::new(inner)
    }

    pub fn contiguous(&self) -> Self {
        // already contiguous — no work needed
        if self.is_contiguous() {
            return self.clone();
        }

        let old_data = self.cpu().storage();
        let old_data = old_data.as_cpu();
        let shape = self.shape();
        let strides = self.stride();

        let total: usize = shape.iter().product();
        let mut new_data = vec![T::zero(); total];

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

            new_data[flat_idx] = old_data[src_idx];
        }

        // compute new contiguous row-major strides
        let mut new_strides = vec![1usize; shape.len()];
        for d in (0..shape.len() - 1).rev() {
            new_strides[d] = new_strides[d + 1] * shape[d + 1];
        }

        let device = self.device;

        let inner = match device {
            Device::CPU => TensorImpl::from_slice(&new_data, &shape),
            #[cfg(feature = "cuda")]
            Device::CUDA => {
                let _inner = &self.inner.borrow();
                TensorImpl::from_cuda(
                    Storage::to_cuda_slice(&new_data),
                    &shape,
                    _inner._prev.clone(),
                    _inner._op.clone(),
                )
            }
        };

        Self {
            inner: Rc::new(RefCell::new(inner)),
            device: device,
        }
    }

    pub(crate) fn is_contiguous(&self) -> bool {
        let mut expected = 1usize;
        for d in (0..self.shape().len()).rev() {
            if self.stride()[d] != expected {
                return false;
            }
            expected *= self.shape()[d];
        }
        true
    }

    /// Add a `Vec<T>` value to the gradient inside the `TensorImpl`.
    pub(crate) fn add_to_grad(&self, tensor: Tensor<T>) {
        let mut t: std::cell::RefMut<'_, TensorImpl<T>> = self.inner.borrow_mut();
        // t.grad = Some(Storage::from_slice(data, self.device.clone()));
        t.grad = Some(t.grad.clone().unwrap() + tensor);
    }

    /// Returns the gradient vector.
    pub fn grad(&self) -> Option<Tensor<T>> {
        self.inner.borrow().grad.clone()
    }

    /// Replace current data inside the tensor with new `data`
    pub(crate) fn set_data(&self, data: Vec<T>) {
        self.inner.borrow_mut().data.replace_data(data);
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
    pub(crate) fn multicast_op(a: Tensor<T>, b: Tensor<T>, op: Op<T>) -> Self {
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

        let device = a.device;

        let mut mask = vec![0; a.shape().len()];
        let mut data = vec![T::zero(); a.length()];
        // iterate over storage data
        for d in data.iter_mut() {
            // compute index of past position of data
            let a_i = a.data()[a
                .stride()
                .iter()
                .zip(&mask)
                .map(|(a, b)| a * b)
                .sum::<usize>()];
            let b_i = b.data()[b
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
        let inner = TensorImpl::from_op(data, &a.shape(), vec![a, b], op, device);
        Self::new(inner)
    }

    #[cfg(feature = "cuda")]
    pub fn cuda(&self) -> Tensor<T> {
        let inner = self.inner.borrow();

        let new_inner = TensorImpl {
            data: inner.data.to_cuda(),
            shape: self.shape(),
            stride: self.stride(),
            grad: inner.grad.as_ref().map(|g| g.cuda()),
            _prev: inner._prev.clone(),
            _op: inner._op.clone(),
        };

        Tensor {
            inner: Rc::new(RefCell::new(new_inner)),
            device: Device::CUDA,
        }
    }

    pub fn cpu(&self) -> Tensor<T> {
        match self.device {
            Device::CPU => self.to_owned(),
            #[cfg(feature = "cuda")]
            Device::CUDA => {
                let new_storage = self.inner.borrow().data.to_cpu();
                let mut new_data = TensorImpl::from_slice(&[], &self.shape());
                new_data.data = new_storage;
                new_data.grad = self.inner.borrow().grad.clone();
                Self {
                    inner: Rc::new(RefCell::new(new_data)),
                    device: Device::CPU,
                }
            }
        }
    }

    /// Concatenates a slice of tensors along the given dimension.
    /// All tensors must have the same shape except in the `dim` axis.
    pub fn cat(tensors: &[Tensor<T>], dim: isize) -> Self {
        assert!(!tensors.is_empty(), "cat: need at least one tensor");
        assert!(dim >= -1, "cat: `dim` cannot be negative integer");

        let ndim = tensors[0].shape().len();

        let dim: usize = if dim == -1 {
            tensors[0].shape().len() - 1
        } else {
            dim as usize
        };

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
        let mut data = Vec::with_capacity(out_shape.iter().product());

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

        let inner_data = TensorImpl::from_slice(&data, &out_shape);
        Tensor::new(inner_data)
    }

    pub fn cast<U: TensorElement>(self) -> Tensor<U>
    where
        T: Cast<U>,
    {
        let data: Vec<U> = self.storage().iter().map(T::cast).collect();
        crate::tensor(&data, &self.shape())
    }
}

impl<T: TensorFloat> Tensor<T> {
    /// Backward
    ///
    /// Computes the gradients of all the tensors that have been interacting and
    /// have `requires_grad` set to `true`.
    pub fn backward(&self) {
        assert!(
            self.length() == 1,
            "grad can be implicitly created only for scalar outputs"
        );

        self.add_to_grad(crate::tensor(&[<T as TensorElement>::one()], &[1]));
        self._backward()
    }

    /// Backward private
    ///
    /// Evokes Backward function in all components of the computational graph
    fn _backward(&self) {
        let t = self.inner.borrow();
        if t.grad.is_some() && t._op.is_some() {
            t._op.as_ref().unwrap().backward(&self);
            if !t._prev.is_empty() {
                for prev in t._prev.clone() {
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
        let data = self.data().iter().map(|a| a.powi(n)).collect::<Vec<T>>();
        let shape = self.shape();
        let device = self.device;
        let inner = TensorImpl::from_op(data, &shape, vec![self.clone()], Op::Pow(n), device);
        Self::new(inner)
    }

    pub fn log(&self) -> Self {
        let mut data = self.data();
        for item in data.iter_mut() {
            *item = (*item).ln();
        }
        let inner = TensorImpl::from_op(
            data,
            &self.shape(),
            vec![self.clone()],
            Op::Exp(self.clone()),
            self.device,
        );
        Tensor::new(inner)
    }

    /// Converts the tensor to a `String`, so that it can be printed.
    fn tensor_to_str(&self, tensor_str: String, level: usize, range: Range<usize>) -> String {
        let mut width = 1;
        for i in self.data() {
            let s = (i.floor() as i64).to_string();
            if s.len() > width {
                width = s.len();
            }
        }
        width += 5;
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
        // the length of the range
        let len: usize = range.end - range.start;
        // the current dimension from the shape
        let dim = self.shape()[level];
        // convolution to iterate over the data
        let conv = len / dim;
        // the length of shape vector
        let shape_size = self.shape().len();
        let item = self.data();
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
}

impl<T: TensorElement> PartialEq for Tensor<T> {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<T: TensorElement> Display for Tensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data = String::new();
        let shape = self.shape();
        let data = self.tensor_to_str(data, 0, 0..shape.iter().product());
        let res = format!("Tensor({data})");
        write!(f, "{res}")
    }
}

// MACROS

#[macro_export]
#[doc(hidden)]
macro_rules! __tensor_shape {
    ([ $( $elem:tt ),* $(,)? ]) => {{
        let mut shape = vec![0usize];
        $(
            let child = $crate::__tensor_shape!($elem);
            if shape[0] == 0 {
                shape.extend_from_slice(&child);
            } else {
                assert_eq!(
                    &shape[1..],
                    child.as_slice(),
                    "tensor! elements must have consistent inner shapes"
                );
            }
            shape[0] += 1;
        )*
        shape
    }};

    ( $scalar:expr ) => {
        Vec::<usize>::new()
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! __tensor_flatten {
    ( $out:ident ; [ $( $elem:tt ),* $(,)? ] ) => {{
        $(
            $crate::__tensor_flatten!($out ; $elem);
        )*
    }};

    ( $out:ident ; $scalar:expr ) => {{
        $out.push(($scalar) as f64);
    }};
}

#[macro_export]
macro_rules! tensor {
    ($data:tt) => {{
        let shape = $crate::__tensor_shape!($data);
        let mut flat = Vec::new();
        $crate::__tensor_flatten!(flat; $data);
        $crate::tensor(&flat, &shape)
    }};
}

#[macro_export]
macro_rules! randn {
    ($($element:expr),+) => {{
        use rand::Rng;
        // get shape
        let mut shape = Vec::new();
        // fill the shape
        $(shape.push($element);)*;
        // pass the shape to the `randn` method
        delta::randn(&shape)
    }};
    ($($element:expr,)*) => {{
        $crate::tensor::randn![$($element),*]
    }};
}
