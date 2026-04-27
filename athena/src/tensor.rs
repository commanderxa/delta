use std::{
    cell::RefCell,
    f64::consts::E,
    fmt::Display,
    ops::{Add, Div, Mul, Neg, Range, Sub},
    rc::Rc,
};

use crate::{backward::Backward, op::Op, tensor_data::TensorData};

#[derive(Clone, Debug, PartialEq)]
/// # Tensor
///
/// ### Holds the reference to the inner data inside.
///
/// See the documentation for `TensorData`.
pub struct Tensor {
    pub(crate) inner: Rc<RefCell<TensorData>>,
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
}

impl Tensor {
    /// Creates a new instance of a `Tensor`.
    pub(crate) fn new(inner: TensorData, shape: &[usize]) -> Self {
        let mut stride = vec![1; shape.len()];
        // compute stride
        for i in (0..shape.len() - 1).rev() {
            stride[i] = shape[i + 1] * stride[i + 1];
        }
        Self {
            inner: Rc::new(RefCell::new(inner)),
            shape: shape.to_vec(),
            stride,
        }
    }

    /// Creates a new tensor with the random values between 0 and 1
    pub fn randn(shape: &[usize]) -> Self {
        let mut inner = TensorData::from_f64(vec![0.0; shape.iter().product()]);
        Self::fill_tensor(&mut inner, 0.0..1.0);
        Self::new(inner, shape)
    }

    /// Creates a new tensor, where all the values are 0.
    pub fn zeros(shape: &[usize]) -> Self {
        let mut inner = TensorData::new(shape);
        inner.data.fill(0.0);
        Self::new(inner, shape)
    }

    /// Creates a new tensor like the inputted one, where all the values are 0.
    pub fn zeros_like(tensor: &Tensor) -> Self {
        let mut inner = TensorData::new(tensor.shape.as_slice());
        inner.data.fill(0.0);
        Self::new(inner, tensor.shape.as_slice())
    }

    /// Creates a new tensor, where all the values are 1.
    pub fn ones(shape: &[usize]) -> Self {
        let mut inner = TensorData::new(shape);
        inner.data.fill(1.0);
        Self::new(inner, shape)
    }

    /// Creates a new tensor like the inputted one, where all the values are 1.
    pub fn ones_like(tensor: &Tensor) -> Self {
        let mut inner = TensorData::new(tensor.shape.as_slice());
        inner.data.fill(1.0);
        Self::new(inner, tensor.shape.as_slice())
    }

    #[allow(clippy::self_named_constructors)]
    /// Create a new tensor from the given data and the shape.
    pub fn tensor(data: &[f64], shape: &[usize]) -> Self {
        assert_eq!(
            data.len(),
            shape.iter().product(),
            "The length of the tensor does not match the shape"
        );
        let inner = TensorData::from_f64(data.to_vec());
        Self::new(inner, shape)
    }

    /// Generates a tesnor within a given range with a step.
    ///
    /// The `start` is inclusive and the `end` is exclusive.
    ///
    /// Note:
    /// * (`end` - `start`) / `step` has to be an integer
    /// * if `end` - `start` is negative, then `step` has to be negative
    /// * if `end` - `start` is positive, then `step` has to be positive
    pub fn arange(start: f64, end: f64, step: f64) -> Self {
        let len = (end - start) / step;
        // necessary cheks
        assert_eq!(
            len.fract(),
            0.,
            "Cannot generate a range, since the length of the tensor is not an integer, try to use other parameters"
        );
        if end - start < 1.0 && step > 0.0 {
            panic!("Cannot generate a range, since the step is wrong, try to make it negative");
        } else if end - start > 1.0 && step < 0.0 {
            panic!("Cannot generate a range, since the step is wrong, try to make it positive");
        }
        // new tensor
        let mut data = Vec::with_capacity(len as usize);
        for i in 0..len as usize {
            data.push(start + (step * i as f64));
        }
        let inner = TensorData::from_f64(data);
        Self::new(inner, &[len as usize])
    }

    /// Returns the shape of the tensor as vector.
    pub(crate) fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    /// Returns the total length of the vector. It obtains the length by taking
    /// the product of the shape of the tensor.
    ///
    /// E.g. if the shape of the tensor is (3, 2, 3), then the length is
    /// (3 * 2 * 3) => 18.
    pub fn length(&self) -> usize {
        self.shape().iter().product()
    }

    /// Returns the data of `Tensor` as it is stored.
    pub fn storage(&self) -> Vec<f64> {
        self.inner.borrow().data.clone()
    }

    /// Fills the given empty tensor with a values with an inputted range. It
    /// also sets the gradients to 0.
    ///
    /// E.g. if the range is (-1.0..1.0) then each value in the tensor will be
    /// between -1 and 1.
    fn fill_tensor(tensor: &mut TensorData, range: Range<f64>) {
        for i in 0..tensor.data.len() {
            let data = rand::random_range(range.clone());
            tensor.data[i] = data;
            tensor.grad.as_mut().unwrap().push(0.0);
        }
    }

    /// Returns an owned copy of tensor strides
    fn stride(&self) -> Vec<usize> {
        self.stride.clone()
    }

    /// Returns the data of the tensor.
    pub fn item(&self) -> Vec<f64> {
        let storage = self.storage();
        let stride = self.stride();
        let shape = self.shape();
        let mut mask = vec![0; shape.len()];
        let mut data = vec![0.0; self.length()];
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
            self.inner.borrow_mut().grad = Some(vec![0.0; self.length()]);
        } else {
            self.inner.borrow_mut().grad = None;
        }
        self
    }

    pub fn sum(&self, dim: Option<usize>, keepdim: bool) -> Tensor {
        match dim {
            None => {
                let value: f64 = self.item().iter().sum();
                let shape = if keepdim {
                    vec![1; self.shape.len().max(1)]
                } else {
                    vec![1]
                };
                let inner = TensorData::from_op(
                    vec![value],
                    vec![self.clone()],
                    Op::Sum { dim: None, keepdim },
                );
                Tensor::new(inner, &shape)
            }
            Some(dim) => {
                assert!(dim < self.shape.len(), "sum: dim out of range");

                let input = self.item();
                let outer: usize = self.shape[..dim].iter().product();
                let reduce: usize = self.shape[dim];
                let inner: usize = self.shape[dim + 1..].iter().product();

                let mut out = vec![0.0; outer * inner];

                for o in 0..outer {
                    for i in 0..inner {
                        let mut acc = 0.0;
                        for r in 0..reduce {
                            let idx = o * reduce * inner + r * inner + i;
                            acc += input[idx];
                        }
                        out[o * inner + i] = acc;
                    }
                }

                let mut out_shape = self.shape.clone();
                if keepdim {
                    out_shape[dim] = 1;
                } else {
                    out_shape.remove(dim);
                    if out_shape.is_empty() {
                        out_shape.push(1);
                    }
                }

                let inner_data = TensorData::from_op(
                    out,
                    vec![self.clone()],
                    Op::Sum {
                        dim: Some(dim),
                        keepdim,
                    },
                );
                Tensor::new(inner_data, &out_shape)
            }
        }
    }

    pub fn mean(&self, dim: Option<usize>, keepdim: bool) -> Tensor {
        match dim {
            None => {
                let n = self.length();
                assert!(n > 0, "mean of empty tensor is undefined");
                let value: f64 = self.item().iter().sum::<f64>() / n as f64;
                let shape = if keepdim {
                    vec![1; self.shape.len().max(1)]
                } else {
                    vec![1]
                };
                let inner = TensorData::from_op(
                    vec![value],
                    vec![self.clone()],
                    Op::Mean {
                        dim: None,
                        keepdim,
                        count: n,
                    },
                );
                Tensor::new(inner, &shape)
            }
            Some(dim) => {
                assert!(dim < self.shape.len(), "mean: dim out of range");

                let input = self.item();
                let outer: usize = self.shape[..dim].iter().product();
                let reduce: usize = self.shape[dim];
                let inner: usize = self.shape[dim + 1..].iter().product();

                let mut out = vec![0.0; outer * inner];

                for o in 0..outer {
                    for i in 0..inner {
                        let mut acc = 0.0;
                        for r in 0..reduce {
                            let idx = o * reduce * inner + r * inner + i;
                            acc += input[idx];
                        }
                        out[o * inner + i] = acc / reduce as f64;
                    }
                }

                let mut out_shape = self.shape.clone();
                if keepdim {
                    out_shape[dim] = 1;
                } else {
                    out_shape.remove(dim);
                    if out_shape.is_empty() {
                        out_shape.push(1);
                    }
                }

                let inner_data = TensorData::from_op(
                    out,
                    vec![self.clone()],
                    Op::Mean {
                        dim: Some(dim),
                        keepdim,
                        count: reduce,
                    },
                );
                Tensor::new(inner_data, &out_shape)
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
        let mut t = self.clone();
        // transpose tensor of 2 and more dimensions
        if t.shape().len() >= 2 {
            t.shape.swap(dim0, dim1);
            t.stride.swap(dim0, dim1);
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
        let mut t = self.clone();
        if self.stride.iter().product::<usize>() == 0 {
            panic!(
                "view size is not compatible with size and stride of input tensor. Use .reshape(...) instead"
            );
        }
        let mut stride = vec![1; shape.len()];
        // compute stride
        for i in (0..shape.len() - 1).rev() {
            stride[i] = shape[i + 1] * stride[i + 1];
        }
        t.stride = stride;
        t.shape = shape.to_vec();
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
        if self.stride.iter().product::<usize>() > 0 {
            return self.view(shape);
        }
        let mut mask = vec![0; self.shape.len()];
        let mut data = vec![0.0; self.length()];
        // iterate over storage data
        for d in data.iter_mut() {
            // compute index of past position of data
            *d = self.storage()[self
                .stride
                .iter()
                .zip(&mask)
                .map(|(a, b)| a * b)
                .sum::<usize>()];
            // iterate over shape
            for j in (0..self.shape.len()).rev() {
                // skip the properly filled dims
                if self.shape[j] - 1 == mask[j] {
                    continue;
                }
                // increment the necessary mask dim
                mask[j] += 1;
                // set to 0 all prevous shape dims
                for k in ((j + 1)..self.shape.len()).rev() {
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
        let mut t = self.clone();
        t.inner.borrow_mut().data = data;
        t.shape = shape.to_vec();
        t.stride = stride;
        t
    }

    /// Inserts a dimension of size 1 at a specified location in shape.
    pub fn unsqueeze(&self, dim: usize) -> Self {
        assert!(
            dim <= self.shape.len(),
            "Dimension out of range (expected range of [0, {}])",
            self.shape.len()
        );
        let mut t = self.clone();
        t.shape.insert(dim, 1);
        let mut replica = 1;
        if dim < self.shape.len() {
            replica = t.stride[dim];
        }
        t.stride.insert(dim, replica);
        t
    }

    /// Returns a tensor with all specified dimensions of shape of size 1 removed.
    ///
    /// * If `dim` is empty it performs removal across the whole shape.
    /// * If `dim` contains dimensions, then it only considers them.
    ///
    /// All the specified dimensions that are more than 1 it leaves as it is.
    pub fn squeeze(&self, dim: &[usize]) -> Self {
        let mut t = self.clone();
        if dim.is_empty() {
            for d in (0..t.shape.len()).rev() {
                if t.shape[d] == 1 {
                    t.shape.remove(d);
                    t.stride.remove(d);
                }
            }
        } else {
            for d in (0..dim.len()).rev() {
                if t.shape[d] == 1 {
                    t.shape.remove(d);
                    t.stride.remove(d);
                }
            }
        }
        t
    }

    /// Exponents each value of the `Tensor`.
    ///
    /// `exp(x)` => `e^(x)`.
    pub fn exp(&self) -> Tensor {
        let mut data = self.item();
        for item in data.iter_mut() {
            *item = E.powf(*item);
        }
        let inner = TensorData::from_op(data, vec![self.clone()], Op::Exp(self.clone()));
        Tensor::new(inner, &self.shape)
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
        let mut t = self.clone();
        let mut _old_shape = self.shape();
        // check if batch dims have to be added in th front
        let dims_to_add = new_shape.len() - _old_shape.len();
        let mut old_shape: Vec<usize> = vec![1; dims_to_add];
        // push neccessary front batch dims
        for _ in 0..dims_to_add {
            t.stride.insert(0, t.stride[0]);
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
                t.stride[i] = 0;
            }
        }

        // change tensor properties
        t.shape = new_shape.to_vec();
        t
    }

    /// Add a `Vec<f64>` value to the gradient inside the `TensorData`.
    pub(crate) fn add_to_grad(&self, data: Vec<f64>) {
        let mut t = self.inner.borrow_mut();
        t.grad = Some(
            t.grad
                .clone()
                .unwrap()
                .iter()
                .zip(data)
                .map(|(a, b)| a + b)
                .collect(),
        );
    }

    /// Returns the gradient vector.
    pub fn grad(&self) -> Option<Vec<f64>> {
        self.inner.borrow().grad.clone()
    }

    /// Replace current data inside the tensor with new `data`
    pub(crate) fn set_data(&self, data: Vec<f64>) {
        self.inner.borrow_mut().data = data;
    }

    /// Powers the `Tensor`
    ///
    /// Accepts `n` integer in which the `Tensor` will be powered.
    ///
    /// For backpropagation it stores the `n` inside the `Op::Pow(n)`.
    pub fn pow(self, n: i32) -> Tensor {
        let data = self.item().iter().map(|a| a.powi(n)).collect::<Vec<f64>>();
        let shape = self.shape();
        let inner = TensorData::from_op(data, vec![self], Op::Pow(n));
        Self::new(inner, &shape)
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

        self.add_to_grad(vec![1.0]);
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
    fn multicast_op(a: Tensor, b: Tensor, op: Op) -> Tensor {
        let mut a = a;
        let mut b = b;
        // check whether to expand any of variables
        if a.shape != b.shape {
            // if `a` tensor is bigger => expand `b`
            // else expand `a`
            if a.length() > b.length() {
                b = b.expand(&a.shape);
            } else {
                a = a.expand(&b.shape);
            }
        }

        let mut mask = vec![0; a.shape.len()];
        let mut data = vec![0.0; a.length()];
        // iterate over storage data
        for d in data.iter_mut() {
            // compute index of past position of data
            let a_i = a.storage()[a
                .stride
                .iter()
                .zip(&mask)
                .map(|(a, b)| a * b)
                .sum::<usize>()];
            let b_i = b.storage()[b
                .stride
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
            for j in (0..a.shape.len()).rev() {
                if a.shape[j] - 1 == mask[j] {
                    continue;
                }
                mask[j] += 1;
                for k in ((j + 1)..a.shape.len()).rev() {
                    mask[k] = 0;
                }
                break;
            }
        }
        let shape = a.shape();
        let inner = TensorData::from_op(data, vec![a, b], op);
        Self::new(inner, &shape)
    }

    /// Converts the tensor to a `String`, so that it can be printed.
    fn tensor_to_str(&self, tensor_str: String, level: usize, range: Range<usize>) -> String {
        let mut width = 1;
        for i in self.storage() {
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
        let item = self.item();
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

    /// Concatenates a slice of tensors along the given dimension.
    /// All tensors must have the same shape except in the `dim` axis.
    pub fn cat(tensors: &[Tensor], dim: usize) -> Self {
        assert!(!tensors.is_empty(), "cat: need at least one tensor");

        let ndim = tensors[0].shape.len();
        assert!(
            dim < ndim,
            "cat: dim {} out of range for {}-D tensor",
            dim,
            ndim
        );

        // Validate all shapes match except on `dim`
        for t in tensors.iter().skip(1) {
            assert_eq!(
                t.shape.len(),
                ndim,
                "cat: all tensors must have same number of dims"
            );
            for d in 0..ndim {
                if d != dim {
                    assert_eq!(
                        t.shape[d], tensors[0].shape[d],
                        "cat: shape mismatch on dim {}",
                        d
                    );
                }
            }
        }

        // Compute output shape
        let mut out_shape = tensors[0].shape.clone();
        out_shape[dim] = tensors.iter().map(|t| t.shape[dim]).sum();

        // Collect contiguous data from each tensor
        let mut data = Vec::with_capacity(out_shape.iter().product());

        // Iterate over all positions except `dim`, then gather slices
        let outer: usize = out_shape[..dim].iter().product();
        let inner: usize = out_shape[dim + 1..].iter().product();

        for o in 0..outer {
            for t in tensors {
                let slice_len = t.shape[dim] * inner;
                let offset = o * slice_len;
                let items = t.item();
                data.extend_from_slice(&items[offset..offset + slice_len]);
            }
        }

        let inner_data = TensorData::from_f64(data);
        Tensor::new(inner_data, &out_shape)
    }
}

impl Add for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Self) -> Self::Output {
        Self::multicast_op(self, rhs, Op::Add)
    }
}

impl Sub for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::multicast_op(self, rhs, Op::Sub)
    }
}

impl Mul for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::multicast_op(self, rhs, Op::Mul)
    }
}

impl Div for Tensor {
    type Output = Tensor;

    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.pow(-1)
    }
}

impl Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        self * -1.0
    }
}

macro_rules! impl_scalar_op {
    ($trait:ident, $method:ident, $op:tt, $($t:ty),+) => {
        $(
            impl $trait<$t> for Tensor {
                type Output = Tensor;
                fn $method(self, rhs: $t) -> Self::Output {
                    for i in 0..self.length() {
                        self.inner.borrow_mut().data[i] $op rhs as f64;
                    }
                    self
                }
            }
        )+
    };
}

impl_scalar_op!(Add, add, +=, i32, i64, f32, f64, usize);
impl_scalar_op!(Sub, sub, -=, i32, i64, f32, f64, usize);
impl_scalar_op!(Mul, mul, *=, i32, i64, f32, f64, usize);
impl_scalar_op!(Div, div, /=, i32, i64, f32, f64, usize);

impl Display for Tensor {
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
    ( [ $( $elem:tt ),* $(,)? ] ) => {{
        let mut shape = vec![0usize];
        $(
            let child = $crate::__tensor_shape!($elem);
            if shape.len() == 1 {
                shape.extend_from_slice(&child);
            } else {
                assert_eq!(
                    &shape[1..],
                    child.as_slice(),
                    "tensor! rows must have consistent inner shapes"
                );
            }
            shape[0] += 1;
        )*
        shape
    }};
    ( & [ $( $elem:tt ),* $(,)? ] ) => {
        $crate::__tensor_shape!([ $( $elem ),* ])
    };
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
    ( $out:ident ; & [ $( $elem:tt ),* $(,)? ] ) => {{
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
        $crate::Tensor::tensor(&flat, &shape)
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
        Tensor::randn(&shape)
    }};
    ($($element:expr,)*) => {{
        $crate::tensor::randn![$($element),*]
    }};
}

#[macro_export]
#[doc(hidden)]
macro_rules! count_shape {
    (@COUNT; $($element:expr),*) => {
        <[()]>::len(&[$($crate::count_shape![@SUBST; $element]),*])
    };
    (@SUBST; $_element:expr) => { () };
}
