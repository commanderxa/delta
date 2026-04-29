use crate::{op::Op, Tensor};

#[derive(Debug, PartialEq)]
/// # TensorData
///
/// ### This type holds the data of the tensor, and is an actual data.
///
/// `Tensor` holds a reference to it, allowing to use the same data.\
/// See the documentation for `Tensor`.
pub(crate) struct TensorData {
    // stored data
    // it is a single vector that is viewed regarding the shape
    pub data: Vec<f64>,
    // gradients
    pub grad: Option<Vec<f64>>,
    // vector of the tensors that were used to produce this tensor
    pub _prev: Vec<Tensor>,
    // operation that was used to produce this tensor
    pub _op: Option<Op>,
}

#[allow(dead_code)]
impl TensorData {
    /// Create a new instance of the TensorData.
    pub fn new(shape: &[usize]) -> Self {
        let len = shape.iter().product();
        Self {
            data: vec![0.0; len],
            grad: Some(vec![0.0; len]),
            _prev: vec![],
            _op: None,
        }
    }

    /// Sets gradients to 0
    pub fn zero_grad(&mut self) {
        let grad = vec![0.0; self.data.len()];
        self.grad = Some(grad);
    }

    /// Sets gradients to `None`
    pub fn grad_none(&mut self) {
        self.grad = None;
    }

    pub fn set_grad(&mut self, grad: Vec<f64>) {
        self.grad = Some(grad);
    }

    /// Creates a new instance of the TensorData from a Vector.
    pub fn from_f64(data: Vec<f64>) -> Self {
        let grad = vec![0.0; data.len()];
        // Self::fill_grad(&mut grad);
        Self {
            data,
            grad: Some(grad),
            _prev: vec![],
            _op: None,
        }
    }

    /// Creates a new instance of the TensorData produced by any `Op`.
    pub fn from_op(data: Vec<f64>, prev: Vec<Tensor>, op: Op) -> Self {
        let grad = vec![0.0; data.len()];
        // Self::fill_grad(&mut grad);
        Self {
            data,
            grad: Some(grad),
            _prev: prev,
            _op: Some(op),
        }
    }
}
