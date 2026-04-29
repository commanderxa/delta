use crate::{Tensor, tensor_data::TensorData};

/// Create a new tensor from the given data and the shape.
pub fn tensor(data: &[f64], shape: &[usize]) -> Tensor {
    assert_eq!(
        data.len(),
        shape.iter().product(),
        "The length of the tensor does not match the shape"
    );
    let inner = TensorData::from_f64(data.to_vec());
    Tensor::new(inner, shape)
}

/// Creates a new tensor with the random values between 0 and 1
pub fn randn(shape: &[usize]) -> Tensor {
    let mut inner = TensorData::from_f64(vec![0.0; shape.iter().product()]);
    Tensor::fill_tensor(&mut inner, 0.0..1.0);
    Tensor::new(inner, shape)
}

/// Creates a new tensor, where all the values are 0.
pub fn zeros(shape: &[usize]) -> Tensor {
    let mut inner = TensorData::new(shape);
    inner.data.fill(0.0);
    Tensor::new(inner, shape)
}

/// Creates a new tensor like the inputted one, where all the values are 0.
pub fn zeros_like(tensor: &Tensor) -> Tensor {
    let mut inner = TensorData::new(tensor.shape.as_slice());
    inner.data.fill(0.0);
    Tensor::new(inner, tensor.shape.as_slice())
}

/// Creates a new tensor, where all the values are 1.
pub fn ones(shape: &[usize]) -> Tensor {
    let mut inner = TensorData::new(shape);
    inner.data.fill(1.0);
    Tensor::new(inner, shape)
}

/// Creates a new tensor like the inputted one, where all the values are 1.
pub fn ones_like(tensor: &Tensor) -> Tensor {
    let mut inner = TensorData::new(tensor.shape.as_slice());
    inner.data.fill(1.0);
    Tensor::new(inner, tensor.shape.as_slice())
}

/// Creates a new tensor, where the values on the main diagonal are ones
/// and the rest values are zeros.
pub fn eye(n: usize) -> Tensor {
    assert!(n > 0, "`n` cannot be less than 1.");
    let shape = &[n, n];
    let mut array: Vec<f64> = vec![0.; n * n];
    for i in 0..n {
        array[i * n + i] = 1.;
    }
    let inner = TensorData::from_f64(array);
    Tensor::new(inner, shape)
}

/// Generates a tesnor within a given range with a step.
///
/// The `start` is inclusive and the `end` is exclusive.
///
/// Note:
/// * (`end` - `start`) / `step` has to be an integer
/// * if `end` - `start` is negative, then `step` has to be negative
/// * if `end` - `start` is positive, then `step` has to be positive
pub fn arange(start: f64, end: f64, step: f64) -> Tensor {
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
    Tensor::new(inner, &[len as usize])
}
