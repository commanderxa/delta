use crate::{
    Tensor, TensorImpl,
    tensor::element::{TensorElement, TensorNum},
};

/// Create a new tensor from the given data and the shape.
pub fn tensor<T: TensorElement>(data: &[T], shape: &[usize]) -> Tensor<T>
where
    T: TensorElement + Clone,
{
    assert_eq!(
        data.len(),
        shape.iter().product(),
        "The length of the tensor does not match the shape"
    );
    let inner = TensorImpl::from_slice(data, shape);
    Tensor::new(inner)
}

/// Creates a new tensor with the random values between 0 and 1
pub fn randn<T: TensorNum>(shape: &[usize]) -> Tensor<f32> {
    let mut inner = TensorImpl::from_slice(&vec![f32::zero(); shape.iter().product()], shape);
    Tensor::fill_tensor(&mut inner, f32::zero()..f32::one());
    Tensor::new(inner)
}

/// Creates a new tensor, where all the values are 0.
pub fn zeros<T: TensorElement>(shape: &[usize]) -> Tensor<f32> {
    let mut inner = TensorImpl::new(shape);
    inner.data.fill(f32::zero());
    Tensor::new(inner)
}

/// Creates a new tensor like the inputted one, where all the values are 0.
pub fn zeros_like<T: TensorElement>(tensor: &Tensor<T>) -> Tensor<f32> {
    let mut inner = TensorImpl::new(tensor.shape().as_slice());
    inner.data.fill(f32::zero());
    Tensor::new(inner)
}

/// Creates a new tensor, where all the values are 1.
pub fn ones<T: TensorElement>(shape: &[usize]) -> Tensor<f32> {
    let mut inner = TensorImpl::new(shape);
    inner.data.fill(f32::zero());
    Tensor::new(inner)
}

/// Creates a new tensor like the inputted one, where all the values are 1.
pub fn ones_like<T: TensorElement>(tensor: &Tensor<T>) -> Tensor<f32> {
    let mut inner = TensorImpl::new(tensor.shape().as_slice());
    inner.data.fill(f32::zero());
    Tensor::new(inner)
}

/// Creates a new tensor, where the values on the main diagonal are ones
/// and the rest values are zeros.
pub fn eye<T: TensorElement>(n: usize) -> Tensor<f32> {
    assert!(n > 0, "`n` cannot be less than 1.");
    let shape = &[n, n];
    let mut array: Vec<f32> = vec![f32::zero(); n * n];
    for i in 0..n {
        array[i * n + i] = f32::one();
    }
    let inner = TensorImpl::from_slice(&array, shape);
    Tensor::new(inner)
}

/// Generates a tesnor within a given range with a step.
///
/// The `start` is inclusive and the `end` is exclusive.
///
/// Note:
/// * (`end` - `start`) / `step` has to be an integer
/// * if `end` - `start` is negative, then `step` has to be negative
/// * if `end` - `start` is positive, then `step` has to be positive
pub fn arange<T: TensorElement>(start: f32, end: f32, step: f32) -> Tensor<f32> {
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
    let mut data: Vec<f32> = Vec::with_capacity(len as usize);
    for i in 0..len as usize {
        data.push((start + (step * i as f32)).into());
    }
    let inner = TensorImpl::from_slice(&data, &[len as usize]);
    Tensor::new(inner)
}

pub(crate) fn create_grad<T: TensorElement>(shape: &[usize]) -> Tensor<T> {
    let len = shape.iter().product();
    crate::tensor(&vec![T::zero(); len], shape)
}
