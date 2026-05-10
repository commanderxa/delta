use crate::{
    Tensor, TensorImpl,
    device::Device,
    tensor::repr::{FloatTensorRepr, TensorRepr},
};

/// Create a new tensor from the given data and the shape.
pub fn tensor<T: TensorRepr>(data: &[T], shape: &[usize], device: Device) -> Tensor {
    assert_eq!(
        data.len(),
        shape.iter().product(),
        "The length of the tensor does not match the shape"
    );
    let inner = TensorImpl::from_slice(&data, shape, device);
    Tensor::new(inner)
}

/// Creates a new tensor with the random values between 0 and 1
pub fn randn<T: FloatTensorRepr>(shape: &[usize], device: Device) -> Tensor {
    let mut inner = TensorImpl::from_slice(
        &vec![<T as TensorRepr>::zero(); shape.iter().product()],
        shape,
        device,
    );
    Tensor::fill_tensor(
        &mut inner,
        <T as TensorRepr>::zero()..<T as TensorRepr>::one(),
    );
    Tensor::new(inner)
}

/// Creates a new tensor, where all the values are 0.
pub fn zeros<T: TensorRepr>(shape: &[usize], device: Device) -> Tensor {
    let mut inner = TensorImpl::new::<f32>(shape, device);
    inner.data.fill(f32::zero());
    Tensor::new(inner)
}

/// Creates a new tensor like the inputted one, where all the values are 0.
pub fn zeros_like<T: TensorRepr>(tensor: &Tensor, device: Device) -> Tensor {
    let mut inner = TensorImpl::new::<f32>(&tensor.shape(), device);
    inner.data.fill(f32::zero());
    Tensor::new(inner)
}

/// Creates a new tensor, where all the values are 1.
pub fn ones<T: TensorRepr>(shape: &[usize], device: Device) -> Tensor {
    let mut inner = TensorImpl::new::<f32>(shape, device);
    inner.data.fill(f32::zero());
    Tensor::new(inner)
}

/// Creates a new tensor like the inputted one, where all the values are 1.
pub fn ones_like<T: TensorRepr>(tensor: &Tensor, device: Device) -> Tensor {
    let mut inner = TensorImpl::new::<f32>(&tensor.shape(), device);
    inner.data.fill(f32::zero());
    Tensor::new(inner)
}

/// Creates a new tensor, where the values on the main diagonal are ones
/// and the rest values are zeros.
pub fn eye<T: TensorRepr>(n: usize, device: Device) -> Tensor {
    assert!(n > 0, "`n` cannot be less than 1.");
    let shape = &[n, n];
    let mut array: Vec<T> = vec![T::zero(); n * n];
    for i in 0..n {
        array[i * n + i] = T::one();
    }
    let inner = TensorImpl::from_slice(&array, shape, device);
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
pub fn arange<T: TensorRepr>(start: f32, end: f32, step: f32, device: Device) -> Tensor {
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
    let mut data: Vec<T> = Vec::with_capacity(len as usize);
    for i in 0..len as usize {
        let val: f32 = start + step * i as f32;
        data.push(T::cast_from(val));
    }
    let inner = TensorImpl::from_slice(&data, &[len as usize], device);
    Tensor::new(inner)
}

pub(crate) fn create_grad<T: TensorRepr>(shape: &[usize], device: Device) -> Tensor {
    let len = shape.iter().product();
    crate::tensor(&vec![T::zero(); len], shape, device)
}
