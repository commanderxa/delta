use half::{bf16, f16};
use crate::f8;

use crate::{
    DType, Device, Tensor, TensorImpl, get_default_dtype,
    tensor::{cast::Cast, repr::TensorRepr},
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
pub fn randn(shape: &[usize], device: Device) -> Tensor {
    let inner = match get_default_dtype() {
        DType::Float8 => {
            let mut inner =
                TensorImpl::from_slice(&vec![f8::zero(); shape.iter().product()], shape, device);
            Tensor::fill_tensor(&mut inner, f8::zero()..f8::one());
            inner
        }
        DType::Float16 => {
            let mut inner =
                TensorImpl::from_slice(&vec![f16::zero(); shape.iter().product()], shape, device);
            Tensor::fill_tensor(&mut inner, f16::zero()..f16::one());
            inner
        }
        DType::BFloat16 => {
            let mut inner =
                TensorImpl::from_slice(&vec![bf16::zero(); shape.iter().product()], shape, device);
            Tensor::fill_tensor(&mut inner, bf16::zero()..bf16::one());
            inner
        }
        DType::Float32 => {
            let mut inner =
                TensorImpl::from_slice(&vec![f32::zero(); shape.iter().product()], shape, device);
            Tensor::fill_tensor(&mut inner, f32::zero()..f32::one());
            inner
        }
        DType::Float64 => {
            let mut inner =
                TensorImpl::from_slice(&vec![f64::zero(); shape.iter().product()], shape, device);
            Tensor::fill_tensor(&mut inner, f64::zero()..f64::one());
            inner
        }
        DType::Int8 => unimplemented!(),
        DType::Int16 => unimplemented!(),
        DType::Int32 => unimplemented!(),
        DType::Int64 => unimplemented!(),
        DType::Bool => unimplemented!(),
    };
    Tensor::new(inner)
}

fn _zeros<T: TensorRepr>(shape: &[usize], device: Device) -> Tensor {
    let mut inner = TensorImpl::new::<T>(shape, device);
    inner.data.fill(T::zero());
    Tensor::new(inner)
}

/// Creates a new tensor, where all the values are 0.
pub fn zeros(shape: &[usize], dtype: DType, device: Device) -> Tensor {
    match dtype {
        DType::Float8 => _zeros::<f8>(shape, device),
        DType::Float16 => _zeros::<f16>(shape, device),
        DType::BFloat16 => _zeros::<bf16>(shape, device),
        DType::Float32 => _zeros::<f32>(shape, device),
        DType::Float64 => _zeros::<f64>(shape, device),
        DType::Int8 => _zeros::<i8>(shape, device),
        DType::Int16 => _zeros::<i16>(shape, device),
        DType::Int32 => _zeros::<i32>(shape, device),
        DType::Int64 => _zeros::<i64>(shape, device),
        _ => _zeros::<f32>(shape, device), // fallback
    }
}

/// Creates a new tensor like the inputted one, where all the values are 0.
pub fn zeros_like(tensor: &Tensor, dtype: DType, device: Device) -> Tensor {
    zeros(&tensor.shape(), dtype, device)
}

fn _ones<T: TensorRepr>(shape: &[usize], device: Device) -> Tensor {
    let mut inner = TensorImpl::new::<T>(shape, device);
    inner.data.fill(T::one());
    Tensor::new(inner)
}

/// Creates a new tensor, where all the values are 1.
pub fn ones(shape: &[usize], dtype: DType, device: Device) -> Tensor {
    match dtype {
        DType::Float8 => _ones::<f8>(shape, device),
        DType::Float16 => _ones::<f16>(shape, device),
        DType::BFloat16 => _ones::<bf16>(shape, device),
        DType::Float32 => _ones::<f32>(shape, device),
        DType::Float64 => _ones::<f64>(shape, device),
        DType::Int8 => _ones::<i8>(shape, device),
        DType::Int16 => _ones::<i16>(shape, device),
        DType::Int32 => _ones::<i32>(shape, device),
        DType::Int64 => _ones::<i64>(shape, device),
        _ => _ones::<f32>(shape, device), // fallback
    }
}

/// Creates a new tensor like the inputted one, where all the values are 1.
pub fn ones_like(tensor: &Tensor, dtype: DType, device: Device) -> Tensor {
    ones(&tensor.shape(), dtype, device)
}

/// Creates a new tensor, where the values on the main diagonal are ones
/// and the rest values are zeros.
pub fn eye(n: usize, dtype: DType, device: Device) -> Tensor {
    assert!(n > 0, "`n` cannot be less than 1.");
    let shape = &[n, n];
    let inner = match dtype {
        DType::Float8 => {
            let mut array = vec![f8::zero(); n * n];
            for i in 0..n {
                array[i * n + i] = f8::one();
            }
            TensorImpl::from_slice(&array, shape, device)
        }
        DType::Float16 => {
            let mut array = vec![f16::zero(); n * n];
            for i in 0..n {
                array[i * n + i] = f16::one();
            }
            TensorImpl::from_slice(&array, shape, device)
        }
        DType::BFloat16 => {
            let mut array = vec![bf16::zero(); n * n];
            for i in 0..n {
                array[i * n + i] = bf16::one();
            }
            TensorImpl::from_slice(&array, shape, device)
        }
        DType::Float32 => {
            let mut array = vec![f32::zero(); n * n];
            for i in 0..n {
                array[i * n + i] = f32::one();
            }
            TensorImpl::from_slice(&array, shape, device)
        }
        DType::Float64 => {
            let mut array = vec![f64::zero(); n * n];
            for i in 0..n {
                array[i * n + i] = f64::one();
            }
            TensorImpl::from_slice(&array, shape, device)
        }
        DType::Int8 => {
            let mut array = vec![i8::zero(); n * n];
            for i in 0..n {
                array[i * n + i] = i8::one();
            }
            TensorImpl::from_slice(&array, shape, device)
        }
        DType::Int16 => {
            let mut array = vec![i16::zero(); n * n];
            for i in 0..n {
                array[i * n + i] = i16::one();
            }
            TensorImpl::from_slice(&array, shape, device)
        }
        DType::Int32 => {
            let mut array = vec![i32::zero(); n * n];
            for i in 0..n {
                array[i * n + i] = i32::one();
            }
            TensorImpl::from_slice(&array, shape, device)
        }
        DType::Int64 => {
            let mut array = vec![i64::zero(); n * n];
            for i in 0..n {
                array[i * n + i] = i64::one();
            }
            TensorImpl::from_slice(&array, shape, device)
        }
        _ => {
            let mut array = vec![bool::zero(); n * n];
            for i in 0..n {
                array[i * n + i] = bool::one();
            }
            TensorImpl::from_slice(&array, shape, device)
        } // fallback
    };
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
pub fn arange(start: f32, end: f32, step: f32, dtype: DType, device: Device) -> Tensor {
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
    let inner = match dtype {
        DType::Float8 => {
            let mut data: Vec<f8> = Vec::with_capacity(len as usize);
            for i in 0..len as usize {
                let val: f8 = <f32 as Cast<f8>>::cast(start + step) * Cast::cast(i as i8);
                data.push(val);
            }
            TensorImpl::from_slice(&data, &[len as usize], device)
        }
        DType::Float16 => {
            let mut data: Vec<f16> = Vec::with_capacity(len as usize);
            for i in 0..len as usize {
                let val: f16 =
                    <f32 as Cast<f16>>::cast(start + step) * <i16 as Cast<f16>>::cast(i as i16);
                data.push(val);
            }
            TensorImpl::from_slice(&data, &[len as usize], device)
        }
        DType::BFloat16 => {
            let mut data: Vec<bf16> = Vec::with_capacity(len as usize);
            for i in 0..len as usize {
                let val: bf16 =
                    <f32 as Cast<bf16>>::cast(start + step) * <i16 as Cast<bf16>>::cast(i as i16);
                data.push(val);
            }
            TensorImpl::from_slice(&data, &[len as usize], device)
        }
        DType::Float32 => {
            let mut data: Vec<f32> = Vec::with_capacity(len as usize);
            for i in 0..len as usize {
                let val: f32 = start + step * i as f32;
                data.push(val);
            }
            TensorImpl::from_slice(&data, &[len as usize], device)
        }
        DType::Float64 => {
            let mut data: Vec<f64> = Vec::with_capacity(len as usize);
            for i in 0..len as usize {
                let val: f64 = (start + step * i as f32) as f64;
                data.push(val);
            }
            TensorImpl::from_slice(&data, &[len as usize], device)
        }
        DType::Int8 => {
            let mut data: Vec<i8> = Vec::with_capacity(len as usize);
            for i in 0..len as usize {
                let val: i8 = <f32 as Cast<i8>>::cast(start + step) * i as i8;
                data.push(val);
            }
            TensorImpl::from_slice(&data, &[len as usize], device)
        }
        DType::Int16 => {
            let mut data: Vec<i16> = Vec::with_capacity(len as usize);
            for i in 0..len as usize {
                let val: i16 = <f32 as Cast<i16>>::cast(start + step) * i as i16;
                data.push(val);
            }
            TensorImpl::from_slice(&data, &[len as usize], device)
        }
        DType::Int32 => {
            let mut data: Vec<i32> = Vec::with_capacity(len as usize);
            for i in 0..len as usize {
                let val: i32 = <f32 as Cast<i32>>::cast(start + step) * i as i32;
                data.push(val);
            }
            TensorImpl::from_slice(&data, &[len as usize], device)
        }
        DType::Int64 => {
            let mut data: Vec<i64> = Vec::with_capacity(len as usize);
            for i in 0..len as usize {
                let val: i64 = <f32 as Cast<i64>>::cast(start + step) * i as i64;
                data.push(val);
            }
            TensorImpl::from_slice(&data, &[len as usize], device)
        }
        DType::Bool => panic!("No implementation for type `bool`."),
    };
    Tensor::new(inner)
}

pub(crate) fn create_grad<T: TensorRepr>(shape: &[usize], device: Device) -> Tensor {
    let len = shape.iter().product();
    crate::tensor(&vec![T::zero(); len], shape, device)
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
        use delta_tensor::Device;
        delta_tensor::tensor!($data, Device::CPU)
    }};
    ($data:tt, $device:expr) => {{
        let shape = $crate::__tensor_shape!($data);
        let mut flat = Vec::new();
        $crate::__tensor_flatten!(flat; $data);
        $crate::tensor(&flat, &shape, $device)
    }};
}

#[macro_export]
macro_rules! randn {
    ($($element:expr),+) => {{
        delta_tensor::randn!($($element),+; delta_tensor::cpu)
    }};
    ($($element:expr),+; $device:expr) => {{
        let mut shape: Vec<usize> = Vec::new();
        $(shape.push($element);)*
        delta_tensor::randn(&shape, $device)
    }};
    ($($element:expr,)*) => {{
        $crate::tensor::randn![$($element),*]
    }};
}
