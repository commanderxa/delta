use delta_tensor::{FloatTensorRepr, Op, Tensor, TensorImpl, TensorRepr, f8};
use half::{bf16, f16};

pub fn relu(x: Tensor) -> Tensor {
    let inner = match x.dtype() {
        delta_tensor::DType::Float8 => {
            let mut data = x.data::<f8>();
            for item in data.iter_mut() {
                *item = if *item > f8::zero() {
                    *item
                } else {
                    f8::zero()
                }
            }
            let shape = x.shape();
            TensorImpl::from_op(data, &shape, vec![x.clone()], Op::ReLU, x.device())
        }
        delta_tensor::DType::Float16 => {
            let mut data = x.data::<f16>();
            for item in data.iter_mut() {
                *item = if *item > f16::zero() {
                    *item
                } else {
                    f16::zero()
                }
            }
            let shape = x.shape();
            TensorImpl::from_op(data, &shape, vec![x.clone()], Op::ReLU, x.device())
        }
        delta_tensor::DType::BFloat16 => {
            let mut data = x.data::<bf16>();
            for item in data.iter_mut() {
                *item = if *item > bf16::zero() {
                    *item
                } else {
                    bf16::zero()
                }
            }
            let shape = x.shape();
            TensorImpl::from_op(data, &shape, vec![x.clone()], Op::ReLU, x.device())
        }
        delta_tensor::DType::Float32 => {
            let mut data = x.data::<f32>();
            for item in data.iter_mut() {
                *item = if *item > f32::zero() {
                    *item
                } else {
                    f32::zero()
                }
            }
            let shape = x.shape();
            TensorImpl::from_op(data, &shape, vec![x.clone()], Op::ReLU, x.device())
        }
        delta_tensor::DType::Float64 => {
            let mut data = x.data::<f64>();
            for item in data.iter_mut() {
                *item = if *item > f64::zero() {
                    *item
                } else {
                    f64::zero()
                }
            }
            let shape = x.shape();
            TensorImpl::from_op(data, &shape, vec![x.clone()], Op::ReLU, x.device())
        }
        delta_tensor::DType::Int8 => {
            let mut data = x.data::<i8>();
            for item in data.iter_mut() {
                *item = if *item > i8::zero() {
                    *item
                } else {
                    i8::zero()
                }
            }
            let shape = x.shape();
            TensorImpl::from_op(data, &shape, vec![x.clone()], Op::ReLU, x.device())
        }
        delta_tensor::DType::Int16 => {
            let mut data = x.data::<i16>();
            for item in data.iter_mut() {
                *item = if *item > i16::zero() {
                    *item
                } else {
                    i16::zero()
                }
            }
            let shape = x.shape();
            TensorImpl::from_op(data, &shape, vec![x.clone()], Op::ReLU, x.device())
        }
        delta_tensor::DType::Int32 => {
            let mut data = x.data::<i32>();
            for item in data.iter_mut() {
                *item = if *item > i32::zero() {
                    *item
                } else {
                    i32::zero()
                }
            }
            let shape = x.shape();
            TensorImpl::from_op(data, &shape, vec![x.clone()], Op::ReLU, x.device())
        }
        delta_tensor::DType::Int64 => {
            let mut data = x.data::<i64>();
            for item in data.iter_mut() {
                *item = if *item > i64::zero() {
                    *item
                } else {
                    i64::zero()
                }
            }
            let shape = x.shape();
            TensorImpl::from_op(data, &shape, vec![x.clone()], Op::ReLU, x.device())
        }
        delta_tensor::DType::Bool => {
            let mut data = x.data::<bool>();
            for item in data.iter_mut() {
                *item = if *item > bool::zero() {
                    *item
                } else {
                    bool::zero()
                }
            }
            let shape = x.shape();
            TensorImpl::from_op(data, &shape, vec![x.clone()], Op::ReLU, x.device())
        }
    };
    Tensor::new(inner)
}

pub fn sigmoid(x: Tensor) -> Tensor {
    let inner = match x.dtype() {
        delta_tensor::float8 => {
            let data = ((-x.clone()).exp() + f8::one()).pow(-1);
            TensorImpl::from_op(
                data.data::<f8>(),
                &data.shape(),
                vec![x.clone()],
                Op::Sigmoid(x.clone()),
                x.device(),
            )
        }
        delta_tensor::float16 => {
            let data = ((-x.clone()).exp() + f16::one()).pow(-1);
            TensorImpl::from_op(
                data.data::<f16>(),
                &data.shape(),
                vec![x.clone()],
                Op::Sigmoid(x.clone()),
                x.device(),
            )
        }
        delta_tensor::bfloat16 => {
            let data = ((-x.clone()).exp() + bf16::one()).pow(-1);
            TensorImpl::from_op(
                data.data::<bf16>(),
                &data.shape(),
                vec![x.clone()],
                Op::Sigmoid(x.clone()),
                x.device(),
            )
        }
        delta_tensor::float32 => {
            let data = ((-x.clone()).exp() + f32::one()).pow(-1);
            TensorImpl::from_op(
                data.data::<f32>(),
                &data.shape(),
                vec![x.clone()],
                Op::Sigmoid(x.clone()),
                x.device(),
            )
        }
        delta_tensor::float64 => {
            let data = ((-x.clone()).exp() + f64::one()).pow(-1);
            TensorImpl::from_op(
                data.data::<f64>(),
                &data.shape(),
                vec![x.clone()],
                Op::Sigmoid(x.clone()),
                x.device(),
            )
        }
        delta_tensor::int8 => todo!(),
        delta_tensor::int16 => todo!(),
        delta_tensor::int32 => todo!(),
        delta_tensor::int64 => todo!(),
        delta_tensor::bool => todo!(),
    };
    Tensor::new(inner)
}

pub fn softmax(x: Tensor, dim: isize) -> Tensor {
    assert!(dim >= -1, "cat: `dim` cannot be negative integer");

    let shape = x.shape();

    let dim: usize = if dim == -1 {
        shape.len() - 1
    } else {
        dim as usize
    };

    let mut shape2 = shape.clone();
    assert_eq!(
        dim,
        shape.len() - 1,
        "Softmax for dimensions other than the last one is not supported."
    );
    let inner = match x.dtype() {
        delta_tensor::float8 => todo!(),
        delta_tensor::float16 => todo!(),
        delta_tensor::bfloat16 => todo!(),
        delta_tensor::float32 => {
            let mut result = vec![<f32 as TensorRepr>::zero(); x.length()];
            let data = x.data();
            // get batch dimensions if they exist
            let mut batches: Vec<usize> = vec![];
            for i in 2..shape.len() {
                batches.push(shape[i - 2]);
            }
            // remove batch dimensions from the A tensor shape
            shape2.drain(0..batches.len());
            let batch_prod = batches.iter().product::<usize>();
            let m = shape2[0];
            let n = shape2[1];
            // iterate over the batch dimensions
            // `k` is a batch dimension
            for k in 0..batch_prod {
                for i in 0..m {
                    let _x = &data[(k * m + i * n)..(k * m + i * n + n)];
                    // do operations
                    let max_x = _x.iter().cloned().fold(
                        <f32 as FloatTensorRepr>::neg_infinity(),
                        |a, b| {
                            if a > b { a } else { b }
                        },
                    );
                    let exp_x: Vec<f32> = _x.iter().copied().map(|xi| (xi - max_x).exp()).collect();
                    let sum_exp_x: f32 = exp_x
                        .iter()
                        .copied()
                        .fold(<f32 as TensorRepr>::zero(), |acc, x| acc + x);
                    result[(k * m + i * n)..(k * m + i * n + n)].copy_from_slice(
                        &exp_x.iter().map(|&ei| ei / sum_exp_x).collect::<Vec<f32>>(),
                    );
                }
            }
            // create new tensor
            TensorImpl::from_op(
                result,
                &shape,
                vec![x.clone()],
                Op::Softmax(x.clone(), dim),
                x.device(),
            )
        }
        delta_tensor::float64 => todo!(),
        _ => unreachable!(),
    };
    Tensor::new(inner)
}
