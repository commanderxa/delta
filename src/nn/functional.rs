use crate::{op::Op, tensor_data::TensorData, Tensor};

pub fn relu(x: Tensor) -> Tensor {
    let mut data = x.item();
    for item in data.iter_mut() {
        *item = if *item > 0.0 { *item } else { 0.0 }
    }
    let shape = x.shape();
    let inner = TensorData::from_op(data, vec![x], Op::ReLU);
    Tensor::new(inner, &shape)
}

pub fn sigmoid(x: Tensor) -> Tensor {
    let data = ((-x.clone()).exp() + 1.0 as f64).pow(-1);
    let inner = TensorData::from_op(data.item(), vec![x.clone()], Op::Sigmoid(x));
    Tensor::new(inner, &data.shape)
}

pub fn softmax(x: Tensor, dim: usize) -> Tensor {
    let shape = x.shape();
    let mut shape2 = shape.clone();
    assert_eq!(
        dim,
        shape.len() - 1,
        "Softmax for dimensions other than the last one is not supported."
    );
    let mut result = vec![0.0; x.length()];
    let data = x.item();
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
            let max_x = _x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_x: Vec<f64> = _x.iter().map(|&xi| (xi - max_x).exp()).collect();
            let sum_exp_x: f64 = exp_x.iter().sum();
            result[(k * m + i * n)..(k * m + i * n + n)]
                .copy_from_slice(&exp_x.iter().map(|&ei| ei / sum_exp_x).collect::<Vec<f64>>());
        }
    }
    // create new tensor
    let inner = TensorData::from_op(result, vec![x.clone()], Op::Softmax(x, dim));
    Tensor::new(inner, &shape)
}
