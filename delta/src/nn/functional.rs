use crate::{
    Tensor, TensorImpl,
    op::Op,
    tensor::repr::{FloatTensorRepr, TensorRepr},
};

pub fn relu<T: TensorRepr>(x: Tensor) -> Tensor {
    let mut data = x.data();
    for item in data.iter_mut() {
        *item = if *item > T::zero() { *item } else { T::zero() }
    }
    let shape = x.shape();
    let inner = TensorImpl::from_op(data, &shape, vec![x.clone()], Op::ReLU, x.device());
    Tensor::new(inner)
}

pub fn sigmoid<T: FloatTensorRepr>(x: Tensor) -> Tensor {
    let data = ((-x.clone()).exp::<T>() + <T as TensorRepr>::one()).pow(-1);
    let inner = TensorImpl::from_op(
        data.data::<T>(),
        &data.shape(),
        vec![x.clone()],
        Op::Sigmoid(x.clone()),
        x.device(),
    );
    Tensor::new(inner)
}

pub fn softmax<T: FloatTensorRepr>(x: Tensor, dim: isize) -> Tensor {
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
    let mut result = vec![<T as TensorRepr>::zero(); x.length()];
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
            let max_x = _x
                .iter()
                .cloned()
                .fold(<T as FloatTensorRepr>::neg_infinity(), |a, b| {
                    if a > b { a } else { b }
                });
            let exp_x: Vec<T> = _x.iter().copied().map(|xi| (xi - max_x).exp()).collect();
            let sum_exp_x: T = exp_x
                .iter()
                .copied()
                .fold(<T as TensorRepr>::zero(), |acc, x| acc + x);
            result[(k * m + i * n)..(k * m + i * n + n)]
                .copy_from_slice(&exp_x.iter().map(|&ei| ei / sum_exp_x).collect::<Vec<T>>());
        }
    }
    // create new tensor
    let inner = TensorImpl::from_op(
        result,
        &shape,
        vec![x.clone()],
        Op::Softmax(x.clone(), dim),
        x.device(),
    );
    Tensor::new(inner)
}
