use crate::{Tensor, tensor::element::TensorElement};

pub fn sum<T: TensorElement>(input: &Tensor<T>, dim: Option<usize>, keepdim: bool) -> Tensor<T> {
    input.sum(dim, keepdim)
}

pub fn mean<T: TensorElement>(input: &Tensor<T>, dim: Option<usize>, keepdim: bool) -> Tensor<T> {
    input.mean(dim, keepdim)
}

pub fn cat<T: TensorElement>(tensors: &[Tensor<T>], dim: isize) -> Tensor<T> {
    Tensor::cat(tensors, dim)
}
