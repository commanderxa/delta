use crate::{Tensor, tensor::repr::TensorRepr};

pub fn sum(input: &Tensor, dim: Option<usize>, keepdim: bool) -> Tensor {
    input.sum(dim, keepdim)
}

pub fn mean(input: &Tensor, dim: Option<usize>, keepdim: bool) -> Tensor {
    input.mean(dim, keepdim)
}

pub fn cat<T: TensorRepr>(tensors: &[Tensor], dim: isize) -> Tensor {
    Tensor::cat(tensors, dim)
}
