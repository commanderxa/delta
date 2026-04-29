use crate::Tensor;

pub fn sum(input: &Tensor, dim: Option<usize>, keepdim: bool) -> Tensor {
    input.sum(dim, keepdim)
}

pub fn mean(input: &Tensor, dim: Option<usize>, keepdim: bool) -> Tensor {
    input.mean(dim, keepdim)
}

pub fn cat(tensors: &[Tensor], dim: isize) -> Tensor {
    Tensor::cat(tensors, dim)
}
