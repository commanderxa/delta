use crate::{Tensor, tensor::element::TensorElement};

/// X is the input data
pub type X<T: TensorElement> = Tensor<T>;
/// Y is the target data
pub type Y<T: TensorElement> = Tensor<T>;
/// Sample represents zipped input and target data
pub type Sample<T, U> = (X<T>, Y<U>);
