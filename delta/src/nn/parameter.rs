use crate::{Tensor, tensor::element::TensorElement};

#[derive(Clone, Debug)]
pub struct Parameter<T: TensorElement>(pub Tensor<T>);

impl<T: TensorElement> std::ops::Deref for Parameter<T> {
    type Target = Tensor<T>;

    fn deref(&self) -> &Tensor<T> {
        &self.0
    }
}
