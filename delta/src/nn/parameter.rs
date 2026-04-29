use crate::Tensor;

#[derive(Clone, Debug)]
pub struct Parameter(pub Tensor);

impl std::ops::Deref for Parameter {
    type Target = Tensor;

    fn deref(&self) -> &Tensor {
        &self.0
    }
}
