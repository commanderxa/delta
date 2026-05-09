pub mod backward;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod data;
#[macro_use]
pub mod device;
pub mod ivalue;
pub mod linalg;
pub mod nn;
mod op;
pub mod optim;
pub mod tensor;

// define short paths
pub use tensor::Tensor;
pub use tensor::dtype::*;
pub use tensor::init::*;
pub use tensor::operations::*;

// short paths for crate-wide used
pub(crate) use tensor::impl_::TensorImpl;
pub(crate) use tensor::storage::Storage;
