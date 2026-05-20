pub mod backward;
#[cfg(feature = "cuda")]
pub mod cuda;
#[macro_use]
pub mod device;
pub mod ivalue;
pub mod linalg;
pub mod op;
pub mod tensor;

use std::sync::RwLock;

// define short paths
pub use device::*;
pub use tensor::Tensor;
pub use tensor::dtype::*;
pub use tensor::init::*;
pub use tensor::operations::*;
// pub use delta_macros::tensor;

// short paths for crate-wide used
pub use op::Op;
pub use tensor::impl_::TensorImpl;
pub use tensor::repr::{FloatTensorRepr, NumTensorRepr, TensorRepr};
pub use tensor::storage::Storage;
pub use tensor::storage_impl::StorageRepr;

static DEFAULT_DTYPE: RwLock<DType> = RwLock::new(crate::float32);
