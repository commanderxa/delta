#[cfg(feature = "cuda")]
use std::sync::{Arc, OnceLock};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DeviceRepr};

#[cfg(feature = "cuda")]
use crate::tensor::storage_impl::StorageRepr;

#[cfg(feature = "cuda")]
static CUDA_CONTEXT: OnceLock<Arc<CudaContext>> = OnceLock::new();

#[cfg(feature = "cuda")]
pub fn current_stream() -> Arc<CudaStream> {
    let ctx = CUDA_CONTEXT
        .get_or_init(|| CudaContext::new(0).expect("failed to initialize CUDA device 0"));
    ctx.default_stream()
}

#[cfg(feature = "cuda")]
pub fn is_available() -> bool {
    cudarc::driver::CudaContext::new(0).is_ok()
}

#[cfg(feature = "cuda")]
pub(crate) fn array_to_cuda_slice<T: StorageRepr + DeviceRepr>(data: &[T]) -> CudaSlice<T> {
    let stream = crate::cuda::current_stream();
    stream.clone_htod(data).expect("failed to copy CPU -> GPU")
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_slice_to_array<T: StorageRepr + DeviceRepr>(data: &CudaSlice<T>) -> Vec<T> {
    let stream = crate::cuda::current_stream();
    stream.clone_dtoh(data).expect("failed to copy GPU -> CPU")
}
