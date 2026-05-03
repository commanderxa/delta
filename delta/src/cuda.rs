#[cfg(feature = "cuda")]
use std::sync::{Arc, OnceLock};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaStream};

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
