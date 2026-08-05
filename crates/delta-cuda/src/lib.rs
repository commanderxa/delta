use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DeviceRepr};
use delta_cpu::repr::CPUStorageRepr;
use std::sync::{Arc, OnceLock};

pub mod repr;
pub mod storage;
pub mod to_cuda;

static CUDA_CONTEXT: OnceLock<Arc<CudaContext>> = OnceLock::new();

pub fn current_stream() -> Arc<CudaStream> {
    let ctx = CUDA_CONTEXT
        .get_or_init(|| CudaContext::new(0).expect("failed to initialize CUDA device 0"));
    ctx.default_stream()
}

pub fn is_available() -> bool {
    cudarc::driver::CudaContext::new(0).is_ok()
}

pub fn array_to_cuda_slice<T: CPUStorageRepr + DeviceRepr>(data: &[T]) -> CudaSlice<T> {
    let stream = crate::current_stream();
    stream.clone_htod(data).expect("failed to copy CPU -> GPU")
}

pub fn cuda_slice_to_array<T: CPUStorageRepr + DeviceRepr>(data: &CudaSlice<T>) -> Vec<T> {
    let stream = crate::current_stream();
    stream.clone_dtoh(data).expect("failed to copy GPU -> CPU")
}
