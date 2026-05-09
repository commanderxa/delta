#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;

use crate::{DType, device::Device, tensor::element::TensorElement};

#[derive(Debug)]
pub enum Storage<T: TensorElement> {
    CPU(Vec<T>),
    #[cfg(feature = "cuda")]
    CUDA(CudaSlice<T>),
}

impl<T: TensorElement> Storage<T> {
    pub fn from_slice(data: &[T], device: Device) -> Self {
        match device {
            Device::CPU => Self::CPU(data.to_vec()),
            #[cfg(feature = "cuda")]
            Device::CUDA => Self::CUDA(Self::vec_to_cuda(data.to_vec())),
        }
    }

    pub fn replace_data(&mut self, data: Vec<T>) {
        *self = match self {
            Storage::CPU(..) => Storage::CPU(data),
            #[cfg(feature = "cuda")]
            Storage::CUDA(..) => Storage::CUDA(Self::vec_to_cuda(data)),
        };
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn to_cuda_slice(data: &Vec<T>) -> CudaSlice<T> {
        let stream = crate::cuda::current_stream();
        stream.clone_htod(data).expect("failed to copy CPU -> GPU")
    }

    #[cfg(feature = "cuda")]
    pub fn to_cpu(&self) -> Self {
        Storage::CPU(self.to_vec())
    }

    #[cfg(feature = "cuda")]
    pub fn to_cuda(&self) -> Self {
        let host = self.to_vec(); // works from any source device
        let data = Self::to_cuda_slice(&host);
        Storage::CUDA(data)
    }

    pub fn as_cpu(&self) -> &[T] {
        match self {
            Storage::CPU(data) => data.as_slice(),
            #[cfg(feature = "cuda")]
            Storage::CUDA { .. } => panic!("Tensor is on CUDA, not CPU."),
        }
    }

    #[cfg(feature = "cuda")]
    pub fn as_cuda(&self) -> &CudaSlice<T> {
        match self {
            Storage::CPU(_) => panic!("Tensor is on CPU, not CUDA."),
            Storage::CUDA(data) => data,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Storage::CPU(data) => data.len(),
            #[cfg(feature = "cuda")]
            Storage::CUDA(data) => data.len(),
        }
    }

    pub fn fill(&mut self, value: T) {
        match self {
            Storage::CPU(data) => data.fill(value),
            #[cfg(feature = "cuda")]
            Storage::CUDA(data) => {
                let stream = crate::cuda::current_stream();
                let host = vec![value; data.len()];
                let new_data = stream.clone_htod(&host).expect("fill htod failed");
                *data = new_data;
            }
        }
    }

    pub fn device(&self) -> Device {
        match self {
            Storage::CPU(..) => Device::CPU,
            #[cfg(feature = "cuda")]
            Storage::CUDA(..) => Device::CUDA,
        }
    }

    pub fn dtype(&self) -> DType {
        T::dtype()
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn vec_to_cuda(data: Vec<T>) -> CudaSlice<T>
    where
        T: cudarc::driver::DeviceRepr,
    {
        let stream = crate::cuda::current_stream();
        stream.clone_htod(&data).expect("failed to copy CPU -> GPU")
    }

    pub fn iter(&self) -> std::vec::IntoIter<T> {
        self.to_vec().into_iter()
    }

    pub fn map_inplace<F>(&mut self, mut f: F)
    where
        F: FnMut(T) -> T,
    {
        match self {
            Storage::CPU(data) => {
                for x in data.iter_mut() {
                    *x = f(*x);
                }
            }

            #[cfg(feature = "cuda")]
            Storage::CUDA(data) => {
                let stream = crate::cuda::current_stream();
                let mut host = stream.clone_dtoh(data).expect("failed to copy GPU -> CPU");

                for x in host.iter_mut() {
                    *x = f(*x);
                }

                let new_data = stream.clone_htod(&host).expect("failed to copy CPU -> GPU");

                *data = new_data;
            }
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn to_vec_slice(data: &CudaSlice<T>) -> Vec<T> {
        let stream = crate::cuda::current_stream();
        stream.clone_dtoh(data).expect("failed to copy GPU -> CPU")
    }

    /// Materialize data as a CPU Vec (always safe to call)
    pub fn to_vec(&self) -> Vec<T> {
        match self {
            Storage::CPU(data) => data.to_vec(),
            #[cfg(feature = "cuda")]
            Storage::CUDA(data) => Self::to_vec_slice(data),
        }
    }
}

impl<T: TensorElement> Clone for Storage<T> {
    fn clone(&self) -> Self {
        match self {
            Storage::CPU(data) => Storage::CPU(data.clone()),
            #[cfg(feature = "cuda")]
            Storage::CUDA(data) => {
                // allocate a new buffer and copy device → device
                let stream = crate::cuda::current_stream();
                let mut new_buf = stream.alloc_zeros::<T>(data.len()).expect("alloc failed");
                stream
                    .memcpy_dtod(data, &mut new_buf)
                    .expect("dtod copy failed");
                Storage::CUDA(new_buf)
            }
        }
    }
}

impl<T: TensorElement> PartialEq for Storage<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Storage::CPU(a), Storage::CPU(b)) => a == b,
            #[cfg(feature = "cuda")]
            (Storage::CUDA(_), Storage::CUDA(_)) => self.to_vec() == other.to_vec(),
            #[cfg(feature = "cuda")]
            (Storage::CPU(_), Storage::CUDA(_)) => {
                panic!("Expected all data to be on the same device.")
            }
            #[cfg(feature = "cuda")]
            (Storage::CUDA(_), Storage::CPU(_)) => {
                panic!("Expected all data to be on the same device.")
            }
        }
    }
}
