#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;

use crate::device::Device;

#[derive(Debug)]
pub(crate) enum Storage {
    CPU {
        data: Vec<f64>,
    },
    #[cfg(feature = "cuda")]
    CUDA {
        data: CudaSlice<f64>,
    },
}

impl Storage {
    /// Create a new storage based on device
    pub fn new(data: Vec<f64>, device: Device) -> Self {
        match device {
            Device::CPU => Self::CPU { data },
            #[cfg(feature = "cuda")]
            Device::CUDA => Self::CUDA {
                data: Self::vec_to_cuda(data),
            },
        }
    }

    pub fn replace_data(&mut self, data: Vec<f64>) {
        *self = match self {
            Storage::CPU { .. } => Storage::CPU { data },

            #[cfg(feature = "cuda")]
            Storage::CUDA { .. } => Storage::CUDA {
                data: Self::vec_to_cuda(data),
            },
        };
    }

    /// Materialize data as a CPU Vec (always safe to call)
    pub fn to_vec(&self) -> Vec<f64> {
        match self {
            Storage::CPU { data: v } => v.clone(),
            #[cfg(feature = "cuda")]
            Storage::CUDA { data, .. } => Self::to_vec_slice(data),
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn to_cuda_slice(data: &Vec<f64>) -> CudaSlice<f64> {
        let stream = crate::cuda::current_stream();
        stream.clone_htod(data).expect("failed to copy CPU -> GPU")
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn to_vec_slice(data: &CudaSlice<f64>) -> Vec<f64> {
        let stream = crate::cuda::current_stream();
        stream.clone_dtoh(data).expect("failed to copy GPU -> CPU")
    }

    #[cfg(feature = "cuda")]
    pub fn to_cpu(&self) -> Storage {
        Storage::CPU {
            data: self.to_vec(),
        }
    }

    #[cfg(feature = "cuda")]
    pub fn to_cuda(&self) -> Storage {
        let host = self.to_vec(); // works from any source device
        let data = Self::to_cuda_slice(&host);
        Storage::CUDA { data }
    }

    pub fn as_cpu(&self) -> &[f64] {
        match self {
            Storage::CPU { data } => data.as_slice(),
            #[cfg(feature = "cuda")]
            Storage::CUDA { .. } => panic!("Tensor is on CUDA, not CPU."),
        }
    }

    #[cfg(feature = "cuda")]
    pub fn as_cuda(&self) -> &CudaSlice<f64> {
        match self {
            Storage::CPU { .. } => panic!("Tensor is on CPU, not CUDA."),
            Storage::CUDA { data, .. } => data,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Storage::CPU { data: v } => v.len(),
            #[cfg(feature = "cuda")]
            Storage::CUDA { data, .. } => data.len(),
        }
    }

    pub fn fill(&mut self, value: f64) {
        match self {
            Storage::CPU { data } => data.fill(value),
            #[cfg(feature = "cuda")]
            Storage::CUDA { data } => {
                let stream = crate::cuda::current_stream();
                let host = vec![value; data.len()];
                let new_data = stream.clone_htod(&host).expect("fill htod failed");
                *data = new_data;
            }
        }
    }

    pub fn device(&self) -> Device {
        match self {
            Storage::CPU { .. } => Device::CPU,
            #[cfg(feature = "cuda")]
            Storage::CUDA { .. } => Device::CUDA,
        }
    }

    #[cfg(feature = "cuda")]
    fn vec_to_cuda(data: Vec<f64>) -> CudaSlice<f64> {
        let stream = crate::cuda::current_stream();
        stream.clone_htod(&data).expect("failed to copy CPU -> GPU")
    }

    pub fn iter(&self) -> std::vec::IntoIter<f64> {
        self.to_vec().into_iter()
    }

    pub fn map_inplace<F>(&mut self, mut f: F)
    where
        F: FnMut(f64) -> f64,
    {
        match self {
            Storage::CPU { data } => {
                for x in data.iter_mut() {
                    *x = f(*x);
                }
            }

            #[cfg(feature = "cuda")]
            Storage::CUDA { data } => {
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
}

impl Clone for Storage {
    fn clone(&self) -> Self {
        match self {
            Storage::CPU { data: v } => Storage::CPU { data: v.clone() },
            #[cfg(feature = "cuda")]
            Storage::CUDA { data } => {
                // allocate a new buffer and copy device → device
                let stream = crate::cuda::current_stream();
                let mut new_buf = stream.alloc_zeros::<f64>(data.len()).expect("alloc failed");
                stream
                    .memcpy_dtod(data, &mut new_buf)
                    .expect("dtod copy failed");
                Storage::CUDA { data: new_buf }
            }
        }
    }
}

impl PartialEq for Storage {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Storage::CPU { data: a }, Storage::CPU { data: b }) => a == b,
            #[cfg(feature = "cuda")]
            (Storage::CUDA { .. }, Storage::CUDA { .. }) => self.to_vec() == other.to_vec(),
            #[cfg(feature = "cuda")]
            (Storage::CPU { .. }, Storage::CUDA { .. }) => {
                panic!("Expected all data to be on the same device.")
            }
            #[cfg(feature = "cuda")]
            (Storage::CUDA { .. }, Storage::CPU { .. }) => {
                panic!("Expected all data to be on the same device.")
            }
        }
    }
}
