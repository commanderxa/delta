#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;

#[cfg(feature = "cuda")]
use crate::tensor::storage_impl::CUDAStorage;
use crate::{
    DType,
    device::Device,
    tensor::{repr::TensorRepr, storage_impl::{CPUStorage, StorageRepr}},
};

#[derive(Debug)]
pub enum Storage {
    CPU(CPUStorage),
    #[cfg(feature = "cuda")]
    CUDA(CUDAStorage),
}

impl Storage {
    pub fn from_slice<T: StorageRepr>(data: &[T], device: Device) -> Self {
        match device {
            Device::CPU => Self::CPU(T::into_cpu_storage(data)),
            #[cfg(feature = "cuda")]
            Device::CUDA => Self::CUDA(T::into_cuda_storage(data)),
        }
    }

    pub fn replace_data<T: StorageRepr>(&mut self, data: &[T]) {
        *self = match self {
            Storage::CPU(..) => Storage::CPU(T::into_cpu_storage(data)),
            #[cfg(feature = "cuda")]
            Storage::CUDA(..) => Storage::CUDA(T::into_cuda_storage(data)),
        };
    }

    #[cfg(feature = "cuda")]
    pub fn to_cpu(&self) -> Self {
        match self {
            Storage::CPU(_) => panic!("Storage is already on CPU."),
            Storage::CUDA(s) => Storage::CPU(s.to_cpu()),
        }
    }

    #[cfg(feature = "cuda")]
    pub fn to_cuda(&self) -> Self {
        match self {
            Storage::CUDA(_) => panic!("Storage is already on CUDA."),
            Storage::CPU(s) => Storage::CUDA(s.to_cuda()),
        }
    }

    pub fn as_cpu<T: StorageRepr>(&self) -> &[T] {
        match self {
            Storage::CPU(data) => T::cpu_storage_as_slice(data).unwrap(),
            #[cfg(feature = "cuda")]
            Storage::CUDA(_) => panic!("Tensor is on CUDA, not CPU."),
        }
    }

    pub fn as_cpu_mut<T: StorageRepr>(&mut self) -> &mut [T] {
        match self {
            Storage::CPU(data) => T::cpu_storage_as_slice_mut(data).expect("dtype mismatch"),
            #[cfg(feature = "cuda")]
            Storage::CUDA(_) => panic!("Storage is on CUDA, call .to_cpu() first"),
        }
    }

    #[cfg(feature = "cuda")]
    pub fn as_cuda<T: StorageRepr>(&self) -> &CudaSlice<T> {
        match self {
            Storage::CPU(_) => panic!("Tensor is on CPU, not CUDA."),
            Storage::CUDA(data) => T::cuda_storage_as_slice(data).unwrap(),
        }
    }

    #[cfg(feature = "cuda")]
    pub fn as_cuda_mut<T: StorageRepr>(&mut self) -> &mut CudaSlice<T> {
        match self {
            Storage::CPU(_) => panic!("Storage is on CUDA, call .to_cpu() first"),
            #[cfg(feature = "cuda")]
            Storage::CUDA(data) => T::cuda_storage_as_slice_mut(data).expect("dtype mismatch"),
        }
    }

    pub fn iter<T: StorageRepr>(&self) -> std::slice::Iter<T> {
        self.as_cpu::<T>().iter()
    }

    pub fn map_inplace<T: StorageRepr, F: FnMut(T) -> T>(&mut self, mut f: F) {
        match self {
            Storage::CPU(_) => {
                for x in self.as_cpu_mut() {
                    *x = f(*x);
                }
            }

            #[cfg(feature = "cuda")]
            Storage::CUDA(data) => {
                let mut cpu = data.to_cpu();
                for x in T::cpu_storage_as_slice_mut(&mut cpu).unwrap() {
                    *x = f(*x);
                }
                *data = cpu.to_cuda();
            }
        }
    }

    pub fn cast_to<U: TensorRepr>(&self) -> Storage {
        match self {
            Storage::CPU(cpu) => Storage::CPU(cpu.cast_to::<U>()),
            #[cfg(feature = "cuda")]
            Storage::CUDA(_) => todo!("CUDA cast"),
        }
    }

    pub fn fill<T: TensorRepr>(&mut self, value: T) {
        match self {
            Storage::CPU(data) => data.fill(value),
            #[cfg(feature = "cuda")]
            Storage::CUDA(_) => todo!(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Storage::CPU(data) => data.len(),
            #[cfg(feature = "cuda")]
            Storage::CUDA(data) => data.len(),
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
        match self {
            Storage::CPU(cpustorage) => cpustorage.dtype(),
            #[cfg(feature = "cuda")]
            Storage::CUDA(cudastorage) => cudastorage.dtype(),
        }
    }
}

impl Clone for Storage {
    fn clone(&self) -> Self {
        match self {
            Storage::CPU(data) => Storage::CPU(data.clone()),
            #[cfg(feature = "cuda")]
            Storage::CUDA(data) => Storage::CUDA(data.clone()),
        }
    }
}

impl PartialEq for Storage {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Storage::CPU(a), Storage::CPU(b)) => a == b,
            #[cfg(feature = "cuda")]
            (Storage::CUDA(_), Storage::CUDA(_)) => self.to_cpu() == other.to_cpu(),
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
