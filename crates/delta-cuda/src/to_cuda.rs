pub trait ToCuda {
    fn to_cuda(self) -> CUDAStorage;
}

impl ToCuda for CPUStorage {
    fn to_cuda(self) -> CUDAStorage {
        match self {
            CPUStorage::I8(v) => CUDAStorage::I8(array_to_cuda_slice(v)),
            CPUStorage::I16(v) => CUDAStorage::I16(array_to_cuda_slice(v)),
            CPUStorage::I32(v) => CUDAStorage::I32(array_to_cuda_slice(v)),
            CPUStorage::I64(v) => CUDAStorage::I64(array_to_cuda_slice(v)),
            CPUStorage::F8(v) => CUDAStorage::F8(array_to_cuda_slice(v)),
            CPUStorage::F16(v) => CUDAStorage::F16(array_to_cuda_slice(v)),
            CPUStorage::BF16(v) => CUDAStorage::BF16(array_to_cuda_slice(v)),
            CPUStorage::F32(v) => CUDAStorage::F32(array_to_cuda_slice(v)),
            CPUStorage::F64(v) => CUDAStorage::F64(array_to_cuda_slice(v)),
            CPUStorage::Bool(v) => CUDAStorage::Bool(array_to_cuda_slice(v)),
        }
    }
}
