pub(crate) mod errors;
#[macro_use]
pub(crate) mod macros;

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Device {
    CPU,
    #[cfg(feature = "cuda")]
    CUDA,
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::CPU => write!(f, "CPU"),
            #[cfg(feature = "cuda")]
            Device::CUDA => write!(f, "CUDA"),
        }
    }
}

#[allow(non_upper_case_globals)]
pub const cpu: Device = Device::CPU;
#[allow(non_upper_case_globals)]
#[cfg(feature = "cuda")]
pub const cuda: Device = Device::CUDA;
