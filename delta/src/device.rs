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
