use crate::device::Device;

#[derive(Debug)]
pub enum DeviceError {
    DeviceMismatch { expected: Device, got: Device },
}

impl std::fmt::Display for DeviceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceError::DeviceMismatch { expected, got } => {
                write!(f, "Device mismatch: expected {}, got {}", expected, got)
            }
        }
    }
}

impl std::error::Error for DeviceError {}
