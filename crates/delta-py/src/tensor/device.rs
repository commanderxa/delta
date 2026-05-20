use delta::Device;
use pyo3::prelude::*;

pub fn register_submodule(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let device = PyModule::new(py, "_delta.tensor.device")?;
    device.add_class::<PyDevice>()?;
    device.add("cpu", PyDevice::CPU)?;
    #[cfg(feature = "cuda")]
    device.add("cuda", PyDevice::CUDA)?;
    parent.add_submodule(&device)?;
    Ok(())
}

#[pyclass(name = "Device", module = "delta", from_py_object)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PyDevice {
    CPU,
    #[cfg(feature = "cuda")]
    CUDA,
}

#[pymethods]
impl PyDevice {
    fn __repr__(&self) -> &'static str {
        match self {
            PyDevice::CPU => "delta.cpu",
            #[cfg(feature = "cuda")]
            PyDevice::CUDA => "delta.cuda",
        }
    }

    fn __str__(&self) -> &'static str {
        self.__repr__()
    }
}

impl From<Device> for PyDevice {
    fn from(value: Device) -> Self {
        match value {
            Device::CPU => PyDevice::CPU,
            #[cfg(feature = "cuda")]
            Device::CUDA => PyDevice::CUDA,
        }
    }
}

impl From<PyDevice> for Device {
    fn from(value: PyDevice) -> Self {
        match value {
            PyDevice::CPU => Device::CPU,
            #[cfg(feature = "cuda")]
            PyDevice::CUDA => Device::CUDA,
        }
    }
}
