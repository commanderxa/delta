macro_rules! check_device {
    ($first:expr $(, $rest:expr)*) => {{
        let base = (*$first).device();
        $(
            let rest_device = (*$rest).device();
            if rest_device != base {
                panic!("{}", $crate::device::errors::DeviceError::DeviceMismatch {
                    expected: base,
                    got: rest_device,
                });
            }
        )*
        base
    }};
}

macro_rules! device_op {
    ($device:expr, cpu => $cpu_expr:expr, cuda => $cuda_expr:expr) => {{
        match $device {
            $crate::device::Device::CPU => $cpu_expr,
            #[cfg(feature = "cuda")]
            $crate::device::Device::CUDA => {
                #[cfg(not(feature = "cuda"))]
                {
                    panic!("{}", $crate::device::errors::DeviceError::CudaUnavailable);
                }
                #[cfg(feature = "cuda")]
                $cuda_expr
            }
        }
    }};
}
