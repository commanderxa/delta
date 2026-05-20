#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use delta_tensor::{Tensor, device::Device, linalg};

    #[test]
    /// Matrix multiplication
    fn matmul_2d() {
        let a = delta_tensor::tensor(&[0., 1., 2., 3., 4., 5.], &[2, 3], delta_tensor::cpu).cuda();
        let b: Tensor = delta_tensor::tensor(&[6., 7., 8., 9., 10., 11.], &[3, 2], delta_tensor::cpu).cuda();
        let c = delta_tensor::tensor(&[28., 31., 100., 112.], &[2, 2], delta_tensor::cpu).cuda();
        let mm = linalg::matmul(a.clone(), b.clone());
        assert_eq!(mm.data::<f64>(), c.data());
        assert_eq!(a.device(), Device::CUDA);
        assert_eq!(b.device(), Device::CUDA);
        assert_eq!(mm.device(), Device::CUDA);
        assert_eq!(mm.shape(), c.shape());
    }

    #[test]
    #[should_panic]
    /// Matrix multiplication
    fn matmul_2d_panic() {
        let a: Tensor = delta_tensor::tensor(&[6., 7., 8., 9., 10., 11.], &[3, 2], delta_tensor::cpu).cuda();
        let b = delta_tensor::tensor(&[28., 31., 100., 112.], &[2, 2], delta_tensor::cpu).cuda();
        linalg::matmul(b, a);
    }

    #[test]
    /// Batched matrix multiplication
    fn matmul_batched() {
        let a = delta_tensor::tensor(
            &[
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
                19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34.,
                35., 36., 37., 38., 39., 40.,
            ],
            &[2, 4, 5],
            delta_tensor::cpu,
        )
        .cuda();
        let b = delta_tensor::tensor(
            &[9., 5., 3., 2., 6., 9., 5., 3., 2., 6.0],
            &[5, 2],
            delta_tensor::cpu,
        )
        .cuda();
        let right = delta_tensor::tensor(
            &[
                63.0000, 78.0000, 188.0000, 203.0000, 313.0000, 328.0000, 438.0000, 453.0000,
                563.0000, 578.0000, 688.0000, 703.0000, 813.0000, 828.0000, 938.0000, 953.0000,
            ],
            &[2, 4, 2],
            delta_tensor::cpu,
        )
        .cuda();
        let c = linalg::matmul(a.clone(), b.clone());
        assert_eq!(c.data::<f64>(), right.data());
        assert_eq!(a.device(), Device::CUDA);
        assert_eq!(b.device(), Device::CUDA);
        assert_eq!(c.device(), Device::CUDA);
        assert_eq!(c.shape(), right.shape());
    }
}
