#[cfg(test)]
mod tests {
    use delta::{linalg, Tensor};

    #[test]
    /// Matrix multiplication
    fn matmul_2d() {
        let a = Tensor::tensor(&[0., 1., 2., 3., 4., 5.], &[2, 3]);
        let b: Tensor = Tensor::tensor(&[6., 7., 8., 9., 10., 11.], &[3, 2]);
        let c = Tensor::tensor(&[28., 31., 100., 112.], &[2, 2]);
        let mm = linalg::matmul(a, b);
        assert_eq!(mm.item(), c.item());
        assert_eq!(mm.shape, c.shape);
    }

    #[test]
    #[should_panic]
    /// Matrix multiplication
    fn matmul_2d_panic() {
        let a: Tensor = Tensor::tensor(&[6., 7., 8., 9., 10., 11.], &[3, 2]);
        let b = Tensor::tensor(&[28., 31., 100., 112.], &[2, 2]);
        linalg::matmul(b, a);
    }

    #[test]
    /// Batched matrix multiplication
    fn matmul_batched() {
        let a = Tensor::tensor(
            &[
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
                19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34.,
                35., 36., 37., 38., 39., 40.,
            ],
            &[2, 4, 5],
        );
        let b = Tensor::tensor(&[9., 5., 3., 2., 6., 9., 5., 3., 2., 6.0], &[5, 2]);
        let right = Tensor::tensor(
            &[
                63.0000, 78.0000, 188.0000, 203.0000, 313.0000, 328.0000, 438.0000, 453.0000,
                563.0000, 578.0000, 688.0000, 703.0000, 813.0000, 828.0000, 938.0000, 953.0000,
            ],
            &[2, 4, 2],
        );
        let c = linalg::matmul(a.clone(), b.clone());
        assert_eq!(c.item(), right.item());
        assert_eq!(c.shape, right.shape);
    }

    #[test]
    fn cross_1d() {
        let a = Tensor::tensor(&[0.6, -20.5, 5.8], &[3]);
        let b = Tensor::tensor(&[10.2, -4.6, -34.], &[3]);
        let c = linalg::cross(a.clone(), b.clone());
        assert_eq!(vec![723.6800, 79.5600, 206.3400], c.item());
    }

    #[test]
    fn cross_multidim() {
        let a = Tensor::ones(&[3, 2, 3, 2, 2, 3]);
        let b = Tensor::tensor(
            &[
                1.9269, 1.4873, 0.9007, -2.1055, 0.6784, -1.2345, -0.0431, -1.6047, -0.7521,
                1.6487, -0.3925, -1.4036, -0.7279, -0.5594, -0.7688, 0.7624, 1.6423, -0.1596,
                -0.4974, 0.4396, -0.7581, 1.0783, 0.8008, 1.6806, 1.2791, 1.2964, 0.6105, 1.3347,
                -0.2316, 0.0418, -0.2516, 0.8599, -1.3847, -0.8712, -0.2234, 1.7174, 0.3189,
                -0.4245, -0.8140, -0.7360, -0.8371, -0.9224, 1.8113, 0.1606, 0.3672, 0.1754,
                -1.1845, 1.3835, -1.2024, 0.7078, -1.0759, 0.5357, 1.1754, 0.5612,
            ],
            &[3, 1, 3, 1, 2, 3],
        );
        let c = delta::linalg::cross(a.clone(), b.clone());
        let correct = vec![
            -0.5866, 1.0262, -0.4396, -1.9130, -0.8710, 2.7839, -0.5866, 1.0262, -0.4396, -1.9130,
            -0.8710, 2.7839, 0.8525, 0.7091, -1.5616, -1.0111, 3.0523, -2.0412, 0.8525, 0.7091,
            -1.5616, -1.0111, 3.0523, -2.0412, -0.2094, 0.0410, 0.1685, -1.8019, 0.9220, 0.8799,
            -0.2094, 0.0410, 0.1685, -1.8019, 0.9220, 0.8799, -0.5866, 1.0262, -0.4396, -1.9130,
            -0.8710, 2.7839, -0.5866, 1.0262, -0.4396, -1.9130, -0.8710, 2.7839, 0.8525, 0.7091,
            -1.5616, -1.0111, 3.0523, -2.0412, 0.8525, 0.7091, -1.5616, -1.0111, 3.0523, -2.0412,
            -0.2094, 0.0410, 0.1685, -1.8019, 0.9220, 0.8799, -0.2094, 0.0410, 0.1685, -1.8019,
            0.9220, 0.8799, -1.1977, 0.2607, 0.9370, 0.8798, -0.6023, -0.2775, -1.1977, 0.2607,
            0.9370, 0.8798, -0.6023, -0.2775, -0.6860, 0.6687, 0.0173, 0.2734, 1.2930, -1.5664,
            -0.6860, 0.6687, 0.0173, 0.2734, 1.2930, -1.5664, -2.2445, 1.1331, 1.1114, 1.9407,
            -2.5886, 0.6479, -2.2445, 1.1331, 1.1114, 1.9407, -2.5886, 0.6479, -1.1977, 0.2607,
            0.9370, 0.8798, -0.6023, -0.2775, -1.1977, 0.2607, 0.9370, 0.8798, -0.6023, -0.2775,
            -0.6860, 0.6687, 0.0173, 0.2734, 1.2930, -1.5664, -0.6860, 0.6687, 0.0173, 0.2734,
            1.2930, -1.5664, -2.2445, 1.1331, 1.1114, 1.9407, -2.5886, 0.6479, -2.2445, 1.1331,
            1.1114, 1.9407, -2.5886, 0.6479, -0.3895, 1.1329, -0.7434, -0.0853, 0.1864, -0.1011,
            -0.3895, 1.1329, -0.7434, -0.0853, 0.1864, -0.1011, 0.2067, 1.4441, -1.6508, 2.5681,
            -1.2081, -1.3599, 0.2067, 1.4441, -1.6508, 2.5681, -1.2081, -1.3599, -1.7837, -0.1266,
            1.9103, -0.6142, -0.0255, 0.6397, -1.7837, -0.1266, 1.9103, -0.6142, -0.0255, 0.6397,
            -0.3895, 1.1329, -0.7434, -0.0853, 0.1864, -0.1011, -0.3895, 1.1329, -0.7434, -0.0853,
            0.1864, -0.1011, 0.2067, 1.4441, -1.6508, 2.5681, -1.2081, -1.3599, 0.2067, 1.4441,
            -1.6508, 2.5681, -1.2081, -1.3599, -1.7837, -0.1266, 1.9103, -0.6142, -0.0255, 0.6397,
            -1.7837, -0.1266, 1.9103, -0.6142, -0.0255, 0.6397,
        ];
        assert_eq!(c.shape, a.shape);
        assert_eq!(c.stride, a.stride);
        let c_item = c.item();
        for i in 0..c.length() {
            assert!((c_item[i] - correct[i]).abs() < 0.001);
        }
    }

    #[test]
    fn mul_scalar_2d() {
        let a = Tensor::arange(1., 11., 1.0);
        let a = a.reshape(&[2, 5]);
        let b = 5 as i32;
        let c = a * b;
        assert_eq!(
            c.storage(),
            vec![5., 10., 15., 20., 25., 30., 35., 40., 45., 50.,]
        );
    }

    #[test]
    fn mul_scalar_3d() {
        let a = Tensor::arange(1., 13., 1.0);
        let a = a.reshape(&[2, 2, 3]);
        let b: i32 = 5;
        let c = a * b;
        assert_eq!(
            c.storage(),
            vec![5., 10., 15., 20., 25., 30., 35., 40., 45., 50., 55., 60.]
        );
    }

    #[test]
    fn mul_3d_and_1d() {
        let a = Tensor::arange(1., 9., 1.0).reshape(&[2, 2, 2]);
        let b = Tensor::tensor(&[-2., 2.0], &[2]);
        let c = a * b;
        assert_eq!(c.storage(), vec![-2., 4., -6., 8., -10., 12., -14., 16.0]);
    }

    #[test]
    fn mul_3d_and_2d() {
        let a = Tensor::arange(1., 9., 1.).reshape(&[2, 2, 2]);
        let b = Tensor::tensor(&[-2., 3., 1., 4.], &[2, 2]);
        let c = a * b;
        assert_eq!(c.storage(), vec![-2., 6., 3., 16., -10., 18., 7., 32.]);
    }

    #[test]
    fn mul_1d_and_1d() {
        let a = Tensor::tensor(&[10., 20., 30., 40., 50., 60.], &[6]);
        let b = Tensor::arange(1., 7., 1.);
        let c = a * b;
        assert_eq!(c.storage(), vec![10., 40., 90., 160., 250., 360.]);
    }

    #[test]
    fn div_scalar_2d() {
        let a = Tensor::arange(1., 61., 10.);
        let a = a.reshape(&[2, 3]);
        let b: i32 = 5;
        let c = a / b;
        assert_eq!(c.storage(), vec![0.2, 2.2, 4.2, 6.2, 8.2, 10.2]);
    }

    #[test]
    fn div_scalar_3d() {
        let a = Tensor::arange(1., 13., 1.);
        let a = a.reshape(&[2, 2, 3]);
        let b: i32 = 5;
        let c = a / b;
        assert_eq!(
            c.storage(),
            vec![0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.4, 1.6, 1.8, 2., 2.2, 2.4]
        );
    }

    #[test]
    fn div_3d_and_1d() {
        let a = Tensor::arange(1., 9., 1.).reshape(&[2, 2, 2]);
        let b = Tensor::tensor(&[-2., 2.], &[2]);
        let c = a / b;
        assert_eq!(
            c.storage(),
            vec![-0.5, 1.0, -1.5, 2.0, -2.5, 3.0, -3.5, 4.0]
        );
    }

    #[test]
    fn div_3d_and_2d() {
        let a = Tensor::arange(1., 9., 1.).reshape(&[2, 2, 2]);
        let b = Tensor::tensor(&[-2., 3., 10., 4.], &[2, 2]);
        let c = a / b;
        assert_eq!(
            c.storage()
                .iter()
                .map(|x| (x * 10000.).round() / 10000.)
                .collect::<Vec<f64>>(),
            vec![-0.5, 0.6667, 0.3, 1.0, -2.5, 2.0, 0.7, 2.0]
        );
    }

    #[test]
    fn div_1d_and_1d() {
        let a = Tensor::tensor(&[10., 20., 30., 40., 50., 60.], &[6]);
        let b = Tensor::arange(1., 7., 1.);
        let c = a / b;
        assert_eq!(c.storage(), vec![10., 10., 10., 10., 10., 10.]);
    }

    #[test]
    fn add_scalar_3d() {
        let a = Tensor::arange(1., 9., 1.0).reshape(&[2, 2, 2]);
        let b: f32 = 2.0;
        let c = a + b;
        assert_eq!(c.storage(), vec![3., 4., 5., 6., 7., 8., 9., 10.0]);
    }

    #[test]
    fn add_1d_and_1d() {
        let a = Tensor::arange(0., 11., 1.);
        let b = Tensor::arange(0., 11., 1.);
        let c = a + b;
        assert_eq!(
            c.storage(),
            vec![0., 2., 4., 6., 8., 10., 12., 14., 16., 18., 20.]
        );
    }

    #[test]
    fn add_3d_and_3d() {
        let a = Tensor::arange(1., 9., 1.0).reshape(&[2, 2, 2]);
        let b = Tensor::arange(8., 0., -1.0).reshape(&[2, 2, 2]);
        let c = a + b;
        assert_eq!(c.storage(), vec![9., 9., 9., 9., 9., 9., 9., 9.]);
    }

    #[test]
    fn add_3d_and_2d() {
        let a = Tensor::arange(1., 9., 1.0).reshape(&[2, 2, 2]);
        let b = Tensor::tensor(&[-2., 3., 10., 4.0], &[2, 2]);
        let c = a + b;
        assert_eq!(c.storage(), vec![-1., 5., 13., 8., 3., 9., 17., 12.]);
    }

    #[test]
    fn sub_scalar_3d() {
        let a = Tensor::arange(1., 9., 1.).reshape(&[2, 2, 2]);
        let b: f32 = 2.;
        let c = a - b;
        assert_eq!(c.storage(), vec![-1., 0., 1., 2., 3., 4., 5., 6.]);
    }

    #[test]
    fn sub_3d_and_3d() {
        let a = Tensor::arange(1., 9., 1.0).reshape(&[2, 2, 2]);
        let b = Tensor::arange(8., 0., -1.0).reshape(&[2, 2, 2]);
        let c = a - b;
        assert_eq!(c.storage(), vec![-7., -5., -3., -1., 1., 3., 5., 7.]);
    }

    #[test]
    fn sub_3d_and_2d() {
        let a = Tensor::arange(1., 9., 1.0).reshape(&[2, 2, 2]);
        let b = Tensor::tensor(&[-2., 3., 10., 4.0], &[2, 2]);
        let c = a - b;
        assert_eq!(c.storage(), vec![3., -1., -7., 0., 7., 3., -3., 4.]);
    }

    #[test]
    fn sub_1d_and_1d() {
        let a = Tensor::tensor(&[10., 20., 30., 40., 50., 60.], &[6]);
        let b = Tensor::arange(1., 7., 1.);
        let c = a - b;
        assert_eq!(c.storage(), vec![9., 18., 27., 36., 45., 54.]);
    }
}
