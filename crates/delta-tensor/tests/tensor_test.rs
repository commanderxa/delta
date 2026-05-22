#[cfg(test)]
mod tests {
    use delta_tensor::randn;

    #[test]
    /// Valid shape of the tensor
    fn valid_shape() {
        delta_tensor::tensor(&[0., 1., 2., 3., 4., 5.], &[2, 3], delta_tensor::cpu);
        delta_tensor::tensor(&[0., 1., 2., 3., 4., 5.], &[3, 2], delta_tensor::cpu);
        delta_tensor::tensor(&[0., 1., 2., 3., 4., 5.], &[1, 6], delta_tensor::cpu);
        delta_tensor::tensor(&[0., 1., 2., 3., 4., 5.], &[6, 1], delta_tensor::cpu);
    }

    #[test]
    #[should_panic]
    /// Invalid shape of the tensor
    fn invalid_shape() {
        delta_tensor::tensor(&[0., 1., 2., 3., 4., 5.], &[1, 3], delta_tensor::cpu);
    }

    #[test]
    fn zeros_like() {
        let a = delta_tensor::tensor(&[0., 1., 2., 3., 4., 5.], &[2, 3], delta_tensor::cpu)
            .cast(delta_tensor::float32);
        let a1 = delta_tensor::zeros_like(&a, a.dtype(), delta_tensor::cpu);
        assert_eq!(
            0.0f32,
            a1.data::<f32>().iter().sum(),
            "zeros_like produces not zeros"
        );
        assert_eq!(
            a.shape(),
            a1.shape(),
            "zeros_like produce wrong shape of a tensor"
        );
        let b = delta_tensor::randn(&[4, 10, 8], delta_tensor::cpu).cast(delta_tensor::float32);
        let b1 = delta_tensor::zeros_like(&b, b.dtype(), delta_tensor::cpu);
        assert_eq!(
            0.0f32,
            b1.data::<f32>().iter().sum(),
            "zeros_like produces not zeros"
        );
        assert_eq!(
            b.shape(),
            b1.shape(),
            "zeros_like produce wrong shape of a tensor"
        );
    }

    #[test]
    fn ones_like() {
        let a = delta_tensor::tensor(&[0., 1., 2., 3., 4., 5.], &[2, 3], delta_tensor::cpu)
            .cast(delta_tensor::float32);
        let a1 = delta_tensor::ones_like(&a, a.dtype(), delta_tensor::cpu);
        assert_eq!(
            1.0f32,
            a1.data::<f32>().iter().product(),
            "ones_like produces not ones"
        );
        assert_eq!(
            a.shape(),
            a1.shape(),
            "ones_like produce wrong shape of a tensor"
        );
        let b = delta_tensor::randn(&[4, 10, 8], delta_tensor::cpu).cast(delta_tensor::float32);
        let b1 = delta_tensor::ones_like(&b, b.dtype(), delta_tensor::cpu);
        assert_eq!(
            1.0f32,
            b1.data::<f32>().iter().product(),
            "ones_like produces not ones"
        );
        assert_eq!(
            b.shape(),
            b1.shape(),
            "ones_like produce wrong shape of a tensor"
        );
    }

    #[test]
    /// Matrix transpose
    fn t_2d() {
        let a = delta_tensor::tensor(&[0., 1., 2., 3., 4., 5.], &[2, 3], delta_tensor::cpu);
        let t = delta_tensor::tensor(&[0., 3., 1., 4., 2., 5.], &[3, 2], delta_tensor::cpu);
        assert_eq!(a.t().shape(), t.shape(), "Shapes are wrong");
        assert_eq!(a.t().data::<f64>(), t.data(), "Data is wrong");
    }

    #[test]
    /// New tensor of ordered numbers
    fn arange() {
        let a = delta_tensor::arange(0., 6., 1., delta_tensor::float32, delta_tensor::cpu);
        assert_eq!(a.shape(), vec![a.length()], "Shape is wrong");
        assert_eq!(
            a.data::<f32>(),
            vec![0., 1., 2., 3., 4., 5.],
            "Data is wrong"
        );
    }

    #[test]
    /// Reshape the tensor
    fn reshape() {
        let a = delta_tensor::tensor(
            &[0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
            &[3, 4],
            delta_tensor::cpu,
        );
        assert_eq!(a.shape(), vec![3, 4]);
        let mut _a = a.reshape(&[1, 12]);
        assert_eq!(_a.shape(), vec![1, 12]);
        _a = a.reshape(&[12, 1]);
        assert_eq!(_a.shape(), vec![12, 1]);
        _a = a.reshape(&[2, 6]);
        assert_eq!(_a.shape(), vec![2, 6]);
        _a = a.reshape(&[1, 3, 4]);
        assert_eq!(_a.shape(), vec![1, 3, 4]);
        _a = a.reshape(&[2, 2, 3]);
        assert_eq!(_a.shape(), vec![2, 2, 3]);
        assert_eq!(a.shape(), vec![3, 4]);

        let a = delta_tensor::arange(0., 9., 1., delta_tensor::float32, delta_tensor::cpu)
            .view(&[3, 1, 3]);
        assert_eq!(a.stride(), vec![3, 3, 1]);
        assert_eq!(a.shape(), vec![3, 1, 3]);
        let a = a.expand(&[3, 2, 3]);
        assert_eq!(a.stride(), vec![3, 0, 1]);
        assert_eq!(a.shape(), vec![3, 2, 3]);
        let a = a.reshape(&[2, 3, 3]);
        assert_eq!(a.stride(), vec![9, 3, 1]);
        assert_eq!(a.shape(), vec![2, 3, 3]);
        assert_eq!(
            a.data::<f64>(),
            vec![
                0., 1., 2., 0., 1., 2., 3., 4., 5., 3., 4., 5., 6., 7., 8., 6., 7., 8.0
            ]
        );
    }

    #[test]
    #[should_panic]
    /// Invalid view of the tensor
    fn reshape_invalid() {
        let a = delta_tensor::tensor(
            &[0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
            &[3, 4],
            delta_tensor::cpu,
        );
        a.reshape(&[4, 5]);
    }

    #[test]
    /// View the tensor
    fn view() {
        let a = delta_tensor::tensor(
            &[0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
            &[3, 4],
            delta_tensor::cpu,
        );
        let b = a.view(&[1, 12]);
        assert_eq!(b.shape(), vec![1, 12]);
        let c = a.view(&[12, 1]);
        assert_eq!(c.shape(), vec![12, 1]);
        let d = a.view(&[2, 6]);
        assert_eq!(d.shape(), vec![2, 6]);
        let e = a.view(&[1, 3, 4]);
        assert_eq!(e.shape(), vec![1, 3, 4]);
        let f = a.view(&[2, 2, 3]);
        assert_eq!(f.shape(), vec![2, 2, 3]);
    }

    #[test]
    #[should_panic]
    /// Invalid view of the tensor
    fn view_invalid() {
        let a = delta_tensor::tensor(
            &[0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
            &[3, 4],
            delta_tensor::cpu,
        );
        a.view(&[4, 5]);
    }

    #[test]
    #[should_panic]
    fn view_invalid_2() {
        let a = delta_tensor::arange(0., 9., 1., delta_tensor::float32, delta_tensor::cpu)
            .view(&[3, 1, 3]);
        let a = a.expand(&[3, 2, 3]);
        a.view(&[2, 3, 3]);
    }

    #[test]
    fn indexing() {
        let a = delta_tensor::arange(0., 9., 1., delta_tensor::float32, delta_tensor::cpu)
            .reshape(&[3, 3]);
        let b = a.slice([0..2, 0..2]);
        let b_correct = delta_tensor::tensor(&[0, 1, 3, 4], &[2, 2], delta_tensor::cpu)
            .cast(delta_tensor::float32);
        assert_eq!(b.data::<f32>(), b_correct.data::<f32>());

        let a = delta_tensor::arange(0., 9., 1., delta_tensor::float32, delta_tensor::cpu)
            .reshape(&[3, 3]);
        let b = a.slice([1..3, 0..2]);
        let b_correct = delta_tensor::tensor(&[3, 4, 6, 7], &[2, 2], delta_tensor::cpu)
            .cast(delta_tensor::float32);
        assert_eq!(b.data::<f32>(), b_correct.data::<f32>());

        let a = delta_tensor::arange(0., 9., 1., delta_tensor::float32, delta_tensor::cpu)
            .reshape(&[1, 1, 3, 3]);
        let b = a.slice([0..-1, 0..-1, 1..3, 0..2]);
        let b_correct = delta_tensor::tensor(&[3, 4, 6, 7], &[2, 2], delta_tensor::cpu)
            .cast(delta_tensor::float32);
        assert_eq!(b.data::<f32>(), b_correct.data::<f32>());
    }

    #[test]
    fn pow() {
        let a = delta_tensor::tensor(&[0., 1., 2., 3., 4., 5.], &[2, 3], delta_tensor::cpu);
        let b = delta_tensor::tensor(&[0., 1., 4., 9., 16., 25.], &[2, 3], delta_tensor::cpu);
        assert_eq!(a.pow(2).data::<f64>(), b.data(), "Pow is wrong");
    }

    #[test]
    fn log() {
        let a = delta_tensor::tensor(
            &[4.7767, 4.3234, 1.2156, 0.2411, 4.5739],
            &[5],
            delta_tensor::cpu,
        );
        let b = delta_tensor::tensor(
            &[
                1.563749931514684,
                1.4640421297418154,
                0.19523778206050096,
                -1.4225434937950117,
                1.520366232659367,
            ],
            &[5],
            delta_tensor::cpu,
        );
        assert_eq!(a.log().data::<f64>(), b.data(), "Pow is wrong");
    }

    #[test]
    fn neg() {
        let a = delta_tensor::arange(0., 10., 1.0, delta_tensor::float32, delta_tensor::cpu);
        let b = -a.clone();
        assert_eq!(
            b,
            delta_tensor::arange(0., -10., -1.0, delta_tensor::float32, delta_tensor::cpu)
        );
    }

    #[test]
    fn randn_macro() {
        let a = randn![2, 3];
        let b = delta_tensor::randn(&[2, 3], delta_tensor::cpu);
        assert_eq!(a.length(), b.length());
        assert_eq!(a.shape(), b.shape());
    }

    #[test]
    fn stride() {
        let a = delta_tensor::ones(
            &[1, 1, 3, 1, 3, 3],
            delta_tensor::float32,
            delta_tensor::cpu,
        );
        assert_eq!(a.stride(), vec![27, 27, 9, 9, 3, 1]);

        let a = a.expand(&[2, 2, 3, 3, 3, 3]);
        assert_eq!(a.stride(), vec![0, 0, 9, 0, 3, 1]);

        let a = delta_tensor::ones(&[4, 1], delta_tensor::float32, delta_tensor::cpu);
        assert_eq!(a.stride(), vec![1, 1]);

        let a = a.expand(&[4, 5]);
        assert_eq!(a.stride(), vec![1, 0]);

        let a = delta_tensor::ones(&[4, 1, 1], delta_tensor::float32, delta_tensor::cpu);
        assert_eq!(a.stride(), vec![1, 1, 1]);

        let a = a.expand(&[4, 3, 5]);
        assert_eq!(a.stride(), vec![1, 0, 0]);
    }

    #[test]
    fn expand() {
        let a = delta_tensor::ones(
            &[1, 1, 3, 1, 3, 3],
            delta_tensor::float32,
            delta_tensor::cpu,
        );
        let b = a.expand(&[2, 2, 3, 3, 3, 3]);
        assert_eq!(a.shape(), vec![1, 1, 3, 1, 3, 3]);
        assert_eq!(b.shape(), vec![2, 2, 3, 3, 3, 3]);
        assert_eq!(a.stride(), vec![27, 27, 9, 9, 3, 1]);
        assert_eq!(b.stride(), vec![0, 0, 9, 0, 3, 1]);
    }

    #[test]
    fn unsqueeze() {
        let a = randn!(2, 3, 4);
        let b = a.unsqueeze(1);
        assert_eq!(a.shape(), vec![2, 3, 4]);
        assert_eq!(b.stride(), vec![12, 4, 4, 1]);
        assert_eq!(b.shape(), vec![2, 1, 3, 4]);
        let c = b.unsqueeze(4);
        assert_eq!(a.shape(), vec![2, 3, 4]);
        assert_eq!(b.shape(), vec![2, 1, 3, 4]);
        assert_eq!(c.shape(), vec![2, 1, 3, 4, 1]);
        let d = a.unsqueeze(0);
        assert_eq!(d.shape(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn squeeze() {
        let a = randn!(2, 1, 3, 4, 1);
        let b = a.squeeze(&[]);
        assert_eq!(a.shape(), vec![2, 1, 3, 4, 1]);
        assert_eq!(b.shape(), vec![2, 3, 4]);
        let c = a.squeeze(&[1, 2]);
        assert_eq!(a.shape(), vec![2, 1, 3, 4, 1]);
        assert_eq!(b.shape(), vec![2, 3, 4]);
        assert_eq!(c.shape(), vec![2, 3, 4, 1]);
    }

    #[test]
    fn eye() {
        let expectation = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let actual: Vec<f64> =
            delta_tensor::eye(4, delta_tensor::float64, delta_tensor::cpu).data();
        assert_eq!(actual, expectation);
    }

    #[test]
    fn cat() {
        let exp = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let a = delta_tensor::ones(&[2, 4], delta_tensor::float64, delta_tensor::cpu);
        let b = delta_tensor::zeros(&[1, 4], delta_tensor::float64, delta_tensor::cpu);
        let c = delta_tensor::cat(&[a, b], 0);
        assert_eq!(c.data::<f64>(), exp);

        let exp = vec![1., 1., 1., 1., 0., 0., 0., 0.];
        let a = delta_tensor::ones(&[1, 4], delta_tensor::float64, delta_tensor::cpu);
        let b = delta_tensor::zeros(&[1, 4], delta_tensor::float64, delta_tensor::cpu);
        let c = delta_tensor::cat(&[a, b], -1);
        assert_eq!(c.data::<f64>(), exp);
    }

    #[test]
    fn sum() {
        let a = delta_tensor::ones(&[1, 4], delta_tensor::float64, delta_tensor::cpu)
            .sum(Some(1), false);
        let a_e = vec![4.];
        assert_eq!(a.data::<f64>(), a_e);
        let b = delta_tensor::ones(&[4, 4], delta_tensor::float64, delta_tensor::cpu)
            .sum(Some(0), false);
        let b_e = vec![4., 4., 4., 4.];
        assert_eq!(b.data::<f64>(), b_e);
    }

    #[test]
    fn mean() {
        let a = delta_tensor::arange(0., 9., 1., delta_tensor::float64, delta_tensor::cpu)
            .reshape(&[3, 3]);
        let a0 = a.reshape(&[9]).mean(Some(0), false);
        let a0_e = vec![4.];
        assert_eq!(a0.data::<f64>(), a0_e);
        let a1 = a.mean(Some(0), false);
        let a1_e = vec![3., 4., 5.];
        assert_eq!(a1.data::<f64>(), a1_e);
        let a2 = a.mean(Some(1), false);
        let a2_e = vec![1., 4., 7.];
        assert_eq!(a2.data::<f64>(), a2_e);
    }

    #[test]
    fn tensor_macro() {
        let a = delta_tensor::tensor(&[1., 2., 3.], &[3], delta_tensor::cpu);
        let b = delta_tensor::tensor!([1, 2, 3]);
        assert_eq!(a.data::<f64>(), b.data());
        assert_eq!(a.shape(), b.shape());
        assert_eq!(a.stride(), b.stride());

        let a = delta_tensor::arange(0., 9., 1., delta_tensor::int32, delta_tensor::cpu)
            .reshape(&[3, 3]);
        let b = delta_tensor::tensor!([[0, 1, 2], [3, 4, 5], [6, 7, 8]]);
        assert_eq!(a.data::<i32>(), b.data());
        assert_eq!(a.shape(), b.shape());
        assert_eq!(a.stride(), b.stride());

        let a = delta_tensor::arange(0., 9., 1., delta_tensor::int32, delta_tensor::cpu)
            .reshape(&[1, 3, 3]);
        let b = delta_tensor::tensor!([[[0, 1, 2], [3, 4, 5], [6, 7, 8]]]);
        assert_eq!(a.data::<i32>(), b.data());
        assert_eq!(a.shape(), b.shape());
        assert_eq!(a.stride(), b.stride());
    }

    #[test]
    fn eq() {
        let a = delta_tensor::tensor(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[1, 2, 3],
            delta_tensor::cpu,
        );
        let b = delta_tensor::tensor(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[1, 2, 3],
            delta_tensor::cpu,
        );
        assert_eq!(a, b);
    }

    #[test]
    fn not_eq() {
        let a = delta_tensor::tensor(
            &[2.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[1, 2, 3],
            delta_tensor::cpu,
        );
        let b = delta_tensor::tensor(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[1, 2, 3],
            delta_tensor::cpu,
        );
        assert_ne!(a, b);

        let a = delta_tensor::tensor(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[1, 2, 3],
            delta_tensor::cpu,
        );
        let b = delta_tensor::tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], delta_tensor::cpu);
        assert_ne!(a, b);
    }
}
