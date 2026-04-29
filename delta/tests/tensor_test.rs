#[cfg(test)]
mod tests {
    use delta::randn;

    #[test]
    /// Valid shape of the tensor
    fn valid_shape() {
        delta::tensor(&[0., 1., 2., 3., 4., 5.], &[2, 3]);
        delta::tensor(&[0., 1., 2., 3., 4., 5.], &[3, 2]);
        delta::tensor(&[0., 1., 2., 3., 4., 5.], &[1, 6]);
        delta::tensor(&[0., 1., 2., 3., 4., 5.], &[6, 1]);
    }

    #[test]
    #[should_panic]
    /// Invalid shape of the tensor
    fn invalid_shape() {
        delta::tensor(&[0., 1., 2., 3., 4., 5.], &[1, 3]);
    }

    #[test]
    fn zeros_like() {
        let a = delta::tensor(&[0., 1., 2., 3., 4., 5.], &[2, 3]);
        let a1 = delta::zeros_like(&a);
        assert_eq!(0., a1.item().iter().sum(), "zeros_like produces not zeros");
        assert_eq!(
            a.shape, a1.shape,
            "zeros_like produce wrong shape of a tensor"
        );
        let b = delta::randn(&[4, 10, 8]);
        let b1 = delta::zeros_like(&b);
        assert_eq!(0., b1.item().iter().sum(), "zeros_like produces not zeros");
        assert_eq!(
            b.shape, b1.shape,
            "zeros_like produce wrong shape of a tensor"
        );
    }

    #[test]
    fn ones_like() {
        let a = delta::tensor(&[0., 1., 2., 3., 4., 5.], &[2, 3]);
        let a1 = delta::ones_like(&a);
        assert_eq!(
            1.,
            a1.item().iter().product(),
            "ones_like produces not ones"
        );
        assert_eq!(
            a.shape, a1.shape,
            "ones_like produce wrong shape of a tensor"
        );
        let b = delta::randn(&[4, 10, 8]);
        let b1 = delta::ones_like(&b);
        assert_eq!(
            1.,
            b1.item().iter().product(),
            "ones_like produces not ones"
        );
        assert_eq!(
            b.shape, b1.shape,
            "ones_like produce wrong shape of a tensor"
        );
    }

    #[test]
    /// Matrix transpose
    fn t_2d() {
        let a = delta::tensor(&[0., 1., 2., 3., 4., 5.], &[2, 3]);
        let t = delta::tensor(&[0., 3., 1., 4., 2., 5.], &[3, 2]);
        assert_eq!(a.t().shape, t.shape, "Shapes are wrong");
        assert_eq!(a.t().item(), t.item(), "Data is wrong");
    }

    #[test]
    /// New tensor of ordered numbers
    fn arange() {
        let a = delta::arange(0., 6., 1.);
        assert_eq!(a.shape, vec![a.length()], "Shape is wrong");
        assert_eq!(a.item(), vec![0., 1., 2., 3., 4., 5.], "Data is wrong");
    }

    #[test]
    /// Reshape the tensor
    fn reshape() {
        let a = delta::tensor(&[0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.], &[3, 4]);
        assert_eq!(a.shape, vec![3, 4]);
        let mut _a = a.reshape(&[1, 12]);
        assert_eq!(_a.shape, vec![1, 12]);
        _a = a.reshape(&[12, 1]);
        assert_eq!(_a.shape, vec![12, 1]);
        _a = a.reshape(&[2, 6]);
        assert_eq!(_a.shape, vec![2, 6]);
        _a = a.reshape(&[1, 3, 4]);
        assert_eq!(_a.shape, vec![1, 3, 4]);
        _a = a.reshape(&[2, 2, 3]);
        assert_eq!(_a.shape, vec![2, 2, 3]);
        assert_eq!(a.shape, vec![3, 4]);

        let a = delta::arange(0., 9., 1.).view(&[3, 1, 3]);
        assert_eq!(a.stride, vec![3, 3, 1]);
        assert_eq!(a.shape, vec![3, 1, 3]);
        let a = a.expand(&[3, 2, 3]);
        assert_eq!(a.stride, vec![3, 0, 1]);
        assert_eq!(a.shape, vec![3, 2, 3]);
        let a = a.reshape(&[2, 3, 3]);
        assert_eq!(a.stride, vec![9, 3, 1]);
        assert_eq!(a.shape, vec![2, 3, 3]);
        assert_eq!(
            a.storage(),
            vec![
                0., 1., 2., 0., 1., 2., 3., 4., 5., 3., 4., 5., 6., 7., 8., 6., 7., 8.0
            ]
        );
    }

    #[test]
    #[should_panic]
    /// Invalid view of the tensor
    fn reshape_invalid() {
        let a = delta::tensor(&[0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.], &[3, 4]);
        a.reshape(&[4, 5]);
    }

    #[test]
    /// View the tensor
    fn view() {
        let a = delta::tensor(&[0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.], &[3, 4]);
        let b = a.view(&[1, 12]);
        assert_eq!(b.shape, vec![1, 12]);
        let c = a.view(&[12, 1]);
        assert_eq!(c.shape, vec![12, 1]);
        let d = a.view(&[2, 6]);
        assert_eq!(d.shape, vec![2, 6]);
        let e = a.view(&[1, 3, 4]);
        assert_eq!(e.shape, vec![1, 3, 4]);
        let f = a.view(&[2, 2, 3]);
        assert_eq!(f.shape, vec![2, 2, 3]);
    }

    #[test]
    #[should_panic]
    /// Invalid view of the tensor
    fn view_invalid() {
        let a = delta::tensor(&[0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.], &[3, 4]);
        a.view(&[4, 5]);
    }

    #[test]
    #[should_panic]
    fn view_invalid_2() {
        let a = delta::arange(0., 9., 1.).view(&[3, 1, 3]);
        let a = a.expand(&[3, 2, 3]);
        a.view(&[2, 3, 3]);
    }

    #[test]
    fn pow() {
        let a = delta::tensor(&[0., 1., 2., 3., 4., 5.], &[2, 3]);
        let b = delta::tensor(&[0., 1., 4., 9., 16., 25.], &[2, 3]);
        assert_eq!(a.pow(2).item(), b.item(), "Pow is wrong");
    }

    #[test]
    fn neg() {
        let a = delta::arange(0., 10., 1.0);
        let b = -a.clone();
        assert_eq!(b, delta::arange(0., -10., -1.0));
    }

    #[test]
    fn randn_macro() {
        let a = randn![2, 3];
        let b = delta::randn(&[2, 3]);
        assert_eq!(a.length(), b.length());
        assert_eq!(a.shape, b.shape);
    }

    #[test]
    fn stride() {
        let a = delta::ones(&[1, 1, 3, 1, 3, 3]);
        assert_eq!(a.stride, vec![27, 27, 9, 9, 3, 1]);

        let a = a.expand(&[2, 2, 3, 3, 3, 3]);
        assert_eq!(a.stride, vec![0, 0, 9, 0, 3, 1]);

        let a = delta::ones(&[4, 1]);
        assert_eq!(a.stride, vec![1, 1]);

        let a = a.expand(&[4, 5]);
        assert_eq!(a.stride, vec![1, 0]);

        let a = delta::ones(&[4, 1, 1]);
        assert_eq!(a.stride, vec![1, 1, 1]);

        let a = a.expand(&[4, 3, 5]);
        assert_eq!(a.stride, vec![1, 0, 0]);
    }

    #[test]
    fn storage() {
        let a = delta::ones(&[2, 3, 4]);
        let b: Vec<f64> = vec![1.0; 2 * 3 * 4];
        assert_eq!(a.storage(), b);
    }

    #[test]
    fn expand() {
        let a = delta::ones(&[1, 1, 3, 1, 3, 3]);
        let b = a.expand(&[2, 2, 3, 3, 3, 3]);
        assert_eq!(a.shape, vec![1, 1, 3, 1, 3, 3]);
        assert_eq!(b.shape, vec![2, 2, 3, 3, 3, 3]);
        assert_eq!(a.stride, vec![27, 27, 9, 9, 3, 1]);
        assert_eq!(b.stride, vec![0, 0, 9, 0, 3, 1]);
    }

    #[test]
    fn unsqueeze() {
        let a = randn!(2, 3, 4);
        let b = a.unsqueeze(1);
        assert_eq!(a.shape, vec![2, 3, 4]);
        assert_eq!(b.stride, vec![12, 4, 4, 1]);
        assert_eq!(b.shape, vec![2, 1, 3, 4]);
        let c = b.unsqueeze(4);
        assert_eq!(a.shape, vec![2, 3, 4]);
        assert_eq!(b.shape, vec![2, 1, 3, 4]);
        assert_eq!(c.shape, vec![2, 1, 3, 4, 1]);
        let d = a.unsqueeze(0);
        assert_eq!(d.shape, vec![1, 2, 3, 4]);
    }

    #[test]
    fn squeeze() {
        let a = randn!(2, 1, 3, 4, 1);
        let b = a.squeeze(&[]);
        assert_eq!(a.shape, vec![2, 1, 3, 4, 1]);
        assert_eq!(b.shape, vec![2, 3, 4]);
        let c = a.squeeze(&[1, 2]);
        assert_eq!(a.shape, vec![2, 1, 3, 4, 1]);
        assert_eq!(b.shape, vec![2, 3, 4]);
        assert_eq!(c.shape, vec![2, 3, 4, 1]);
    }

    #[test]
    fn eye() {
        let expectation = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let actual = delta::eye(4).item();
        assert_eq!(actual, expectation);
    }

    #[test]
    fn cat() {
        let exp = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let a = delta::ones(&[2, 4]);
        let b = delta::zeros(&[1, 4]);
        let c = delta::cat(&[a, b], 0);
        assert_eq!(c.item(), exp);

        let exp = vec![1., 1., 1., 1., 0., 0., 0., 0.];
        let a = delta::ones(&[1, 4]);
        let b = delta::zeros(&[1, 4]);
        let c = delta::cat(&[a, b], -1);
        assert_eq!(c.item(), exp);
    }

    #[test]
    fn sum() {
        let a = delta::ones(&[1, 4]).sum(Some(1), false);
        let a_e = vec![4.];
        assert_eq!(a.item(), a_e);
        let b = delta::ones(&[4, 4]).sum(Some(0), false);
        let b_e = vec![4., 4., 4., 4.];
        assert_eq!(b.item(), b_e);
    }

    #[test]
    fn mean() {
        let a = delta::arange(0., 9., 1.).reshape(&[3, 3]);
        let a0 = a.reshape(&[9]).mean(Some(0), false);
        let a0_e = vec![4.];
        assert_eq!(a0.item(), a0_e);
        let a1 = a.mean(Some(0), false);
        let a1_e = vec![3., 4., 5.];
        assert_eq!(a1.item(), a1_e);
        let a2 = a.mean(Some(1), false);
        let a2_e = vec![1., 4., 7.];
        assert_eq!(a2.item(), a2_e);
    }
}
