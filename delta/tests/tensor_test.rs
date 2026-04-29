#[cfg(test)]
mod tests {
    use delta::{randn, Tensor};

    #[test]
    /// Valid shape of the tensor
    fn valid_shape() {
        Tensor::tensor(&[0., 1., 2., 3., 4., 5.], &[2, 3]);
        Tensor::tensor(&[0., 1., 2., 3., 4., 5.], &[3, 2]);
        Tensor::tensor(&[0., 1., 2., 3., 4., 5.], &[1, 6]);
        Tensor::tensor(&[0., 1., 2., 3., 4., 5.], &[6, 1]);
    }

    #[test]
    #[should_panic]
    /// Invalid shape of the tensor
    fn invalid_shape() {
        Tensor::tensor(&[0., 1., 2., 3., 4., 5.], &[1, 3]);
    }

    #[test]
    fn zeros_like() {
        let a = Tensor::tensor(&[0., 1., 2., 3., 4., 5.], &[2, 3]);
        let a1 = Tensor::zeros_like(&a);
        assert_eq!(0., a1.item().iter().sum(), "zeros_like produces not zeros");
        assert_eq!(
            a.shape, a1.shape,
            "zeros_like produce wrong shape of a tensor"
        );
        let b = Tensor::randn(&[4, 10, 8]);
        let b1 = Tensor::zeros_like(&b);
        assert_eq!(0., b1.item().iter().sum(), "zeros_like produces not zeros");
        assert_eq!(
            b.shape, b1.shape,
            "zeros_like produce wrong shape of a tensor"
        );
    }

    #[test]
    fn ones_like() {
        let a = Tensor::tensor(&[0., 1., 2., 3., 4., 5.], &[2, 3]);
        let a1 = Tensor::ones_like(&a);
        assert_eq!(
            1.,
            a1.item().iter().product(),
            "ones_like produces not ones"
        );
        assert_eq!(
            a.shape, a1.shape,
            "ones_like produce wrong shape of a tensor"
        );
        let b = Tensor::randn(&[4, 10, 8]);
        let b1 = Tensor::ones_like(&b);
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
        let a = Tensor::tensor(&[0., 1., 2., 3., 4., 5.], &[2, 3]);
        let t = Tensor::tensor(&[0., 3., 1., 4., 2., 5.], &[3, 2]);
        assert_eq!(a.t().shape, t.shape, "Shapes are wrong");
        assert_eq!(a.t().item(), t.item(), "Data is wrong");
    }

    #[test]
    /// New tensor of ordered numbers
    fn arange() {
        let a = Tensor::arange(0., 6., 1.);
        assert_eq!(a.shape, vec![a.length()], "Shape is wrong");
        assert_eq!(a.item(), vec![0., 1., 2., 3., 4., 5.], "Data is wrong");
    }

    #[test]
    /// Reshape the tensor
    fn reshape() {
        let a = Tensor::tensor(&[0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.], &[3, 4]);
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
    }

    #[test]
    fn reshape_2() {
        let a = Tensor::arange(0., 9., 1.).view(&[3, 1, 3]);
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
            vec![0., 1., 2., 0., 1., 2., 3., 4., 5., 3., 4., 5., 6., 7., 8., 6., 7., 8.0]
        );
    }

    #[test]
    #[should_panic]
    /// Invalid view of the tensor
    fn reshape_invalid() {
        let a = Tensor::tensor(&[0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.], &[3, 4]);
        a.reshape(&[4, 5]);
    }

    #[test]
    /// View the tensor
    fn view() {
        let a = Tensor::tensor(&[0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.], &[3, 4]);
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
        let a = Tensor::tensor(&[0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.], &[3, 4]);
        a.view(&[4, 5]);
    }

    #[test]
    #[should_panic]
    fn view_invalid_2() {
        let a = Tensor::arange(0., 9., 1.).view(&[3, 1, 3]);
        let a = a.expand(&[3, 2, 3]);
        a.view(&[2, 3, 3]);
    }

    #[test]
    fn pow() {
        let a = Tensor::tensor(&[0., 1., 2., 3., 4., 5.], &[2, 3]);
        let b = Tensor::tensor(&[0., 1., 4., 9., 16., 25.], &[2, 3]);
        assert_eq!(a.pow(2).item(), b.item(), "Pow is wrong");
    }

    #[test]
    fn neg() {
        let a = Tensor::arange(0., 10., 1.0);
        let b = -a.clone();
        assert_eq!(b, Tensor::arange(0., -10., -1.0));
    }

    #[test]
    fn randn_macro() {
        let a = randn![2, 3];
        let b = Tensor::randn(&[2, 3]);
        assert_eq!(a.length(), b.length());
        assert_eq!(a.shape, b.shape);
    }

    #[test]
    fn stride() {
        let a = Tensor::ones(&[1, 1, 3, 1, 3, 3]);
        assert_eq!(a.stride, vec![27, 27, 9, 9, 3, 1]);

        let a = a.expand(&[2, 2, 3, 3, 3, 3]);
        assert_eq!(a.stride, vec![0, 0, 9, 0, 3, 1]);

        let a = Tensor::ones(&[4, 1]);
        assert_eq!(a.stride, vec![1, 1]);

        let a = a.expand(&[4, 5]);
        assert_eq!(a.stride, vec![1, 0]);

        let a = Tensor::ones(&[4, 1, 1]);
        assert_eq!(a.stride, vec![1, 1, 1]);

        let a = a.expand(&[4, 3, 5]);
        assert_eq!(a.stride, vec![1, 0, 0]);
    }

    #[test]
    fn storage() {
        let a = Tensor::ones(&[2, 3, 4]);
        let b = vec![1.0; 2 * 3 * 4];
        assert_eq!(a.storage(), b);
    }

    #[test]
    fn expand() {
        let a = Tensor::ones(&[1, 1, 3, 1, 3, 3]);
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
}
