#[cfg(test)]
mod tests {
    use delta::{nn::functional as F, Tensor};

    #[test]
    fn sigmoid() {
        let t = Tensor::tensor(&[0.0], &[1]);
        let t_act = F::sigmoid(t);
        assert_eq!(t_act.item()[0], 0.5);
    }

    #[test]
    fn relu() {
        let t = Tensor::tensor(&[0.0], &[1]);
        let t_act = F::relu(t);
        assert_eq!(t_act.item()[0], 0.0);

        let t = Tensor::tensor(&[-20.0], &[1]);
        let t_act = F::relu(t);
        assert_eq!(t_act.item()[0], 0.0);

        let t = Tensor::tensor(&[100.0], &[1]);
        let t_act = F::relu(t);
        assert_eq!(t_act.item()[0], 100.0);
    }
}
