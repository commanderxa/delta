#[cfg(test)]
mod tests {
    use delta_nn::functional as F;

    #[test]
    fn sigmoid() {
        let t = delta_tensor::tensor!([0.0]).cast(delta_tensor::float32);
        let t_act = F::sigmoid(t);
        assert_eq!(t_act.data::<f32>()[0], 0.5);
    }

    #[test]
    fn relu() {
        let t = delta_tensor::tensor!([0.0]).cast(delta_tensor::float32);
        let t_act = F::relu(t);
        assert_eq!(t_act.data()[0], 0.0);

        let t = delta_tensor::tensor!([-20.0]).cast(delta_tensor::float32);
        let t_act = F::relu(t);
        assert_eq!(t_act.data()[0], 0.0);

        let t = delta_tensor::tensor!([100.0]).cast(delta_tensor::float32);
        let t_act = F::relu(t);
        assert_eq!(t_act.data()[0], 100.0);
    }
}
