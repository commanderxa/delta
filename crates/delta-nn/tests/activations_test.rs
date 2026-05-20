#[cfg(test)]
mod tests {
    use delta::nn::functional as F;

    #[test]
    fn sigmoid() {
        let t = delta::tensor!([0.0]).cast(delta::float32);
        let t_act = F::sigmoid(t);
        assert_eq!(t_act.data::<f32>()[0], 0.5);
    }

    #[test]
    fn relu() {
        let t = delta::tensor!([0.0]).cast(delta::float32);
        let t_act = F::relu(t);
        assert_eq!(t_act.data()[0], 0.0);

        let t = delta::tensor!([-20.0]).cast(delta::float32);
        let t_act = F::relu(t);
        assert_eq!(t_act.data()[0], 0.0);

        let t = delta::tensor!([100.0]).cast(delta::float32);
        let t_act = F::relu(t);
        assert_eq!(t_act.data()[0], 100.0);
    }
}
