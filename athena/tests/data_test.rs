#[cfg(test)]
mod tests {
    use athena::{
        data::{dataloader::Dataloader, dataset::Dataset, sample::Sample},
        Tensor,
    };

    struct MyDataset {
        data: Vec<Sample>,
    }

    impl MyDataset {
        fn new() -> Self {
            Self {
                data: vec![
                    (Tensor::tensor(&[0.0], &[1]), Tensor::tensor(&[0.0], &[1])),
                    (Tensor::tensor(&[1.0], &[1]), Tensor::tensor(&[2.0], &[1])),
                    (Tensor::tensor(&[2.0], &[1]), Tensor::tensor(&[4.0], &[1])),
                    (Tensor::tensor(&[3.0], &[1]), Tensor::tensor(&[6.0], &[1])),
                    (Tensor::tensor(&[4.0], &[1]), Tensor::tensor(&[8.0], &[1])),
                    (Tensor::tensor(&[5.0], &[1]), Tensor::tensor(&[10.0], &[1])),
                ],
            }
        }
    }

    impl Dataset<Sample> for MyDataset {
        fn len(&self) -> usize {
            self.data.len()
        }

        fn sample(&self, index: usize) -> Sample {
            self.data[index].clone()
        }
    }

    #[test]
    fn dataset() {
        let _ = MyDataset::new();
    }

    #[test]
    fn dataloader() {
        let dataset = MyDataset::new();
        let dataloader = Dataloader::new(Box::new(dataset), 1, true);
        for (x, y) in dataloader.clone() {
            assert_eq!((x * 2 as i64).item(), y.item());
        }
    }
}
