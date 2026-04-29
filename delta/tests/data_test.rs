#[cfg(test)]
mod tests {
    use delta::data::{dataloader::Dataloader, dataset::Dataset, sample::Sample};

    struct MyDataset {
        data: Vec<Sample>,
    }

    impl MyDataset {
        fn new() -> Self {
            Self {
                data: vec![
                    (delta::tensor(&[0.0], &[1]), delta::tensor(&[0.0], &[1])),
                    (delta::tensor(&[1.0], &[1]), delta::tensor(&[2.0], &[1])),
                    (delta::tensor(&[2.0], &[1]), delta::tensor(&[4.0], &[1])),
                    (delta::tensor(&[3.0], &[1]), delta::tensor(&[6.0], &[1])),
                    (delta::tensor(&[4.0], &[1]), delta::tensor(&[8.0], &[1])),
                    (delta::tensor(&[5.0], &[1]), delta::tensor(&[10.0], &[1])),
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
