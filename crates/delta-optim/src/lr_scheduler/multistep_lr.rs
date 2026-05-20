use delta::FloatTensorRepr;

use crate::Optim;

use super::Scheduler;

pub struct MultiStepLR<T: FloatTensorRepr> {
    optimizer: Box<dyn Optim<T>>,
    pub milestones: Vec<usize>,
    pub gamma: T,
    count: usize,
}

impl<T: FloatTensorRepr> MultiStepLR<T> {
    pub fn new(optimizer: Box<dyn Optim<T>>, milestones: &[usize], gamma: T) -> Self {
        Self {
            optimizer,
            milestones: milestones.to_vec(),
            gamma: gamma,
            count: 0,
        }
    }
}

impl<T: FloatTensorRepr> Scheduler for MultiStepLR<T> {
    fn step(&mut self) -> () {
        for m in &self.milestones {
            if self.count == *m {
                self.optimizer.as_mut().change_lr(self.gamma);
            }
        }
        self.optimizer.as_mut().step();
        self.count += 1;
    }
}
