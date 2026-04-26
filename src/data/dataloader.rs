use std::{cell::RefCell, rc::Rc};

use rand::{seq::SliceRandom, rng};

use super::dataset::Dataset;

/// Inner of `Dataloader`
struct DataloaderInner<T> {
    dataset: Box<dyn Dataset<T>>,
    batch_size: usize,
    shuffle: bool,
    indices: Vec<usize>,
    index: usize,
}

impl<T> DataloaderInner<T> {
    /// Set index to 0
    fn reset_index(&mut self) {
        self.index = 0;
    }

    /// Make index bigger by 1
    fn increment_index(&mut self) {
        self.index += 1;
    }

    /// Mix the indices up to obtain random sequence
    fn shuffle_indices(&mut self) {
        self.indices.shuffle(&mut rng());
    }
}

/// # Dataloader
///
/// It is used for data acquisition.
///
/// `Dataloader` implements `Iterator`.
#[derive(Clone)]
pub struct Dataloader<T>(Rc<RefCell<DataloaderInner<T>>>);

impl<T> Dataloader<T> {
    /// Create a new `Dataloader`
    pub fn new(dataset: Box<impl Dataset<T> + 'static>, batch_size: usize, shuffle: bool) -> Self {
        // create a list of indices that would correspond to the indices in real data
        let mut indices = vec![];
        for i in 0..dataset.len() {
            indices.push(i);
        }
        // shuffle all indices if it is specified so
        if shuffle {
            indices.shuffle(&mut rng());
        }
        Self(Rc::new(RefCell::new(DataloaderInner {
            dataset,
            batch_size,
            shuffle,
            indices,
            index: 0,
        })))
    }

    /// Get batch size
    pub fn batch_size(&self) -> usize {
        self.0.borrow().batch_size
    }

    /// Indicates whether data was shuffled
    pub fn is_shuffle(&self) -> bool {
        self.0.borrow().shuffle
    }

    pub fn len(&self) -> usize {
        self.0.borrow().dataset.len()
    }
}

impl<T> Iterator for Dataloader<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let mut inner = self.0.borrow_mut();
        if inner.index >= inner.indices.len() {
            // iterator is exhausted, before returning None
            // prepare for the next iteration:
            // set index to 0 and shuffle the indices
            inner.reset_index();
            inner.shuffle_indices();
            return None;
        }
        // get the next sample
        let sample = Some(inner.dataset.sample(inner.indices[inner.index]));
        // increment the index
        inner.increment_index();
        sample
    }
}
