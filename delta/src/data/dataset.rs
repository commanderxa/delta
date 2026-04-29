/// # Dataset Trait
///
/// Defines the behavior of `Dataset`.
/// Methods:
/// - len - returns the length of the dataset. E.g. number of rows from a table
/// - sample - returns a sample given the index
pub trait Dataset<T> {
    /// Returns length of `Dataset`
    fn len(&self) -> usize;

    /// Checks whether the `Dataset` is empty or not
    /// when method `len` is implemented a good practice is to implement
    /// `is_empty` also.
    fn is_empty(&self) -> bool {
        if self.len() == 0 {
            return true;
        }
        false
    }

    /// Provides a Sample of <T> given the index
    fn sample(&self, index: usize) -> T;
}
