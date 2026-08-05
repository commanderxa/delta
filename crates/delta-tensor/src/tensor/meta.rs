#[derive(Debug, Clone)]
pub struct TensorMetaData {
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

impl TensorMetaData {
    pub fn new(shape: Vec<usize>) -> Self {
        Self {
            shape: shape.clone(),
            strides: Self::compute_strides(&shape),
            offset: 0,
        }
    }

    pub(crate) fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
        strides
    }

    pub fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    pub fn set_shape(&mut self, shape: &[usize]) {
        self.shape = shape.to_vec();
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn set_offset(&mut self, offset: usize) {
        self.offset = offset;
    }

    pub fn strides(&self) -> Vec<usize> {
        self.strides.clone()
    }

    pub fn set_stride(&mut self, stride: &[usize]) {
        self.strides = stride.to_vec();
    }

    pub fn rank(&self) -> usize {
        self.shape().len()
    }
}
