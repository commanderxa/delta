use crate::{DType, Tensor};

pub(crate) fn promote_(tensors: &mut [&mut Tensor]) {
    let dtypes: Vec<DType> = tensors.iter().map(|x| x.dtype()).collect();
    if !dtypes.iter().all(|x| x == &dtypes[0]) {
        let mut dom_dtype = dtypes[0];
        for dt in dtypes {
            if dt.rank() == dom_dtype.rank() {
                if dt.itemsize() >= dom_dtype.itemsize() {
                    dom_dtype = dt;
                } else {
                    continue;
                }
            } else if dt.rank() > dom_dtype.rank() {
                if dt.itemsize() >= dom_dtype.itemsize() {
                    dom_dtype = dt;
                } else {
                    todo!()
                }
            } else {
                if dt.itemsize() >= dom_dtype.itemsize() {
                    todo!()
                } else {
                    continue;
                }
            }
        }
        for t in tensors {
            if t.dtype() != dom_dtype {
                t.cast_(dom_dtype);
            }
        }
    }
}

#[macro_export]
macro_rules! promote_tensors {
    ($($tensor:expr),+) => {
        crate::tensor::promote::promote_(&mut [$($tensor),+])
    };
}
