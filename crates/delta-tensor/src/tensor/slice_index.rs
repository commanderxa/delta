pub enum SliceIndexEnum {
    Range(SliceIndex),
    Full,
    Index(isize),
}

impl From<usize> for SliceIndexEnum {
    fn from(i: usize) -> Self {
        SliceIndexEnum::Index(i as isize)
    }
}

impl From<isize> for SliceIndexEnum {
    fn from(i: isize) -> Self {
        SliceIndexEnum::Index(i)
    }
}

impl From<std::ops::Range<usize>> for SliceIndexEnum {
    fn from(r: std::ops::Range<usize>) -> Self {
        SliceIndexEnum::Range(SliceIndex { start: r.start as isize, end: r.end as isize, step: 1 })
    }
}

impl From<std::ops::Range<i32>> for SliceIndexEnum {
    fn from(r: std::ops::Range<i32>) -> Self {
        SliceIndexEnum::Range(SliceIndex { start: r.start as isize, end: r.end as isize, step: 1 })
    }
}

impl From<std::ops::RangeFull> for SliceIndexEnum {
    fn from(_: std::ops::RangeFull) -> Self {
        SliceIndexEnum::Full
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SliceIndex {
    pub start: isize,
    pub end: isize,
    pub step: isize,
}

impl SliceIndex {
    pub(crate) fn map_negative(index: isize, dim_len: isize) -> isize {
        if index >= 0 {
            index
        } else {
            let corrected = dim_len + index;
            if corrected < 0 {
                panic!("Index `{}` out of bounds!", index);
            } else {
                corrected
            }
        }
    }
}

impl From<std::ops::Range<isize>> for SliceIndex {
    fn from(r: std::ops::Range<isize>) -> Self {
        Self {
            start: r.start,
            end: r.end,
            step: 1,
        }
    }
}

impl From<std::ops::Range<i32>> for SliceIndex {
    fn from(r: std::ops::Range<i32>) -> Self {
        Self {
            start: r.start as isize,
            end: r.end as isize,
            step: 1,
        }
    }
}

pub trait SliceIndexArg {
    fn as_slice_arg(&self) -> Vec<SliceIndexEnum>;
}

impl<T, const N: usize> SliceIndexArg for [T; N]
where
    T: Into<SliceIndexEnum> + Clone,
{
    fn as_slice_arg(&self) -> Vec<SliceIndexEnum> {
        self.iter().map(|s| s.clone().into()).collect()
    }
}