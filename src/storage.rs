use rutorch::*;
use std::ops::{Index, IndexMut};
//use std::convert::From;
use std::slice;

pub trait Storage {
    fn new() -> Self;
    fn with_capacity(len: isize) -> Self;
}

pub struct FloatStorage {
    pub t: *mut THFloatStorage,
}

impl Storage for FloatStorage {
    fn new() -> Self {
        unsafe { FloatStorage { t: THFloatStorage_new() } }
    }
    fn with_capacity(size: isize) -> Self {
        unsafe { FloatStorage { t: THFloatStorage_newWithSize(size) } }
    }
}
// XXX annotate with lifetime for safety's sake
impl FloatStorage {
    fn into_slice(&self) -> &[f32] {
        unsafe { slice::from_raw_parts((*self.t).data, (*self.t).size as usize) }
    }
    fn into_slice_mut(&mut self) -> &mut [f32] {
        unsafe { slice::from_raw_parts_mut((*self.t).data, (*self.t).size as usize) }
    }
    pub fn iter(&self) -> slice::Iter<f32> {
        self.into_slice().iter()
    }
    pub fn iter_mut(&mut self) -> slice::IterMut<f32> {
        self.into_slice_mut().iter_mut()
    }
}

impl Index<isize> for FloatStorage {
    type Output = f32;

    fn index(&self, idx: isize) -> &f32 {
        unsafe { &*(*self.t).data.offset(idx) }
    }
}

impl IndexMut<isize> for FloatStorage {
    fn index_mut(&mut self, idx: isize) -> &mut f32 {
        unsafe { &mut *(*self.t).data.offset(idx) }
    }
}

//fn from(args: &ArgsArray<T>) -> Self;
