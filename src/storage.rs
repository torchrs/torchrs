use rutorch::*;
use std::ops::{Index, IndexMut};
use std::slice;

pub struct FloatStorage {
    pub t: *mut THFloatStorage,
}
impl Drop for FloatStorage {
    fn drop(&mut self) {
        unsafe { THFloatStorage_free(self.t) }
    }
}
impl FloatStorage {
    pub fn new() -> Self {
        unsafe { FloatStorage { t: THFloatStorage_new() } }
    }
    pub fn with_capacity(size: usize) -> Self {
        unsafe { FloatStorage { t: THFloatStorage_newWithSize(size as isize) } }
    }
    pub fn into_slice(&self) -> &[f32] {
        unsafe { slice::from_raw_parts((*self.t).data, (*self.t).size as usize) }
    }
    pub fn into_slice_mut(&mut self) -> &mut [f32] {
        unsafe { slice::from_raw_parts_mut((*self.t).data, (*self.t).size as usize) }
    }
    pub fn iter(&self) -> slice::Iter<f32> {
        self.into_slice().iter()
    }
    pub fn iter_mut(&mut self) -> slice::IterMut<f32> {
        self.into_slice_mut().iter_mut()
    }
}

pub struct LongStorage {
    pub t: *mut THLongStorage,
}
impl Drop for LongStorage {
    fn drop(&mut self) {
        unsafe { THLongStorage_free(self.t) }
    }
}
impl LongStorage {
    pub fn new() -> Self {
        unsafe { LongStorage { t: THLongStorage_new() } }
    }
    pub fn with_capacity(size: usize) -> Self {
        unsafe { LongStorage { t: THLongStorage_newWithSize(size as isize) } }
    }
    pub fn into_slice(&self) -> &[i64] {
        unsafe { slice::from_raw_parts((*self.t).data, (*self.t).size as usize) }
    }
    pub fn into_slice_mut(&mut self) -> &mut [i64] {
        unsafe { slice::from_raw_parts_mut((*self.t).data, (*self.t).size as usize) }
    }
    pub fn iter(&self) -> slice::Iter<i64> {
        self.into_slice().iter()
    }
    pub fn iter_mut(&mut self) -> slice::IterMut<i64> {
        self.into_slice_mut().iter_mut()
    }
}

impl Index<isize> for FloatStorage {
    type Output = f32;
    fn index(&self, idx: isize) -> &Self::Output {
        unsafe { &*(*self.t).data.offset(idx) }
    }
}
impl IndexMut<isize> for FloatStorage {
    fn index_mut(&mut self, idx: isize) -> &mut Self::Output {
        unsafe { &mut *(*self.t).data.offset(idx) }
    }
}

impl Index<isize> for LongStorage {
    type Output = i64;
    fn index(&self, idx: isize) -> &Self::Output {
        unsafe { &*(*self.t).data.offset(idx) }
    }
}
impl IndexMut<isize> for LongStorage {
    fn index_mut(&mut self, idx: isize) -> &mut Self::Output {
        unsafe { &mut *(*self.t).data.offset(idx) }
    }
}


//fn from(args: &ArgsArray<T>) -> Self;
