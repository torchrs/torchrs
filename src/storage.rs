use rutorch::*;
use std::ops::{Index, IndexMut};
use std::slice;

macro_rules! impl_storage_impl {
    ($name:ident, $type:ident, $thname:ident) => {
        pub struct $name {
            pub t: *mut $thname,
        }
        impl $name {
            pub fn new() -> Self {
                unsafe { $name { t: concat_idents!($thname, _new)() } }
            }
            pub fn with_capacity(size: usize) -> Self {
                unsafe { $name { t: concat_idents!($thname, _newWithSize)(size as isize) } }
            }
            pub fn into_slice(&self) -> &[$type] {
                unsafe { slice::from_raw_parts((*self.t).data, (*self.t).size as usize) }
            }
            pub fn into_slice_mut(&mut self) -> &mut [$type] {
                unsafe { slice::from_raw_parts_mut((*self.t).data, (*self.t).size as usize) }
            }
            pub fn iter(&self) -> slice::Iter<$type> {
                self.into_slice().iter()
            }
            pub fn iter_mut(&mut self) -> slice::IterMut<$type> {
                self.into_slice_mut().iter_mut()
            }
        }
        impl Drop for $name {
            fn drop(&mut self) {
                unsafe { concat_idents!($thname, _free)(self.t) }
            }
        }
        impl Index<isize> for $name {
            type Output = $type;
            fn index(&self, idx: isize) -> &Self::Output {
                unsafe { &*(*self.t).data.offset(idx) }
            }
        }
        impl IndexMut<isize> for $name {
            fn index_mut(&mut self, idx: isize) -> &mut Self::Output {
                unsafe { &mut *(*self.t).data.offset(idx) }
            }
        }
    }
}

impl_storage_impl!(FloatStorage, f32, THFloatStorage);
impl_storage_impl!(DoubleStorage, f64, THDoubleStorage);
impl_storage_impl!(LongStorage, i64, THLongStorage);
impl_storage_impl!(ByteStorage, u8, THByteStorage);
