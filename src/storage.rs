use rutorch::*;
use std::ops::{Index, IndexMut};
use std::{slice, ops};

macro_rules! impl_storage_impl {
    ($name:ident, $type:ident, $thname:ident) => {
        pub struct $name {
            pub t: *mut $thname,
        }
        impl Clone for $name {
            fn clone(&self) -> Self {
                unsafe { concat_idents!($thname, _retain)(self.t)};
                $name {t: self.t }
            }
        }
        impl $name {
            pub fn new() -> Self {
                unsafe { $name { t: concat_idents!($thname, _new)() } }
            }
            pub fn len(&self) -> usize {
                let t = unsafe {(*self.t).size };
                t as usize
            }
            pub fn with_capacity(size: usize) -> Self {
                unsafe { $name { t: concat_idents!($thname, _newWithSize)(size as isize) } }
            }
            pub fn with_data<D>(data: D) -> Self
                where D: AsRef<[$type]>
            {
                let data = data.as_ref();
                let mut store = unsafe { $name {
                    t: concat_idents!($thname, _newWithSize)(data.len() as isize)
                } };
                for i in 0..data.len() {
                    store[i] = data[i];
                }
                store
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
        impl ops::Deref for $name {
            type Target = [$type];

            fn deref(&self) -> &Self::Target {
                self.into_slice()
            }
        }
        impl ops::DerefMut for $name {
            fn deref_mut (&mut self) -> &mut Self::Target {
                self.into_slice_mut()
            }
        }
        impl Drop for $name {
            fn drop(&mut self) {
                unsafe { concat_idents!($thname, _free)(self.t) }
            }
        }
        impl Index<usize> for $name {
            type Output = $type;
            fn index(&self, idx: usize) -> &Self::Output {
                &(**self)[idx]
            }
        }
        impl Index<ops::Range<usize>> for $name {
            type Output = [$type];
            fn index(&self, idx: ops::Range<usize>) -> &Self::Output {
                Index::index(&**self, idx)
            }
        }
        impl Index<ops::RangeTo<usize>> for $name {
            type Output = [$type];
            fn index(&self, idx: ops::RangeTo<usize>) -> &Self::Output {
                Index::index(&**self, idx)
            }
        }
        impl Index<ops::RangeFrom<usize>> for $name {
            type Output = [$type];
            fn index(&self, idx: ops::RangeFrom<usize>) -> &Self::Output {
                Index::index(&**self, idx)
            }
        }
        impl Index<ops::RangeFull> for $name {
            type Output = [$type];
            fn index(&self, _idx: ops::RangeFull) -> &Self::Output {
                self
            }
        }
        impl IndexMut<usize> for $name {
            fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
                &mut (**self)[idx]
            }
        }
    }
}

impl_storage_impl!(FloatStorage, f32, THFloatStorage);
impl_storage_impl!(DoubleStorage, f64, THDoubleStorage);
impl_storage_impl!(LongStorage, i64, THLongStorage);
impl_storage_impl!(ByteStorage, u8, THByteStorage);
