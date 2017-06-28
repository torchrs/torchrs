use rutorch::*;
use std::ops::{Index, IndexMut};
use std::slice;


impl_storage_impl!(FloatStorage, f32, THFloatStorage);
impl_storage_impl!(LongStorage, i64, THLongStorage);
impl_storage_impl!(ByteStorage, u8, THByteStorage);


//fn from(args: &ArgsArray<T>) -> Self;
