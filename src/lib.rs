extern crate num;
extern crate rutorch;

use rutorch::*;
use std::ops::Index;
//use std::convert::From;
//use num::Num;
use std::cmp::max;


trait Storage {
	fn new() -> Self;
	fn sized(len: isize) -> Self;
}

struct FloatStorage {
	t: *mut THFloatStorage,
}

impl Storage for FloatStorage {
	fn new() -> Self {
		unsafe { FloatStorage  {t : THFloatStorage_new()} }
	}
	fn sized(size: isize) ->Self {
		unsafe { FloatStorage  {t : THFloatStorage_newWithSize(size)} }
	}
}

impl Index<isize> for FloatStorage {
	type Output = f32;

	fn index(&self, idx: isize) -> &f32 {
		unsafe {&*(*self.t).data.offset(idx)}
	}
}

//fn from(args: &ArgsArray<T>) -> Self;

trait Tensor<'a> : Index<&'a [isize]> {
	fn new() -> Self;
	fn sized(size: &'a [isize]) -> Self;
	fn randn(dims: &'a [isize]) -> Self;
}

struct FloatTensor {
	t: *mut THFloatTensor,
	storage: FloatStorage,
	dims: Vec<isize>,
}

impl <'a>Tensor<'a> for FloatTensor {
	fn new() -> Self {
		unsafe { FloatTensor { t : THFloatTensor_new(), storage: FloatStorage::new(), dims: Vec::new()  }}
	}
	fn sized(dims: &'a [isize]) -> Self {
		let size = dims.iter().product();
		let storage = FloatStorage::sized(size);
		let strides = vec![1; dims.len()];
		let mut t = THFloatTensor { size : dims.clone().as_ptr() as *mut ::std::os::raw::c_long,
			stride: strides.as_ptr() as *mut ::std::os::raw::c_long,
			nDimension : dims.len() as i32, storage: storage.t, storageOffset: 0, refcount: 1, 
			flag: TH_TENSOR_REFCOUNTED as i8};
		FloatTensor {t : &mut t, storage: storage, dims: Vec::from(dims)}
	}
	fn randn(dims: &'a [isize]) -> Self {
		/* XXX */
		Tensor::sized(dims)
	}
}

impl <'a> Index<&'a [isize]> for FloatTensor {
	type Output = f32;

	fn index(&self, idx: &'a [isize]) -> &f32 {
		let mut index : isize = 0;
		let lastidx = max(0, idx.len() as isize - 1) as usize;
		if idx.len() != self.dims.len() { panic!("bad dimlen")}
		for i in 0..lastidx {
			if idx[i] >= self.dims[i] { panic!("bad dimlen")}
			index += idx[i] * self.dims[i]
		}
		if idx[lastidx] >= self.dims[lastidx] { panic!("bad dimlen")}
		index += idx[lastidx];
		&self.storage[index]
	}
}

#[cfg(test)]
mod tests {
	use super::*;
    #[test]
    fn it_works() {

    }
}
