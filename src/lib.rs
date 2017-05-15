#![feature(trace_macros)]
#![feature(log_syntax)]

extern crate num;
extern crate rand;
extern crate rutorch;
#[macro_use]
extern crate modparse_derive;
extern crate linked_hash_map;


mod nn;
use rutorch::*;
use nn::*;
use std::ops::{Index, IndexMut};
//use std::convert::From;
use std::cmp::max;
use std::slice;

trait Storage {
    fn new() -> Self;
    fn sized(len: isize) -> Self;
}

pub struct FloatStorage {
    t: *mut THFloatStorage,
}

impl Storage for FloatStorage {
    fn new() -> Self {
        unsafe { FloatStorage { t: THFloatStorage_new() } }
    }
    fn sized(size: isize) -> Self {
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

trait Tensor<'a> {
    fn new() -> Self;
    fn sized(size: &'a [isize]) -> Self;
    fn randn(dims: &'a [isize]) -> Self;
}

pub struct FloatTensor {
    t: *mut THFloatTensor,
    storage: FloatStorage,
    dims: Vec<isize>,
}

pub struct Parameter {}

impl<'a> Tensor<'a> for FloatTensor {
    fn new() -> Self {
        unsafe {
            FloatTensor {
                t: THFloatTensor_new(),
                storage: FloatStorage::new(),
                dims: Vec::new(),
            }
        }
    }

    fn sized(dims: &'a [isize]) -> Self {
        let size = dims.iter().product();
        let storage = FloatStorage::sized(size);
        let strides = vec![1; dims.len()];
        let mut t = THFloatTensor {
            size: dims.clone().as_ptr() as *mut ::std::os::raw::c_long,
            stride: strides.as_ptr() as *mut ::std::os::raw::c_long,
            nDimension: dims.len() as i32,
            storage: storage.t,
            storageOffset: 0,
            refcount: 1,
            flag: TH_TENSOR_REFCOUNTED as i8,
        };
        FloatTensor {
            t: &mut t,
            storage: storage,
            dims: Vec::from(dims),
        }
    }
    fn randn(dims: &'a [isize]) -> Self {
        /* XXX */
        let mut t = FloatTensor::sized(dims);
        for x in t.storage.iter_mut() {
            *x = rand::random::<f32>()
        }
        t
    }
}

impl<'a> Index<&'a [isize]> for FloatTensor {
    type Output = f32;

    fn index(&self, idx: &'a [isize]) -> &f32 {
        let mut index: isize = 0;
        let lastidx = max(0, idx.len() as isize - 1) as usize;
        if idx.len() != self.dims.len() {
            panic!("bad dimlen")
        }
        for i in 0..lastidx {
            if idx[i] >= self.dims[i] {
                panic!("bad dimlen")
            }
            index += idx[i] * self.dims[i]
        }
        if idx[lastidx] >= self.dims[lastidx] {
            panic!("bad dimlen")
        }
        index += idx[lastidx];
        &self.storage[index]
    }
}

impl<'a> IndexMut<&'a [isize]> for FloatTensor {
    fn index_mut(&mut self, idx: &'a [isize]) -> &mut f32 {
        let mut index: isize = 0;
        let lastidx = max(0, idx.len() as isize - 1) as usize;
        if idx.len() != self.dims.len() {
            panic!("bad dimlen")
        }
        for i in 0..lastidx {
            if idx[i] >= self.dims[i] {
                panic!("bad dimlen")
            }
            index += idx[i] * self.dims[i]
        }
        if idx[lastidx] >= self.dims[lastidx] {
            panic!("bad dimlen")
        }
        index += idx[lastidx];
        &mut self.storage[index]
    }
}

/*
    def __init__(self):
        self._backend = thnn_backend
        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self._backward_hooks = OrderedDict()
        self._forward_hooks = OrderedDict()
        self._modules = OrderedDict()
        self.training = True
*/


/*
impl fmt::Display for FloatTensor {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match (*self.t).size 
		write!(f, "[torchrs.FloatTensor of size ]")
	}
}
*/
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_works() {}
}
