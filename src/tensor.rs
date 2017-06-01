use rutorch::*;
use std::ops::{Index, IndexMut};
//use std::convert::From;
use std::cmp::max;

use storage::*;
use rand;
use {Ixs, RcMut};

pub enum TensorKind {
    FloatTensor(Tensor<f32>),
    LongTensor(Tensor<i64>),
}

pub type TensorList<T> = Vec<Tensor<T>>;
pub type TensorKindList = Vec<TensorKind>;
pub type RefTensorList<'a, T> = Vec<&'a mut Tensor<T>>;
pub type RefTensorKindList<'a> = Vec<&'a mut TensorKind>;
pub type TensorId = i32;

pub struct Tensor<T> {
    pub id: i32,
    value: RcMut<TensorImpl<T, Output = T>>,
}

pub trait New<D, T> {
    fn new(args: D) -> T;
    fn new_(&self, args: D) -> T;
}

impl<T> New<usize, Tensor<T>> for Tensor<T> {
    fn new(args: usize) -> Self {
        unimplemented!()
    }
    fn new_(&self, args: usize) -> Self {
        unimplemented!()
    }
}

impl<T> Tensor<T> {
    pub fn len(&self) -> usize {
        unimplemented!()
    }
    pub fn cuda(&self) -> Self {
        unimplemented!()
    }
    pub fn cpu(&self) -> Self {
        unimplemented!()
    }
    pub fn reduce_max(&self, axis: usize) -> (Self, Tensor<i64>) {
        unimplemented!()
    }
    pub fn tensor_eq(&self, rhs: &Self) -> Tensor<i64> {
        unimplemented!()
    }
    pub fn sum(&self) -> u32 {
        unimplemented!()
    }
    pub fn s(&self, dim: usize) -> Self {
        unimplemented!()
    }
}

impl<T> Default for Tensor<T> {
    fn default() -> Self {
        unimplemented!()
    }
}

impl<T> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        Tensor {
            id: self.id,
            value: self.value.clone(),
        }
    }
}

impl<T: Copy> Index<isize> for Tensor<T> {
    type Output = T;

    fn index(&self, idx: isize) -> &Self::Output {
        unimplemented!()
    }
}
pub trait TensorImpl<T>: Index<Ixs, Output = T> {
    //fn view<'a>(&self, dims: &[i32]) -> Tensor<'a>;
}


pub struct FloatTensor {
    t: *mut THFloatTensor,
    storage: FloatStorage,
    dims: Vec<isize>,
}

impl Default for FloatTensor {
    fn default() -> Self {
        FloatTensor::new()
    }
}


impl FloatTensor {
    pub fn new() -> Self {
        unsafe {
            FloatTensor {
                t: THFloatTensor_new(),
                storage: FloatStorage::new(),
                dims: Vec::new(),
            }
        }
    }
    pub fn with_capacity(dims: &[isize]) -> Self {
        let size = dims.iter().product();
        let storage = FloatStorage::with_capacity(size);
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
    pub fn randn(dims: &[isize]) -> Self {
        /* XXX */
        let mut t = FloatTensor::with_capacity(dims);
        for x in t.storage.iter_mut() {
            *x = rand::random::<f32>()
        }
        t
    }
}

impl<'a> Index<&'a [isize]> for FloatTensor {
    type Output = f32;

    fn index(&self, idx: &'a [isize]) -> &Self::Output {
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
    fn index_mut(&mut self, idx: &'a [isize]) -> &mut Self::Output {
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
