// while a WIP
#![allow(unused_variables)]
use rutorch::*;
use std::ops::{Index, IndexMut};
use std::convert::From;
use std::cmp::max;
use std::hash::{Hash, Hasher};

use storage::*;
use rand;
use {Ixs, RcMut};


#[derive(Hash)]
pub enum TensorKind {
    FloatTensor(Tensor<f32>),
    LongTensor(Tensor<i64>),
}
pub enum NumKind {
    Float(f32),
    Double(f64),
    Long(i64),
}


pub type TensorList<T> = Vec<Tensor<T>>;
pub type TensorKindList = Vec<TensorKind>;
pub type RefTensorList<'a, T> = Vec<&'a mut Tensor<T>>;
pub type RefTensorKindList<'a> = Vec<&'a TensorKind>;
pub type TensorId = i32;

impl TensorKind {
    pub fn abs(&self) -> Self {
        unimplemented!()
    }
    pub fn abs_(self) -> Self {
        unimplemented!()
    }
    pub fn acos(&self) -> Self {
        unimplemented!()
    }
    pub fn acos_(self) -> Self {
        unimplemented!()
    }
    pub fn add(&self, rhs: &NumKind) -> Self {
        unimplemented!()
    }
    pub fn add_(self, rhs: &NumKind) -> Self {
        unimplemented!()
    }
    pub fn addbmm(&self, beta: &NumKind, alpha: &NumKind, tensor1: &Self, tensor2: &Self) -> Self {
        unimplemented!()
    }
    pub fn addbmm_(self, beta: &NumKind, alpha: &NumKind, tensor1: &Self, tensor2: &Self) -> Self {
        unimplemented!()
    }
    pub fn addcdiv(&self, value: &NumKind, tensor1: &Self, tensor2: &Self) -> Self {
        unimplemented!()
    }
    pub fn addcdiv_(self, value: &NumKind, tensor1: &Self, tensor2: &Self) -> Self {
        unimplemented!()
    }
    pub fn addcmul(&self, value: &NumKind, tensor1: &Self, tensor2: &Self) -> Self {
        unimplemented!()
    }
    pub fn addcmul_(self, value: &NumKind, tensor1: &Self, tensor2: &Self) -> Self {
        unimplemented!()
    }
    pub fn addmm(&self, beta: &NumKind, alpha: &NumKind, tensor1: &Self, tensor2: &Self) -> Self {
        unimplemented!()
    }
    pub fn addmm_(self, beta: &NumKind, alpha: &NumKind, tensor1: &Self, tensor2: &Self) -> Self {
        unimplemented!()
    }
    pub fn addmv(&self, beta: &NumKind, alpha: &NumKind, tensor1: &Self, vec: &Self) -> Self {
        unimplemented!()
    }
    pub fn addmv_(self, beta: &NumKind, alpha: &NumKind, tensor1: &Self, vec: &Self) -> Self {
        unimplemented!()
    }
    pub fn addr(&self, beta: &NumKind, alpha: &NumKind, vec1: &Self, vec2: &Self) -> Self {
        unimplemented!()
    }
    pub fn addr_(self, beta: &NumKind, alpha: &NumKind, vec1: &Self, vec2: &Self) -> Self {
        unimplemented!()
    }
    pub fn asin(&self) -> Self {
        unimplemented!()
    }
    pub fn asin_(self) -> Self {
        unimplemented!()
    }
    pub fn atan(&self) -> Self {
        unimplemented!()
    }
    pub fn atan2(&self) -> Self {
        unimplemented!()
    }
    pub fn atan2_(self) -> Self {
        unimplemented!()
    }
    pub fn baddbmm(&self, beta: &NumKind, alpha: &NumKind, tensor1: &Self, tensor2: &Self) -> Self {
        unimplemented!()
    }
    pub fn baddbmm_(self, beta: &NumKind, alpha: &NumKind, tensor1: &Self, tensor2: &Self) -> Self {
        unimplemented!()
    }
    pub fn bernoulli(&self) -> Self {
        unimplemented!()
    }
    pub fn bernoulli_(self) -> Self {
        unimplemented!()
    }
    pub fn bmm(&self, other: &Self) -> Self {
        unimplemented!()
    }
    pub fn byte(self) -> Self {
        unimplemented!()
    }
    pub fn ceil(&self) -> Self {
        unimplemented!()
    }
    pub fn ceil_(self) -> Self {
        unimplemented!()
    }
    pub fn char(self) -> Self {
        unimplemented!()
    }
    pub fn chunk(&self, n_chunks: usize, dim: usize) -> Vec<Self> {
        unimplemented!()
    }
    pub fn clamp(&self, min: f32, max: f32) -> Self {
        unimplemented!()
    }
    pub fn clamp_(self, min: f32, max: f32) -> Self {
        unimplemented!()
    }
    pub fn contiguous(&self) -> Self {
        unimplemented!()
    }
    // perform deep copy
    pub fn copy(&self) -> Self {
        unimplemented!()
    }
    pub fn copy_(self, src: &Self) -> Self {
        unimplemented!()
    }
    pub fn copy_async_(self, src: &Self) -> Self {
        unimplemented!()
    }
    pub fn cos(&self) -> Self {
        unimplemented!()
    }
    pub fn cos_(self) -> Self {
        unimplemented!()
    }
    pub fn cosh(&self) -> Self {
        unimplemented!()
    }
    pub fn cosh_(self) -> Self {
        unimplemented!()
    }
    pub fn cpu(&self) -> Self {
        unimplemented!()
    }
    pub fn cross(&self, dim: Option<i32>) -> Self {
        unimplemented!()
    }
    pub fn cuda(&self, device: Option<i32>) -> Self {
        unimplemented!()
    }
    pub fn cuda_async(&self, device: Option<i32>) -> Self {
        unimplemented!()
    }
    pub fn diag(&self, diag: u32) -> Self {
        unimplemented!()
    }
    pub fn dist(&self, other: &Self, p: u32) -> f32 {
        unimplemented!()
    }
    pub fn div(&self, value: &Self) -> Self {

        unimplemented!()
    }
    pub fn div_(self, value: &Self) -> Self {
        unimplemented!()
    }
    pub fn dot(&self, other: &Self) -> Self {
        unimplemented!()
    }
    pub fn double(&self) -> Self {
        unimplemented!()
    }
    pub fn eig(&self, eigenvectors: bool) -> (Self, Self) {
        unimplemented!()
    }
    pub fn element_size(&self) -> i32 {
        unimplemented!()
    }
    pub fn eq_tensor(&self, other: &Self) -> Self {
        unimplemented!()
    }
    pub fn eq_tensor_(&self, other: &Self) -> Self {
        unimplemented!()
    }
    pub fn exp(&self) -> Self {
        unimplemented!()
    }
    pub fn exp_(self) -> Self {
        unimplemented!()
    }
    pub fn expand(&self, dims: &[u32]) -> Self {
        unimplemented!()
    }
    pub fn expand_as(&self, tensor: &Self) -> Self {
        unimplemented!()
    }
    pub fn fill_(self) -> Self {
        unimplemented!()
    }
    pub fn float(self) -> Self {
        unimplemented!()
    }
    pub fn floor(&self) -> Self {
        unimplemented!()
    }
    pub fn floor_(self) -> Self {
        unimplemented!()
    }
    pub fn fmod(&self, divisor: &NumKind) -> Self {
        unimplemented!()
    }
    pub fn fmod_(self, divisor: &NumKind) -> Self {
        unimplemented!()
    }
    pub fn frac(&self) -> Self {
        unimplemented!()
    }
    pub fn frac_(self) -> Self {
        unimplemented!()
    }
    pub fn gather(&self, dim: i32, index: Tensor<i64>) {
        unimplemented!()
    }
    pub fn ge_tensor(&self, other: &Self) -> Self {
        unimplemented!()
    }
    pub fn ge_tensor_(&self, other: &Self) -> Self {
        unimplemented!()
    }
    pub fn gels(&self, other: &Self) -> Self {
        unimplemented!();
    }
    pub fn gt_tensor(&self, other: &Self) -> Self {
        unimplemented!()
    }
    pub fn gt_tensor_(&self, other: &Self) -> Self {
        unimplemented!()
    }
    pub fn half(self) -> Self {
        unimplemented!()
    }
    pub fn index_masked(&self, m: &Tensor<u8>) -> Self {
        unimplemented!()
    }

    pub fn long(self) -> Self {
        unimplemented!()
    }

    pub fn new(&self) -> Self {
        unimplemented!()
    }
    pub fn unsqueeze(self, dim: usize) -> Self {
        unimplemented!()
    }
    pub fn in_thft(&self) -> *mut THFloatTensor {
        unimplemented!();
    }
    pub fn in_thlt(&self) -> *mut THLongTensor {
        unimplemented!();
    }
    //    pub fn backend(&self) -> &
}

impl Index<usize> for TensorKind {
    type Output = NumKind;
    fn index(&self, idx: usize) -> &Self::Output {
        unimplemented!()
    }
}

impl PartialEq for TensorKind {
    fn eq(&self, other: &TensorKind) -> bool {
        use self::TensorKind::{FloatTensor, LongTensor};
        match (self, other) {
            (&FloatTensor(ref t1), &FloatTensor(ref t2)) => t1.id == t2.id,
            (&LongTensor(ref t1), &LongTensor(ref t2)) => t1.id == t2.id,
            _ => false,
        }
    }
}
impl Eq for TensorKind {}
impl Clone for TensorKind {
    fn clone(&self) -> Self {
        use self::TensorKind::{FloatTensor, LongTensor};
        match *self {
            FloatTensor(ref t) => FloatTensor(t.clone()),
            LongTensor(ref t) => LongTensor(t.clone()),
        }
    }
}

impl<T> From<Tensor<T>> for TensorKind {
    #[allow(unused_variables)]
    default fn from(input: Tensor<T>) -> Self {
        unreachable!()
    }
}

impl From<Tensor<f32>> for TensorKind {
    fn from(input: Tensor<f32>) -> TensorKind {
        TensorKind::FloatTensor(input)
    }
}

impl From<Tensor<i64>> for TensorKind {
    fn from(input: Tensor<i64>) -> TensorKind {
        TensorKind::LongTensor(input)
    }
}

impl<T> From<TensorKind> for Tensor<T> {
    #[allow(unused_variables)]
    default fn from(input: TensorKind) -> Tensor<T> {
        panic!("bad cast")
    }
}

impl<'a, T> From<&'a TensorKind> for &'a Tensor<T> {
    #[allow(unused_variables)]
    default fn from(input: &'a TensorKind) -> &'a Tensor<T> {
        panic!("bad cast")
    }
}

impl<'a> From<&'a TensorKind> for &'a Tensor<f32> {
    fn from(input: &'a TensorKind) -> Self {
        match *input {
            TensorKind::FloatTensor(ref t) => t,
            _ => unreachable!(),
        }
    }
}

impl<'a> From<&'a TensorKind> for &'a Tensor<i64> {
    fn from(input: &'a TensorKind) -> Self {
        match *input {
            TensorKind::LongTensor(ref t) => t,
            _ => unreachable!(),
        }
    }
}

impl From<TensorKind> for Tensor<f32> {
    fn from(input: TensorKind) -> Self {
        match input {
            TensorKind::FloatTensor(v) => v,
            _ => unimplemented!(),
        }
    }
}

impl From<TensorKind> for Tensor<i64> {
    fn from(input: TensorKind) -> Self {
        match input {
            TensorKind::LongTensor(v) => v,
            _ => unimplemented!(),
        }
    }
}
pub struct Tensor<T> {
    pub id: i32,
    value: RcMut<TensorImpl<T, Output = T>>,
}
impl<T> Hash for Tensor<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state)
    }
}

impl<T: Copy> Index<usize> for Tensor<T> {
    type Output = T;
    fn index(&self, idx: usize) -> &Self::Output {

        //        self.value.borrow_mut().index(idx as isize)
        unimplemented!()
    }
}

impl<T: Copy> Index<i32> for Tensor<T> {
    type Output = T;
    fn index(&self, idx: i32) -> &Self::Output {

        //        self.value.borrow_mut().index(idx as isize)
        unimplemented!()
    }
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
impl<T> New<Vec<usize>, Tensor<T>> for Tensor<T> {
    fn new(args: Vec<usize>) -> Self {
        unimplemented!()
    }
    fn new_(&self, args: Vec<usize>) -> Self {
        unimplemented!()
    }
}

impl<T> Tensor<T> {
    pub fn len(&self) -> usize {
        unimplemented!()
    }
    pub fn size(&self) -> Vec<usize> {
        unimplemented!()
    }
    pub fn zero_(&mut self) -> Self {
        unimplemented!()
    }
    pub fn add_(&mut self, rhs: &T) {
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
