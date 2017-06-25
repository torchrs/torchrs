// while a WIP
#![allow(unused_variables)]
use rutorch::*;
use std::ops::{Index, IndexMut};
use std::convert::From;
use std::cmp::max;
use std::hash::{Hash, Hasher};
use serde::{Serialize, Deserialize, Serializer, Deserializer};

use storage::*;
use ::*;
pub use tensor::tensor_ops::*;
use rand;
use {Ixs, RcMut};

pub enum TensorType {
    Float,
    Double,
    Byte,
    Char,
    Short,
    Int,
    Long,
}

pub trait NumLimits<T>: Copy + Default + ::num::Num {}
impl NumLimits<f32> for f32 {}
impl NumLimits<f64> for f64 {}
impl NumLimits<i32> for i32 {}
impl NumLimits<i64> for i64 {}
impl NumLimits<u8> for u8 {}

#[derive(Hash, Serialize, Deserialize)]
pub enum TensorKind {
    FloatTensor(Tensor<f32>),
    LongTensor(Tensor<i64>),
}

pub type TensorList<T> = Vec<Tensor<T>>;
pub type TensorKindList = Vec<TensorKind>;
pub type OptTensorKindList = Vec<Option<TensorKind>>;
pub type RefTensorList<'a, T> = Vec<&'a mut Tensor<T>>;
pub type RefTensorKindList<'a> = Vec<&'a TensorKind>;
pub type TensorId = usize;


impl TensorKind {
    pub fn backend(&self) -> Box<nn::BackendIntf> {
        unimplemented!()
    }
    pub fn inner(&self) -> *mut ::std::os::raw::c_void {
        unimplemented!()
    }
    pub fn len(&self) -> usize {
        unimplemented!()
    }
    pub fn s(&self, dim: usize) -> Self {
        unimplemented!()
    }
}

/*
impl<T: NumLimits<T>> Index<usize> for TensorKind {
    type Output = T;
    fn index(&self, idx: usize) -> &Self::Output {
        unimplemented!()
    }
}
*/
impl PartialEq for TensorKind {
    fn eq(&self, other: &Self) -> bool {
        use self::TensorKind::{FloatTensor, LongTensor};
        match (self, other) {
            (&FloatTensor(ref t1), &FloatTensor(ref t2)) => t1.id() == t2.id(),
            (&LongTensor(ref t1), &LongTensor(ref t2)) => t1.id() == t2.id(),
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

impl<T: NumLimits<T>> From<Tensor<T>> for TensorKind {
    #[allow(unused_variables)]
    default fn from(input: Tensor<T>) -> Self {
        unreachable!()
    }
}

impl From<Tensor<f32>> for TensorKind {
    fn from(input: Tensor<f32>) -> Self {
        TensorKind::FloatTensor(input)
    }
}

impl From<Tensor<i64>> for TensorKind {
    fn from(input: Tensor<i64>) -> Self {
        TensorKind::LongTensor(input)
    }
}

impl<T: NumLimits<T>> From<TensorKind> for Tensor<T> {
    #[allow(unused_variables)]
    default fn from(input: TensorKind) -> Self {
        panic!("bad cast")
    }
}

impl<'a, T> From<&'a TensorKind> for &'a Tensor<T> {
    #[allow(unused_variables)]
    default fn from(input: &'a TensorKind) -> Self {
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

impl<'a, T> From<&'a mut TensorKind> for &'a mut Tensor<T> {
    #[allow(unused_variables)]
    default fn from(input: &'a mut TensorKind) -> Self {
        panic!("bad cast")
    }
}

impl<'a> From<&'a mut TensorKind> for &'a mut Tensor<f32> {
    fn from(input: &'a mut TensorKind) -> Self {
        match *input {
            TensorKind::FloatTensor(ref mut t) => t,
            _ => unreachable!(),
        }
    }
}

impl<'a> From<&'a mut TensorKind> for &'a mut Tensor<i64> {
    fn from(input: &'a mut TensorKind) -> Self {
        match *input {
            TensorKind::LongTensor(ref mut t) => t,
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
    pub value: RcMut<TensorImpl<T, Output = T>>,
}

impl<T: NumLimits<T>> Serialize for Tensor<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        unimplemented!()
    }
}
impl<'de, T> Deserialize<'de> for Tensor<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where D: Deserializer<'de>
    {
        unimplemented!()
    }
}

impl<T: NumLimits<T>> Hash for Tensor<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id().hash(state)
    }
}

impl<T: NumLimits<T>> Index<usize> for Tensor<T> {
    type Output = T;
    fn index(&self, idx: usize) -> &Self::Output {

        //        self.value.borrow_mut().index(idx as isize)
        unimplemented!()
    }
}

impl<T: NumLimits<T>> Index<i32> for Tensor<T> {
    type Output = T;
    fn index(&self, idx: i32) -> &Self::Output {

        //        self.value.borrow_mut().index(idx as isize)
        unimplemented!()
    }
}


pub trait New<D, T> {
    fn new(args: D) -> T;
}
pub trait NewSelf<D, T> {
    fn new(&self, args: D) -> T;
}

impl<T: NumLimits<T>> NewSelf<usize, Tensor<T>> for Tensor<T> {
    fn new(&self, args: usize) -> Self {
        unimplemented!()
    }
}
impl<T: NumLimits<T>> NewSelf<(), Tensor<T>> for Tensor<T> {
    fn new(&self, args: ()) -> Self {
        unimplemented!()
    }
}
impl<T: NumLimits<T>> NewSelf<Vec<usize>, Tensor<T>> for Tensor<T> {
    fn new(&self, args: Vec<usize>) -> Self {
        unimplemented!()
    }
}
impl NewSelf<usize, TensorKind> for TensorKind {
    fn new(&self, args: usize) -> Self {
        unimplemented!()
    }
}
impl NewSelf<(), TensorKind> for TensorKind {
    fn new(&self, args: ()) -> Self {
        unimplemented!()
    }
}
impl NewSelf<Vec<usize>, TensorKind> for TensorKind {
    fn new(&self, args: Vec<usize>) -> Self {
        unimplemented!()
    }
}
impl NewSelf<[usize; 2], TensorKind> for TensorKind {
    fn new(&self, args: [usize; 2]) -> Self {
        unimplemented!()
    }
}


impl<T: NumLimits<T>> Tensor<T> {
    pub fn len(&self) -> usize {
        unimplemented!()
    }
    pub fn s(&self, dim: usize) -> Self {
        unimplemented!()
    }
    pub fn cast<D>(&self) -> Tensor<D> {
        unimplemented!()
    }
}

impl<T: NumLimits<T>> Default for Tensor<T> {
    fn default() -> Self {
        unimplemented!()
    }
}

impl<T: NumLimits<T>> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        Tensor { value: self.value.clone() }
    }
}

impl<T: NumLimits<T>> Index<isize> for Tensor<T> {
    type Output = T;

    fn index(&self, idx: isize) -> &Self::Output {
        unimplemented!()
    }
}

type RefTI<T: NumLimits<T>> = RcMut<TensorImpl<T, Output = T>>;
pub type TIArg<T: NumLimits<T>> = TensorImpl<T, Output = T>;
pub trait TensorImpl<T: NumLimits<T>>: Index<Ixs, Output = T> {
    //fn view<'a>(&self, dims: &[i32]) -> Tensor<'a>;
    fn new(&self) -> RefTI<T>;
    fn add(&self, value: T, output: &TIArg<T>);
    fn addt(&self, value: T, rhs: &TIArg<T>, output: &TIArg<T>);
    fn inner(&self) -> *mut ::std::os::raw::c_void;
}

impl TensorImpl<f32> for FloatTensor {
    fn new(&self) -> RefTI<f32> {
        // XXX place holder implementation
        RcMutNew(FloatTensor::new())
    }
    fn add(&self, value: f32, output: &TIArg<f32>) {
        let out = typecast!(output.inner(), THFloatTensor);
        unsafe {
            THFloatTensor_add(out, self.t, value);
        };
    }
    fn inner(&self) -> *mut ::std::os::raw::c_void {
        self.t as *mut ::std::os::raw::c_void
    }
    fn addt(&self, value: f32, rhs: &TIArg<f32>, output: &TIArg<f32>) {
        let rhsp = typecast!(rhs.inner(), THFloatTensor);
    }
}

impl Index<isize> for FloatTensor {
    type Output = f32;
    fn index(&self, idx: isize) -> &Self::Output {
        unimplemented!()
    }
}

impl Serialize for FloatTensor {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        unimplemented!()
    }
}
impl<'de> Deserialize<'de> for FloatTensor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where D: Deserializer<'de>
    {
        unimplemented!()
    }
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

pub fn make_vec(val: usize, count: usize) -> Vec<isize> {
    let mut vec = Vec::new();
    for _ in 0..count {
        vec.push(val as isize)
    }
    vec
}
