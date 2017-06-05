// while a WIP
#![allow(unused_variables)]
use rutorch::*;
use std::ops::{Index, IndexMut};
use std::convert::From;
use std::cmp::max;
use std::hash::{Hash, Hasher};

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

impl NumKind {
    pub fn intof32(&self) -> f32 {
        use self::NumKind::{Float, Double, Long};
        match *self {
            Float(v) => v,
            Double(v) => unimplemented!(),
            Long(v) => unimplemented!(),
        }
    }
    pub fn intoi64(&self) -> i64 {
        use self::NumKind::{Float, Double, Long};
        match *self {
            Float(v) => unimplemented!(),
            Double(v) => unimplemented!(),
            Long(v) => v,
        }
    }
}


impl<T: Copy> From<T> for NumKind {
    #[allow(unused_variables)]
    default fn from(input: T) -> Self {
        unreachable!()
    }
}
impl From<f32> for NumKind {
    fn from(input: f32) -> Self {
        NumKind::Float(input)
    }
}
impl From<f64> for NumKind {
    fn from(input: f64) -> Self {
        NumKind::Double(input)
    }
}
impl From<i64> for NumKind {
    fn from(input: i64) -> Self {
        NumKind::Long(input)
    }
}


pub type TensorList<T> = Vec<Tensor<T>>;
pub type TensorKindList = Vec<TensorKind>;
pub type OptTensorKindList = Vec<Option<TensorKind>>;
pub type RefTensorList<'a, T> = Vec<&'a mut Tensor<T>>;
pub type RefTensorKindList<'a> = Vec<&'a TensorKind>;
pub type TensorId = usize;


impl TensorKind {
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

impl<T> From<Tensor<T>> for TensorKind {
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

impl<T> From<TensorKind> for Tensor<T> {
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
impl<T: Copy> Hash for Tensor<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id().hash(state)
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


impl New<usize, TensorKind> for TensorKind {
    fn new(args: usize) -> Self {
        unimplemented!()
    }
    fn new_(&self, args: usize) -> Self {
        unimplemented!()
    }
}
impl New<(), TensorKind> for TensorKind {
    fn new(args: ()) -> Self {
        unimplemented!()
    }
    fn new_(&self, args: ()) -> Self {
        unimplemented!()
    }
}
impl New<Vec<usize>, TensorKind> for TensorKind {
    fn new(args: Vec<usize>) -> Self {
        unimplemented!()
    }
    fn new_(&self, args: Vec<usize>) -> Self {
        unimplemented!()
    }
}
impl New<[usize; 2], TensorKind> for TensorKind {
    fn new(args: [usize; 2]) -> Self {
        unimplemented!()
    }
    fn new_(&self, args: [usize; 2]) -> Self {
        unimplemented!()
    }
}


impl<T> Tensor<T> {
    pub fn len(&self) -> usize {
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
        Tensor { value: self.value.clone() }
    }
}

impl<T: Copy> Index<isize> for Tensor<T> {
    type Output = T;

    fn index(&self, idx: isize) -> &Self::Output {
        unimplemented!()
    }
}

type RefTI<T> = RcMut<TensorImpl<T, Output = T>>;
pub type TIArg<T> = TensorImpl<T, Output = T>;
pub trait TensorImpl<T>: Index<Ixs, Output = T> {
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
