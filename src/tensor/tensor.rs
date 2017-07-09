// while a WIP
#![allow(unused_variables)]
use rutorch::*;
use std::ops::{Index, IndexMut};
use std::convert::From;
use std::cmp::max;
use std::fmt;
use std::hash::{Hash, Hasher};
use serde::{Serialize, Deserialize, Serializer, Deserializer};

use storage::*;
use ::*;
pub use tensor::tensor_ops::*;
use RcMut;

#[derive(Clone, Debug)]
pub struct THVec<T: NumLimits> {
    pub data: Vec<T>,
    pub dims: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct THVecGeneric {
    pub data: Vec<i64>,
    pub dims: Vec<usize>,
}
#[derive(Clone, Debug)]
pub struct THDims {
    pub dims: Vec<usize>,
}

pub enum TensorType {
    Float,
    Double,
    Byte,
    Char,
    Short,
    Int,
    Long,
}

pub trait NumLimits
    : Copy + Default + fmt::Debug + ::num::Num + ::num::NumCast + serde::Serialize {
}
impl NumLimits for f32 {}
impl NumLimits for f64 {}
impl NumLimits for i32 {}
impl NumLimits for u32 {}
impl NumLimits for i64 {}
impl NumLimits for u8 {}

#[derive(Hash, Serialize, Deserialize, Debug)]
pub enum TensorKind {
    FloatTensor(Tensor<f32>),
    LongTensor(Tensor<i64>),
    ByteTensor(Tensor<u8>),
}

pub type TensorList<T> = Vec<Tensor<T>>;
pub type TensorKindList = Vec<TensorKind>;
pub type OptTensorKindList = Vec<Option<TensorKind>>;
pub type RefTensorList<'a, T> = Vec<&'a mut Tensor<T>>;
pub type RefTensorKindList<'a> = Vec<&'a TensorKind>;
pub type TensorId = usize;
type voidp = ::std::os::raw::c_void;


impl TensorKind {
    pub fn new<S>(&self, args: S) -> Self
        where S: Into<THVecGeneric>
    {
        match *self {
            TensorKind::FloatTensor(ref t) => {
                let tv: THVecGeneric = args.into();
                let tv: THVec<f32> = tv.into();
                let mut newt: Tensor<f32> = t.new(tv.dims);
                if tv.data.len() != 0 {
                    newt.set_storage(tv.data);
                }
                newt.into()
            }
            TensorKind::LongTensor(ref t) => {
                let tv: THVecGeneric = args.into();
                let tv: THVec<i64> = tv.into();
                let mut newt: Tensor<i64> = t.new(tv.dims);
                if tv.data.len() != 0 {
                    newt.set_storage(tv.data);
                }
                newt.into()
            }
            TensorKind::ByteTensor(ref t) => {
                let tv: THVecGeneric = args.into();
                let tv: THVec<u8> = tv.into();
                let mut newt: Tensor<u8> = t.new(tv.dims);
                if tv.data.len() != 0 {
                    newt.set_storage(tv.data);
                }
                newt.into()
            }

        }
    }

    pub fn backend(&self) -> Box<nn::BackendIntf> {
        match *self {
            /* XXX avoid repeated boxing on every call */
            TensorKind::FloatTensor(_) => Box::new(nn::FloatBackend.clone()),
            _ => panic!("no corresponding backend"),
        }
    }
    pub fn len(&self) -> usize {
        unimplemented!()
    }
    pub fn s(&self, dim: usize) -> Self {
        unimplemented!()
    }
}

impl PartialEq for TensorKind {
    fn eq(&self, other: &Self) -> bool {
        use self::TensorKind::{FloatTensor, LongTensor, ByteTensor};
        match (self, other) {
            (&FloatTensor(ref t1), &FloatTensor(ref t2)) => t1.id() == t2.id(),
            (&LongTensor(ref t1), &LongTensor(ref t2)) => t1.id() == t2.id(),
            (&ByteTensor(ref t1), &ByteTensor(ref t2)) => t1.id() == t2.id(),
            _ => false,
        }
    }
}
impl Eq for TensorKind {}
impl Clone for TensorKind {
    fn clone(&self) -> Self {
        use self::TensorKind::{FloatTensor, LongTensor, ByteTensor};
        match *self {
            FloatTensor(ref t) => FloatTensor(t.clone()),
            LongTensor(ref t) => LongTensor(t.clone()),
            ByteTensor(ref t) => ByteTensor(t.clone()),
        }
    }
}

impl<T: NumLimits> From<Tensor<T>> for TensorKind {
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
impl From<Tensor<u8>> for TensorKind {
    fn from(input: Tensor<u8>) -> Self {
        TensorKind::ByteTensor(input)
    }
}

impl From<Tensor<i64>> for TensorKind {
    fn from(input: Tensor<i64>) -> Self {
        TensorKind::LongTensor(input)
    }
}

impl<T: NumLimits> From<TensorKind> for Tensor<T> {
    #[allow(unused_variables)]
    default fn from(input: TensorKind) -> Self {
        panic!("bad cast")
    }
}

impl<'a, T: NumLimits> From<&'a TensorKind> for &'a Tensor<T> {
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

impl<'a, T: NumLimits> From<&'a mut TensorKind> for &'a mut Tensor<T> {
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
/*
impl From<TensorKind> for Tensor<f64> {
    fn from(input: TensorKind) -> Self {
        match input {
            TensorKind::DoubleTensor(v) => v,
            _ => unimplemented!(),
        }
    }
}
*/
impl From<TensorKind> for Tensor<i64> {
    fn from(input: TensorKind) -> Self {
        match input {
            TensorKind::LongTensor(v) => v,
            _ => unimplemented!(),
        }
    }
}
impl From<TensorKind> for Tensor<u8> {
    fn from(input: TensorKind) -> Self {
        match input {
            TensorKind::ByteTensor(v) => v,
            _ => unimplemented!(),
        }
    }
}

pub struct Tensor<T: NumLimits> {
    pub value: RcMut<TensorImpl<T, Output = T>>,
}

impl<T: NumLimits> fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Tensor ")
    }
}


impl<T: NumLimits> Serialize for Tensor<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        let rt = self.value.borrow().to_rust_tensor();
        let result = rt.serialize(serializer)?;
        Ok(result)
    }
}
impl<'de, T: NumLimits + Deserialize<'de>> Deserialize<'de> for Tensor<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where D: Deserializer<'de>
    {
        let rt: RustTensor<T> = RustTensor::deserialize(deserializer)?;
        let mut t = ::torch::tensor(());
        t.from_rust_tensor(rt);
        Ok(t)
    }
}

impl<T: NumLimits> Hash for Tensor<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id().hash(state)
    }
}

impl<T: NumLimits> Index<usize> for Tensor<T> {
    type Output = T;
    fn index(&self, idx: usize) -> &Self::Output {
        let t = unsafe { &mut *self.value.as_ptr() };
        t.index(idx)
    }
}

impl<T: NumLimits> Index<i32> for Tensor<T> {
    type Output = T;
    fn index(&self, idx: i32) -> &Self::Output {
        let t = unsafe { &mut *self.value.as_ptr() };
        t.index(idx as usize)
    }
}

impl<T: NumLimits> Tensor<T> {
    pub fn cast<D>(&self) -> Tensor<D>
        where D: NumLimits
    {
        let t: Tensor<D> = torch::tensor(self.size());
        let s: Vec<D> = self.value
            .borrow()
            .iter()
            .map(|v| <D as num::NumCast>::from(v).unwrap())
            .collect();
        t.value.borrow_mut().set_storage(s.as_slice());
        t
    }
    pub fn from_rust_tensor(&mut self, rt: RustTensor<T>) {
        self.value.borrow_mut().from_rust_tensor(rt);
    }
    pub fn get_storage(&self, data: &mut [T], len: usize) {
        assert!(len >= self.size().iter().product());
        let storage = self.value.borrow().get_storage(data);
    }
    pub fn len(&self) -> usize {
        self.value.borrow().len()
    }
    pub fn new<S>(&self, args: S) -> Self
        where S: Into<THVec<T>>
    {
        let mut args: THVec<T> = args.into();
        if args.dims.len() == 0 {
            self.value.borrow().new()
        } else {
            ::torch::tensor(args)
        }
    }
    /* XXX handle non-contiguous case */
    pub fn s<D>(&self, dim: D) -> Self
        where D: AsRef<[usize]>
    {
        self.value.borrow().s(dim.as_ref())
    }
    pub fn set_storage(&mut self, args: Vec<T>) {
        self.value.borrow_mut().set_storage(args.as_slice());
    }
}

impl<T: NumLimits> Default for Tensor<T> {
    fn default() -> Self {
        unimplemented!()
    }
}

impl<T: NumLimits> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        Tensor { value: self.value.clone() }
    }
}

type RefTI<T> = RcMut<TensorImpl<T, Output = T>>;
pub type TIArg<T> = TensorImpl<T, Output = T>;
pub trait TensorImpl<T: NumLimits>: Index<Ix, Output = T> {
    fn new(&self) -> Tensor<T>;
    fn add(&mut self, src: *mut voidp, value: T);
    fn addmm(&mut self, src: *mut voidp, beta: T, alpha: T, mat1: *mut voidp, mat2: *mut voidp);
    fn addt(&mut self, src: *mut voidp, value: *mut voidp);
    fn bernoulli(&mut self, p: f64);
    fn copy(&mut self, src: &RefTI<T>);
    fn dim(&self) -> i32;
    fn div(&mut self, src: *mut voidp, value: T);
    fn divt(&mut self, src: *mut voidp, value: *mut voidp);
    fn expand(&self, dims: &[usize]) -> Tensor<T>;
    fn eq_tensor(&self, other: *mut voidp, out: *mut voidp);
    fn from_rust_tensor(&mut self, rt: RustTensor<T>);
    fn ge_tensor(&self, other: *mut voidp, out: *mut voidp);
    fn get_storage(&self, data: &mut [T]);
    fn gt_tensor(&self, other: *mut voidp, out: *mut voidp);
    fn inner(&self) -> *mut voidp;
    fn iter(&self) -> Box<Iterator<Item = T>>;
    fn le_tensor(&self, other: *mut voidp, out: *mut voidp);
    fn le_value(&self, value: T, out: *mut voidp);
    fn len(&self) -> usize;
    fn lt_tensor(&self, other: *mut voidp, out: *mut voidp);
    fn lt_value(&self, value: T, out: *mut voidp);
    fn max_reduce(&self, values: *mut voidp, indices: *mut voidp, dim: usize, keepdim: bool);
    fn min_reduce(&self, values: *mut voidp, indices: *mut voidp, dim: usize, keepdim: bool);
    fn mm(&mut self, mat1: *mut voidp, mat2: *mut voidp);
    fn mul(&mut self, src: *mut voidp, value: T);
    fn mult(&mut self, src: *mut voidp, value: *mut voidp);
    fn ne_tensor(&self, other: *mut voidp, out: *mut voidp);
    fn prod(&self, result: &mut f64);
    fn resize(&mut self, dims: &[usize]);
    fn s(&self, dim: &[usize]) -> Tensor<T>;
    fn size(&self) -> Vec<usize>;
    fn set_storage(&mut self, v: &[T]);
    fn sum_reduce(&mut self, input: *mut voidp, dim: usize, keepdim: bool);
    fn sum(&self, result: &mut f64);
    fn squeeze(&mut self, dim: Option<usize>);
    fn unsqueeze(&mut self, dim: usize);
    fn transpose(&mut self, src: *mut voidp, dim0: usize, dim1: usize);
    fn to_rust_tensor(&self) -> RustTensor<T>;
    fn uniform_(&mut self, range: (f64, f64));
    fn view(&self, dims: &[isize]) -> Tensor<T>;
    fn zero(&mut self);
}

pub struct Generator {
    t: *mut THGenerator,
}
impl Generator {
    pub fn new() -> Self {
        let t = unsafe { THGenerator_new() };
        Generator { t: t }
    }
}
#[derive(Serialize, Deserialize)]
pub struct RustTensor<T> {
    size: Vec<i64>,
    stride: Vec<i64>,
    storage: Vec<T>,
}

macro_rules!  unsafe_index {
    ($p:ident, $idx:ident) => {
         unsafe { & *(*(* $p .t).storage).data.offset($idx as isize) }
    }
}
macro_rules!  unsafe_index_mut {
    ($p:ident, $idx:ident) => {
         unsafe { &mut *(*(* $p .t).storage).data.offset($idx as isize) }
    }
}

macro_rules! impl_tensor_impl {
    ($name:ident, $type:ident, $thname:ident, $storage_name:ident) => {
        pub struct $name {
            t: *mut $thname,
        }
        impl $name {
            pub fn new() -> Self {
                unsafe {
                    $name {
                        t: concat_idents!($thname, _new)(),
                    }
                }
            }
            fn from_parts(t: *mut $thname) -> Self {
                $name { t: t}
            }
            fn to_rust_tensor(&self) -> RustTensor<$type> {
                let mut size: Vec<i64> = Vec::new();
                let mut stride: Vec<i64> = Vec::new();
                let mut storage = Vec::new();
                let (offset, nd) = (self.storage_offset(), self.dim());
                let need_stride = unsafe {(*self.t).stride != std::ptr::null_mut()};

                for i in 0..nd {
                    let s = unsafe { &*(*self.t).size.offset(i as isize) };
                    size.push(*s);
                }
                if need_stride {
                    for i in 0..nd {
                        let s = unsafe { &*(*self.t).stride.offset(i as isize) };
                        stride.push(*s);
                    }
                }
                let s = self.storage();
                /* XXX this has to be the slowest way possible */
                for i in s.iter() {
                    storage.push(*i);
                }
                RustTensor {size: size, stride: stride, storage: storage}
            }
            pub fn with_capacity<D>(dims: D) -> Self
                where D: AsRef<[usize]>
            {
                let dims_long : Vec<i64> = dims.as_ref().iter().map(|t| *t as i64).collect();
                let dims = dims.as_ref();
                let sizes = LongStorage::with_data(dims_long.as_slice());
                let size = dims.iter().product();
                let storage = $storage_name ::with_capacity(size);
                let t = unsafe {
                    concat_idents!($thname, _newWithStorage)(storage.t,
                                                             0,
                                                             sizes.t,
                                                             std::ptr::null_mut())
                };
                $name { t: t}
            }
            pub fn randn<D>(dims: D) -> Self
                where D: AsRef<[usize]>
            {
                let dims = dims.as_ref();
                unimplemented!()
            }
            fn len(&self) -> usize {
                self.size().iter().product()
            }
            fn storage(&self) -> $storage_name {
                let s = unsafe { (*self.t).storage };
                $storage_name ::from_raw_parts(s)
            }
            fn storage_offset(&self) -> usize {
                (unsafe {(*self.t).storageOffset}) as usize
            }
        }

        impl TensorImpl<$type> for $name {
            fn add(&mut self, src: *mut voidp, value: $type) {
                let srcp = src as *mut $thname;
                unsafe {concat_idents!($thname, _add)(self.t, srcp, value)};
            }
            fn addmm(&mut self,
                    src: *mut voidp,
                    beta: $type,
                    alpha: $type,
                    mat1: *mut voidp,
                    mat2: *mut voidp) {
                let srcp = src as *mut $thname;
                let (mat1p, mat2p) = (mat1 as *mut $thname, mat2 as *mut $thname);
                unsafe {
                    concat_idents!($thname, _addmm)(self.t,
                                                    beta,
                                                    srcp,
                                                    alpha,
                                                    mat1p,
                                                    mat2p)
                };
            }
            fn addt(&mut self,
                    src: *mut voidp,
                    value: *mut voidp) {
                let srcp = src as *mut $thname;
                let valuep = value as *mut $thname;
                unsafe { concat_idents!($thname, _cadd)(self.t, valuep, 1 as $type, srcp)};
            }
            fn bernoulli(&mut self, p: f64) {
                let g = Generator::new();
                unsafe { concat_idents!($thname, _bernoulli)(self.t, g.t, p) };
            }
            fn copy(&mut self, src: &RefTI<$type>) {
                let t = src.borrow_mut().inner() as *mut $thname;
                unsafe {concat_idents!($thname, _copy)(self.t, t)}
            }
            fn dim(&self) -> i32 {
                unsafe {(*self.t).nDimension}
            }
            fn div(&mut self, src: *mut voidp, value: $type) {
                let srcp = src as *mut $thname;
                unsafe {concat_idents!($thname, _div)(self.t, srcp, value)};
            }
            fn divt(&mut self,
                    src: *mut voidp,
                    value: *mut voidp) {
                unimplemented!()
            }
            fn eq_tensor(&self, other: *mut voidp, out: *mut voidp) {
                let outp = out as *mut THByteTensor;
                let otherp = other as *mut $thname;
                unsafe {concat_idents!($thname, _eqTensor)(outp, self.t, otherp) }
            }
            fn expand(&self, dims: &[usize]) -> Tensor<$type> {
                let dims_long : Vec<i64> = dims.iter().map(|t| *t as i64).collect();
                let size = LongStorage::with_data(dims_long.as_slice());
                let newt = unsafe {
                    concat_idents!($thname, _newExpand)(self.t, size.t)
                };
                let t = $name :: from_parts(newt);
                Tensor { value: RcMutNew(t) }
            }
            fn from_rust_tensor(&mut self, rt: RustTensor<$type>) {
                let size : Vec<usize> = rt.size.iter().map(|t| *t as usize).collect();
                self.resize(size.as_slice());
                let s = rt.storage.as_ptr() as *const voidp;
                let d = (unsafe { ((*(*self.t).storage).data) }) as *mut voidp;
                unsafe {memcpy(d, s, rt.storage.len()) };
            }
            fn ge_tensor(&self, other: *mut voidp, out: *mut voidp) {
                unimplemented!()
            }
            fn get_storage(&self, data: &mut [$type]) {
                /* XXX presupposes contiguity */
                let offset = self.storage_offset() as isize;
                let s = unsafe {(*(*self.t).storage).data.offset(offset)};
                let s = s as *const voidp;
                let d = data.as_mut_ptr() as *mut voidp;
                unsafe {memcpy(d, s, data.len()) };
            }
            fn gt_tensor(&self, other: *mut voidp, out: *mut voidp) {
                unimplemented!()
            }
            fn inner(&self) -> *mut voidp {
                self.t as *mut voidp
            }
            fn iter(&self) -> Box<Iterator<Item=$type>> {
                unimplemented!()
            }
            fn le_tensor(&self, other: *mut voidp, out: *mut voidp) {
                unimplemented!()
            }
            fn le_value(&self, value: $type, out: *mut voidp) {
                unimplemented!()
            }
            fn len(&self) -> usize {
                self.len()
            }
            fn lt_tensor(&self, other: *mut voidp, out: *mut voidp) {
                unimplemented!()
            }
            fn lt_value(&self, value: $type, out: *mut voidp) {
                unimplemented!()
            }
            fn max_reduce(&self, values: *mut voidp, indices: *mut voidp, dim: usize, keepdim: bool) {
                let valuesp = values as *mut $thname;
                let indicesp = values as *mut THLongTensor;
                let dim = dim as i32;
                let keep = keepdim as i32;
                unsafe { concat_idents!($thname, _max)(valuesp, indicesp, self.t, dim, keep) }
            }
            fn min_reduce(&self, values: *mut voidp, indices: *mut voidp, dim: usize, keepdim: bool) {
                let valuesp = values as *mut $thname;
                let indicesp = values as *mut THLongTensor;
                let dim = dim as i32;
                let keep = keepdim as i32;
                assert!(dim < self.dim());
                unsafe { concat_idents!($thname, _min)(valuesp, indicesp, self.t, dim, keep) }
            }
            fn mm(&mut self, mat1: *mut voidp, mat2: *mut voidp) {
                let (mat1p, mat2p) = (mat1 as *mut $thname, mat2 as *mut $thname);
                unsafe {
                    concat_idents!($thname, _addmm)(self.t,
                                                    0 as $type,
                                                    self.t,
                                                    1 as $type,
                                                    mat1p,
                                                    mat2p)
                };
            }
            fn mul(&mut self, src: *mut voidp, value: $type) {
                let srcp = src as *mut $thname;
                unsafe { concat_idents!($thname, _mul)(self.t, srcp, value)};
            }
            fn mult(&mut self,
                    src: *mut voidp,
                    value: *mut voidp) {
                let srcp = src as *mut $thname;
                let valuep = value as *mut $thname;
                unsafe { concat_idents!($thname, _cmul)(self.t, valuep, srcp)};
            }
            fn ne_tensor(&self, other: *mut voidp, out: *mut voidp) {
                unimplemented!()
            }
            fn new(&self) -> Tensor<$type> {
                Tensor { value: RcMutNew($name ::new()) }
            }
            fn prod(&self, mut result: &mut f64) {
                let val: $type;
                let r = unsafe { concat_idents!($thname, _prodall)(self.t) };
                *result = <f64 as num::NumCast>::from(r).unwrap();
            }
            fn resize(&mut self, dims: &[usize]) {
                let dims : Vec<i64> = dims.iter().map(|v| *v as i64).collect();
                let dims = LongStorage::with_data(dims);
                unsafe { concat_idents!($thname, _resize)(self.t, dims.t, std::ptr::null_mut()) };
            }
            fn size(&self) -> Vec<usize> {
                let d = unsafe { std::slice::from_raw_parts((*self.t).size as *mut usize,
                                                            (*self.t).nDimension as usize)};
                d.to_vec()
            }
            fn s(&self, dim: &[usize]) -> Tensor<$type> {
                let mut increment : usize = self.size().iter().product();
                let sizes = self.size();
                if sizes.len() < dim.len() {
                    panic!("bad slice index {:?}", dim);
                }
               /* calculate new storage offset and validate */
                for i in 0..dim.len() {
                    if dim[i] >= sizes[i] {
                        panic!("{} out of range {:?}", dim[i], sizes[i]);
                    }
                }
                let mut offset = 0;
                let mut new_dims = sizes.clone();
                for i in 0..dim.len() {
                    increment /= sizes[i];
                    offset += dim[i] * increment;
                    new_dims.remove(0);
                }
                let dims_long : Vec<i64> = new_dims.iter().map(|t| *t as i64).collect();
                let sizes = LongStorage::with_data(dims_long.as_slice());
                let storage = self.storage();
                let t = unsafe {
                    concat_idents!($thname, _newWithStorage)(storage.t,
                                                             offset as isize,
                                                             sizes.t,
                                                             std::ptr::null_mut())
                };
                let t = $name :: from_parts(t);
                Tensor { value: RcMutNew(t) }
            }
            fn set_storage(&mut self, v: &[$type]) {
                let storage_offset = self.storage_offset();
                let mut s = self.storage();
                assert_eq!(v.len(), s.len());
                // XXX memcpy
                for i in 0..s.len() {
                    s[(storage_offset + i)] = v[i]
                }
            }
            fn squeeze(&mut self, dim: Option<usize>) {
                let mut dims = Vec::new();
                if let Some(t) = dim {
                    dims.push(t)
                } else {
                    for (i, d) in self.size().iter().enumerate() {
                        if *d == 1 {
                            dims.push(i);
                        }
                    }
                    dims.reverse();
                }
                for d in dims {
                    let p = ::std::ptr::null_mut();
                    unsafe {concat_idents!($thname, _squeeze1d)(self.t, p, d as i32) };
                }
            }
            fn sum(&self, mut result: &mut f64) {
                let val: $type;
                let r = unsafe { concat_idents!($thname, _sumall)(self.t) };
                *result = <f64 as num::NumCast>::from(r).unwrap();
            }
            fn sum_reduce(&mut self, input: *mut voidp, dim: usize, keepdim: bool) {
                let input = input as *mut $thname;
                unsafe  { concat_idents!($thname, _sum)(self.t, input, dim as i32, keepdim as i32)};
            }
            fn transpose(&mut self, src: *mut voidp, dim0: usize, dim1: usize){
                let (dim0, dim1) = (dim0 as i32, dim1 as i32);
                let srcp = src as *mut $thname;
                unsafe {concat_idents!($thname, _transpose)(self.t, srcp, dim0, dim1)};
            }
            fn to_rust_tensor(&self) -> RustTensor<$type> {
                self.to_rust_tensor()
            }
            fn uniform_(&mut self, range: (f64, f64)) {
                let g = Generator::new();
                #[allow(unused_unsafe)]
                unsafe { concat_idents!($thname, _uniform)(self.t, g.t, range.0, range.1) };
            }
            fn unsqueeze(&mut self, dim: usize) {
                let p = ::std::ptr::null_mut();
                unsafe {concat_idents!($thname, _squeeze1d)(self.t, p, dim as i32) };
            }
            fn view(&self, dims: &[isize]) -> Tensor<$type> {
                let dims_long : Vec<i64> = dims.iter().map(|t| *t as i64).collect();
                let size = LongStorage::with_data(dims_long.as_slice());
                let inferred_size = unsafe {
                    let numel = concat_idents!($thname, _nElement)(self.t);
                    let p = THLongStorage_newInferSize(size.t, numel);
                    LongStorage {t: p}
                };
                let t = unsafe { concat_idents!($thname, _newView)(self.t, inferred_size.t)  };
                let t = $name :: from_parts(t);
                Tensor {value: RcMutNew(t) }
            }
            fn zero(&mut self) {
                unsafe { concat_idents!($thname, _zero)(self.t) };
            }
        }
        impl Default for $name {
            fn default() -> Self {
                $name ::new()
            }
        }
        impl<'a> Index<&'a [isize]> for $name {
            type Output = $type;

            fn index(&self, idx: &'a [isize]) -> &Self::Output {
                let mut index = 0;
                let lastidx = max(0, idx.len() as isize - 1) as usize;
                let dims = self.size();
                if idx.len() != dims.len() {
                    panic!("bad dimlen")
                }
                for i in 0..lastidx {
                    if idx[i] >= dims[i] as isize {
                        panic!("bad dimlen")
                    }
                    index += idx[i] as usize * dims[i];
                }
                if idx[lastidx] >= dims[lastidx] as isize {
                    panic!("bad dimlen")
                }
                index += idx[lastidx] as usize;
                unsafe_index!(self, index)
            }
        }

        impl<'a> IndexMut<&'a [isize]> for $name {
            fn index_mut(&mut self, idx: &'a [isize]) -> &mut Self::Output {
                let mut index = 0;
                let lastidx = max(0, idx.len() as isize - 1) as usize;
                let dims = self.size();
                if idx.len() != dims.len() {
                    panic!("bad dimlen")
                }
                for i in 0..lastidx {
                    if idx[i] >= dims[i] as isize {
                        panic!("bad dimlen")
                    }
                    index += idx[i] as usize * dims[i];
                }
                if idx[lastidx] >= dims[lastidx] as isize {
                    panic!("bad dimlen")
                }
                index += idx[lastidx] as usize;
                unsafe_index_mut!(self, index)
            }
        }

        impl Index<usize> for $name {
            type Output = $type;
            fn index(&self, idx: usize) -> &Self::Output {
                let dims = self.dim();
                if dims != 1 {
                    panic!("bad index size 1D index an {}D dims: {:?}", dims, self.size())
                };
                let size = self.size();
                if size[0] <= idx {
                    panic!("idx {} out of range {}", idx, size[0])
                };
                unsafe_index!(self, idx)
            }
        }
        impl Drop for $name {
            fn drop(&mut self) {
                unsafe { concat_idents!($thname, _free)(self.t) }
            }
        }
        impl Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
                where S: Serializer
            {
                let rt = self.to_rust_tensor();
                let result = rt.serialize(serializer)?;
                Ok(result)
            }
        }
        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
                where D: Deserializer<'de>
            {
                unimplemented!()
            }
        }
    }
}

#[allow(non_snake_case, unused_variables)]
pub fn THByteTensor_uniform(self_: *mut THByteTensor,
                            _generator: *mut THGenerator,
                            a: f64,
                            b: f64) {
    panic!("no such function")
}
#[allow(non_snake_case, unused_variables)]
pub fn THLongTensor_uniform(self_: *mut THLongTensor,
                            _generator: *mut THGenerator,
                            a: f64,
                            b: f64) {
    panic!("no such function")
}


impl_tensor_impl!(FloatTensor, f32, THFloatTensor, FloatStorage);
impl_tensor_impl!(DoubleTensor, f64, THDoubleTensor, DoubleStorage);
impl_tensor_impl!(LongTensor, i64, THLongTensor, LongStorage);
impl_tensor_impl!(ByteTensor, u8, THByteTensor, ByteStorage);

pub fn make_vec(val: usize, count: usize) -> Vec<isize> {
    let mut vec = Vec::new();
    for _ in 0..count {
        vec.push(val as isize)
    }
    vec
}
