use tensor::{Tensor, NumLimits, THVec, THVecGeneric, THDims};
use num::NumCast;

impl<T: NumLimits> THVec<T> {
    pub fn new(dims: Vec<usize>, data: Vec<T>) -> Self {
        THVec {
            dims: dims,
            data: data,
        }
    }
}
impl THVecGeneric {
    pub fn new(dims: Vec<usize>, data: Vec<i64>) -> Self {
        THVecGeneric {
            dims: dims,
            data: data,
        }
    }
}
impl THDims {
    fn new(dims: Vec<usize>) -> Self {
        THDims { dims: dims }
    }
}
impl<T: NumLimits> From<Vec<Vec<Vec<T>>>> for THVec<T> {
    fn from(input: Vec<Vec<Vec<T>>>) -> Self {
        let dims = vec![input.len(), input[0].len(), input[0][0].len()];
        let v = input
            .iter()
            .flat_map(|d| d.iter())
            .flat_map(|d| d.iter())
            .map(|d| *d)
            .collect();
        THVec::new(dims, v)
    }
}

impl<T: NumLimits> From<Vec<Vec<T>>> for THVec<T> {
    fn from(input: Vec<Vec<T>>) -> Self {
        let dims = vec![input.len(), input[0].len()];
        let v = input.iter().flat_map(|d| d.iter()).map(|d| *d).collect();
        THVec::new(dims, v)
    }
}
impl<T: NumLimits> From<Vec<T>> for THVec<T> {
    fn from(input: Vec<T>) -> Self {
        let dims = vec![input.len()];
        THVec::new(dims, input)
    }
}
impl<T: NumLimits> From<T> for THVec<T> {
    fn from(input: T) -> Self {
        let v = vec![input];
        THVec::new(vec![1], v)
    }
}
impl<T: NumLimits> From<T> for THVecGeneric {
    fn from(input: T) -> Self {
        let v: Vec<i64> = vec![<i64 as NumCast>::from(input).unwrap()];
        THVecGeneric::new(vec![1], v)
    }
}

#[allow(unused_variables)]
impl<T: NumLimits> From<()> for THVec<T> {
    fn from(input: ()) -> Self {
        THVec::new(vec![], vec![])
    }
}
#[allow(unused_variables)]
impl From<()> for THVecGeneric {
    fn from(input: ()) -> Self {
        THVecGeneric::new(vec![], vec![])
    }
}
#[allow(unused_variables)]
impl From<()> for THDims {
    fn from(input: ()) -> Self {
        THDims::new(vec![])
    }
}
impl<T: NumLimits> From<[usize; 2]> for THVec<T> {
    fn from(input: [usize; 2]) -> Self {
        THVec::new(input.to_vec(), vec![])
    }
}
impl<T: NumLimits> From<[usize; 4]> for THVec<T> {
    fn from(input: [usize; 4]) -> Self {
        THVec::new(input.to_vec(), vec![])
    }
}
impl<T: NumLimits> From<(usize, usize)> for THVec<T> {
    fn from(input: (usize, usize)) -> Self {
        THVec::new(vec![input.0, input.1], vec![])
    }
}
impl From<(usize, usize)> for THVecGeneric {
    fn from(input: (usize, usize)) -> Self {

        THVecGeneric::new(vec![input.0, input.1], vec![])
    }
}
impl From<(usize)> for THVecGeneric {
    fn from(input: (usize)) -> Self {

        THVecGeneric::new(vec![input], vec![])
    }
}
impl<T: NumLimits> From<(usize)> for THVec<T> {
    fn from(input: (usize)) -> Self {
        THVec::new(vec![input], vec![])
    }
}
impl<T: NumLimits> From<Vec<usize>> for THVec<T> {
    fn from(input: Vec<usize>) -> Self {
        THVec::new(input, vec![])
    }
}
impl From<Vec<usize>> for THVecGeneric {
    fn from(input: Vec<usize>) -> Self {
        THVecGeneric::new(input, vec![])
    }
}
impl From<Vec<usize>> for THDims {
    fn from(input: Vec<usize>) -> Self {
        THDims::new(input)
    }
}
impl<S: NumLimits, D: NumLimits> From<Tensor<S>> for THVec<D> {
    fn from(input: Tensor<S>) -> Self {
        unimplemented!()
    }
}
impl<T: NumLimits> From<Vec<THVec<T>>> for THVec<T> {
    fn from(input: Vec<THVec<T>>) -> Self {
        unimplemented!()
    }
}
impl<S: NumLimits, D: NumLimits> From<Vec<Tensor<S>>> for THVec<D> {
    fn from(input: Vec<Tensor<S>>) -> Self {
        let mut d: Vec<D> = Vec::with_capacity(input.len() * input[0].len());
        let len = input[0].len();
        let mut tmp: Vec<S> = Vec::with_capacity(len);
        unsafe {
            d.set_len(input.len() * input[0].len());
            tmp.set_len(len);
        }
        for i in 0..input.len() {
            input[i].get_storage(&mut tmp, len);
            for j in 0..len {
                unsafe {
                    *(d.get_unchecked_mut(i * len + j)) = <D as ::num::NumCast>::from(tmp[j])
                        .unwrap();
                }
            }
        }
        let mut sizes = input[0].size();
        let mut dims = vec![input.len()];
        dims.append(&mut sizes);
        THVec::new(dims, d)
    }
}
impl From<THVecGeneric> for THVec<u8> {
    fn from(input: THVecGeneric) -> Self {
        let data: Vec<u8> = input
            .data
            .iter()
            .map(|t| <u8 as NumCast>::from(*t).unwrap())
            .collect();
        THVec::new(input.dims.clone(), data)
    }
}
impl From<THVecGeneric> for THVec<f32> {
    fn from(input: THVecGeneric) -> Self {
        let data: Vec<f32> = input
            .data
            .iter()
            .map(|t| <f32 as NumCast>::from(*t).unwrap())
            .collect();
        THVec::new(input.dims.clone(), data)
    }
}
impl From<THVecGeneric> for THVec<i64> {
    fn from(input: THVecGeneric) -> Self {
        let data: Vec<i64> = input
            .data
            .iter()
            .map(|t| <i64 as NumCast>::from(*t).unwrap())
            .collect();
        THVec::new(input.dims.clone(), data)
    }
}

impl<T: NumLimits> From<THVec<T>> for THDims {
    fn from(input: THVec<T>) -> Self {
        THDims::new(input.dims.clone())
    }
}
impl From<THVecGeneric> for THDims {
    fn from(input: THVecGeneric) -> Self {
        THDims::new(input.dims.clone())
    }
}

trait TensorNew<T: NumLimits> {
    fn tensor_new(arg: THDims) -> Tensor<T>;
}

impl<T: NumLimits> TensorNew<T> for Tensor<T> {
    #[allow(unused_variables)]
    default fn tensor_new(arg: THDims) -> Tensor<T> {
        unreachable!()
    }
}

impl TensorNew<u8> for Tensor<u8> {
    fn tensor_new(arg: THDims) -> Tensor<u8> {
        let t = ::RcMutNew(::tensor::ByteTensor::with_capacity(arg.dims.as_slice()));
        ::tensor::Tensor { value: t }
    }
}
impl TensorNew<i64> for Tensor<i64> {
    fn tensor_new(arg: THDims) -> Tensor<i64> {
        let t = ::RcMutNew(::tensor::LongTensor::with_capacity(arg.dims.as_slice()));
        ::tensor::Tensor { value: t }
    }
}
impl TensorNew<f32> for Tensor<f32> {
    fn tensor_new(arg: THDims) -> Tensor<f32> {
        let t = ::RcMutNew(::tensor::FloatTensor::with_capacity(arg.dims.as_slice()));
        ::tensor::Tensor { value: t }
    }
}
impl TensorNew<f64> for Tensor<f64> {
    fn tensor_new(arg: THDims) -> Tensor<f64> {
        let t = ::RcMutNew(::tensor::DoubleTensor::with_capacity(arg.dims.as_slice()));
        ::tensor::Tensor { value: t }
    }
}

pub fn byte_tensor<T>(arg: T) -> Tensor<u8>
    where T: Into<THVec<u8>>
{
    tensor(arg)
}
pub fn long_tensor<T>(arg: T) -> Tensor<i64>
    where T: Into<THVec<i64>>
{
    tensor(arg)
}
pub fn float_tensor<T>(arg: T) -> Tensor<f32>
    where T: Into<THVec<f32>>
{
    tensor(arg)
}
pub fn double_tensor<T>(arg: T) -> Tensor<f64>
    where T: Into<THVec<f64>>
{
    tensor(arg)
}
pub fn zeros<S, T>(arg: S) -> Tensor<T>
    where T: NumLimits,
          S: Into<THVec<T>>
{
    tensor(arg).zero_()
}
pub fn tensor<S, T>(arg: S) -> Tensor<T>
    where T: NumLimits,
          S: Into<THVec<T>>
{
    let t: THVec<T> = arg.into();
    let dims = t.dims.clone().into();
    let mut out = Tensor::tensor_new(dims);
    if t.data.len() > 0 {
        out.set_storage(t.data);
    }
    out
}
