use tensor::{Tensor, NumLimits};

pub struct THVec<T> {
    data: Vec<T>,
    dims: Vec<usize>,
}
impl<T: NumLimits> THVec<T> {
    fn new(dims: Vec<usize>, data: Vec<T>) -> Self {
        THVec {
            dims: dims,
            data: data,
        }
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
        THVec::new(vec![], v)
    }
}
impl<T: NumLimits> From<Vec<Tensor<T>>> for THVec<T> {
    fn from(input: Vec<Tensor<T>>) -> Self {
        /*
        let v = input.iter().flat_map(|d| d.iter()).map(|d| *d).collect();
        THVec::new(2, v)
        */
        unimplemented!()
    }
}
impl<T: NumLimits> From<[T; 0]> for THVec<T> {
    fn from(input: [T; 0]) -> Self {
        unimplemented!()
    }
}
impl<T: NumLimits> From<[T; 1]> for THVec<T> {
    fn from(input: [T; 1]) -> Self {
        unimplemented!()
    }
}
impl<T: NumLimits> From<[T; 2]> for THVec<T> {
    fn from(input: [T; 2]) -> Self {
        unimplemented!()
    }
}
impl<T: NumLimits> From<[T; 3]> for THVec<T> {
    fn from(input: [T; 3]) -> Self {
        unimplemented!()
    }
}

pub trait TensorNew<T: NumLimits> {
    fn tensor_new(arg: THVec<T>) -> Tensor<T> {
        unimplemented!()
    }
}

impl<T: NumLimits> TensorNew<T> for Tensor<T> {
    default fn tensor_new(arg: THVec<T>) -> Tensor<T> {
        unreachable!()
    }
}

impl TensorNew<u8> for Tensor<u8> {
    fn tensor_new(arg: THVec<u8>) -> Tensor<u8> {
        unimplemented!()
    }
}
impl TensorNew<i64> for Tensor<i64> {
    fn tensor_new(arg: THVec<i64>) -> Tensor<i64> {
        unimplemented!()
    }
}
impl TensorNew<f32> for Tensor<f32> {
    fn tensor_new(arg: THVec<f32>) -> Tensor<f32> {
        let t = ::RcMutNew(::tensor::FloatTensor::with_capacity(arg.dims.as_slice()));
        ::tensor::Tensor { value: t }
    }
}
impl TensorNew<f64> for Tensor<f64> {
    fn tensor_new(arg: THVec<f64>) -> Tensor<f64> {
        unimplemented!()
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
pub fn tensor<S, T>(arg: S) -> Tensor<T>
    where T: NumLimits,
          S: Into<THVec<T>>
{
    Tensor::tensor_new(arg.into())
}
