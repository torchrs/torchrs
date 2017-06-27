use tensor::{Tensor, TensorKind, NumLimits};

struct THVec<T> {
    data: Vec<T>,
    dims: u32,
}
impl<T: NumLimits> THVec<T> {
    fn new(dims: u32, data: Vec<T>) -> Self {
        THVec {
            dims: dims,
            data: data,
        }
    }
}

impl<T: NumLimits> From<Vec<Vec<Vec<T>>>> for THVec<T> {
    fn from(input: Vec<Vec<Vec<T>>>) -> Self {
        let v = input
            .iter()
            .flat_map(|d| d.iter())
            .flat_map(|d| d.iter())
            .map(|d| *d)
            .collect();
        //.iter().flat_map(|d| d.to_vec()).collect();
        THVec::new(3, v)
    }
}
impl<T: NumLimits> From<Vec<Vec<T>>> for THVec<T> {
    fn from(input: Vec<Vec<T>>) -> Self {
        let v = input.iter().flat_map(|d| d.iter()).map(|d| *d).collect();
        THVec::new(2, v)
    }
}
impl<T: NumLimits> From<Vec<T>> for THVec<T> {
    fn from(input: Vec<T>) -> Self {
        unimplemented!()
    }
}
impl<T: NumLimits> From<T> for THVec<T> {
    fn from(input: T) -> Self {
        let v = vec![input];
        THVec::new(0, v)
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
    unimplemented!()
}
