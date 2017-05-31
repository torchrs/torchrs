use autograd::{Function, FuncIntf, FuncDelegate, Variable};
use macros::*;
use tensor::{RefTensorList, TensorList};

pub struct Threshold {
    delegate: Function,
    threshold: f32,
    value: f32,
    inplace: bool,
}

impl Threshold {
    pub fn new(threshold: f32, value: f32, inplace: bool) -> Self {
        Threshold {
            delegate: Function::new(),
            threshold: threshold,
            value: value,
            inplace: inplace,
        }
    }
}
impl_func_delegate!(Threshold);

impl FuncIntf for Threshold {
    fn forward<T>(&mut self, mut input: &RefTensorList<T>) -> TensorList<T> {
        unimplemented!()
    }
    fn backward<T>(&mut self, mut input: &RefTensorList<T>) -> TensorList<T> {
        unimplemented!()
    }
}
