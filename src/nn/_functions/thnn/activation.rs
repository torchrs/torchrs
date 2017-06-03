use autograd::{Function, FuncIntf, FuncDelegate, Variable, VarKindList, FIWrap};
use macros::*;
use tensor::TensorKindList;
use ::*;

pub struct Threshold {
    delegate: Function,
    threshold: f32,
    value: f32,
    inplace: bool,
}

impl Threshold {
    pub fn new(threshold: f32, value: f32, inplace: bool) -> FIWrap<Self> {
        FIWrap::new(Threshold {
                        delegate: Function::new(),
                        threshold: threshold,
                        value: value,
                        inplace: inplace,
                    })
    }
}

impl_func_delegate!(Threshold);

impl FuncIntf for Threshold {
    fn forward(&mut self, input: &mut TensorKindList) -> TensorKindList {
        unimplemented!()
    }
    fn backward(&mut self, input: &mut TensorKindList) -> TensorKindList {
        unimplemented!()
    }
}
