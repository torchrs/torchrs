use autograd::{Function, FuncIntf, FuncDelegate, Variable};
use macros::*;
use tensor::TensorKindList;

#[derive(Clone)]
pub struct LinearF {
    delegate: Function,
}

impl LinearF {
    pub fn new() -> Self {
        LinearF { delegate: Function::new() }
    }
}
impl_func_delegate!(LinearF);

impl FuncIntf for LinearF {
    fn forward(&mut self, mut input: &TensorKindList) -> TensorKindList {
        unimplemented!()
    }
    fn backward(&mut self, mut input: &TensorKindList) -> TensorKindList {
        unimplemented!()
    }
}
