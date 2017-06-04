use autograd::{Function, FuncIntf, FuncDelegate, Variable, FIWrap};
use macros::*;
use tensor::TensorKindList;
use ::*;

impl_func!(LinearF);

impl FuncIntf for LinearF {
    fn forward(&mut self, mut input: &mut TensorKindList) -> TensorKindList {
        unimplemented!()
    }
    fn backward(&mut self, mut input: &mut TensorKindList) -> TensorKindList {
        unimplemented!()
    }
}
