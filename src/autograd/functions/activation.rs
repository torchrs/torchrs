use autograd::{Function, FuncIntf, FuncDelegate, Variable};
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
    pub fn new(threshold: f32, value: f32, inplace: bool) -> RcMut<Self> {
        let t = RcMutNew(Threshold {
                             delegate: Function::new(),
                             threshold: threshold,
                             value: value,
                             inplace: inplace,
                         });
        t.borrow_mut().delegate().init(t.clone());
        t
    }
}
type RcMutThresh = RcMut<Threshold>;
impl_func_delegate!(Threshold);

impl FuncIntf for Threshold {
    fn forward(&mut self, mut input: &TensorKindList) -> TensorKindList {
        unimplemented!()
    }
    fn backward(&mut self, mut input: &TensorKindList) -> TensorKindList {
        unimplemented!()
    }
}
