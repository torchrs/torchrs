use autograd::{Function, FuncIntf, FuncDelegate, Variable, VarId, FIWrap};
use macros::*;
use tensor::{RefTensorList, RefTensorKindList, TensorList, TensorKindList};
use ::*;

#[derive(Clone)]
pub struct LogSoftmax {
    delegate: Function,
}

impl LogSoftmax {
    pub fn new() -> FIWrap<Self> {
        FIWrap::new(LogSoftmax { delegate: Function::new() })
    }
}
impl_func_delegate!(LogSoftmax);

impl FuncIntf for LogSoftmax {
    fn forward(&mut self, mut input: &mut TensorKindList) -> TensorKindList {
        unimplemented!()
    }
    fn backward(&mut self, mut input: &mut TensorKindList) -> TensorKindList {
        unimplemented!()
    }
}

#[builder(pattern="owned")]
#[derive(Builder, Clone)]
pub struct NLLLossArgs {
    #[builder(default="false")]
    pub size_average: bool,
    #[builder(default="None")]
    pub weight: Option<VarId>,
}

impl Default for NLLLossArgs {
    fn default() -> Self {
        NLLLossArgsBuilder::default().build().unwrap()
    }
}

#[derive(Clone)]
pub struct NLLLoss {
    delegate: Function,
    args: NLLLossArgs,
}

impl NLLLoss {
    pub fn new(args: &NLLLossArgs) -> FIWrap<Self> {
        FIWrap::new(NLLLoss {
                        delegate: Function::new(),
                        args: args.clone(),
                    })
    }
}

impl FuncDelegate for NLLLoss {
    fn delegate(&mut self) -> &mut Function {
        &mut self.delegate
    }
}

impl FuncIntf for NLLLoss {
    fn forward(&mut self, input: &mut TensorKindList) -> TensorKindList {
        unimplemented!();
    }
    fn backward(&mut self, input: &mut TensorKindList) -> TensorKindList {
        unimplemented!();
    }
}
