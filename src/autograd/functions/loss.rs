use autograd::{Function, FuncIntf, FuncIntfKind, FuncDelegate, Variable, VarId};
use macros::*;
use tensor::{RefTensorList, RefTensorKindList, TensorList, TensorKindList};

pub struct LogSoftmax {
    delegate: Function,
}

impl LogSoftmax {
    pub fn new() -> Self {
        LogSoftmax { delegate: Function::new() }
    }
}
impl_func_delegate!(LogSoftmax);

impl FuncIntf for LogSoftmax {
    fn forward<T>(&mut self, mut input: &RefTensorList<T>) -> TensorList<T> {
        unimplemented!()
    }
    fn backward<T>(&mut self, mut input: &RefTensorList<T>) -> TensorList<T> {
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

pub struct NLLLoss {
    delegate: Function,
    args: NLLLossArgs,
}

impl NLLLoss {
    pub fn new(args: &NLLLossArgs) -> Self {
        NLLLoss {
            delegate: Function::new(),
            args: args.clone(),
        }
    }
}

impl FuncDelegate for NLLLoss {
    fn delegate(&mut self) -> &mut Function {
        &mut self.delegate
    }
}

impl FuncIntfKind for NLLLoss {
    fn forwardx<'a>(&mut self, input: &RefTensorKindList<'a>) -> TensorKindList {
        unimplemented!();
    }
    fn backwardx<'a>(&mut self, input: &RefTensorKindList<'a>) -> TensorKindList {
        unimplemented!();
    }
}
