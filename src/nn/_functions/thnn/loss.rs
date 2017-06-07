use autograd::{Function, FuncIntf, FuncDelegate, Variable, VarId, FIWrap};
use tensor::{RefTensorList, RefTensorKindList, TensorList, TensorKindList, OptTensorKindList};

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

impl_func_args!(NLLLoss, NLLLossArgs);

impl FuncIntf for NLLLoss {
    fn forward(&mut self, input: &mut TensorKindList) -> TensorKindList {
        unimplemented!();
    }
    fn backward(&mut self, input: &mut OptTensorKindList) -> OptTensorKindList {
        unimplemented!();
    }
}
