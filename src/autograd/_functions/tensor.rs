use autograd::{Function, FuncIntf, FuncDelegate, FIWrap};
use tensor::{OptTensorKindList, TensorKindList};

#[derive(Clone)]
pub struct ViewArgs {
    pub dims: Vec<isize>,
}

impl_func_args_other!(View, ViewArgs);
impl FuncIntf for View {
    fn forward(&mut self, input_list: &mut TensorKindList) -> TensorKindList {
        let mut input = input_list.remove(0);
        self.saved_trvalue.push(input.size().into());
        let output = input.view(self.args.dims.clone());
        // XXX mark shared?
        vec![output]
    }
    fn backward(&mut self, grad_output: &mut OptTensorKindList) -> OptTensorKindList {
        let mut grad_output = grad_output.remove(0).unwrap();
        let dims: Vec<isize> = self.saved_trvalue.remove(0).into();
        vec![grad_output.contiguous().view(dims).into(), None]
    }
}
