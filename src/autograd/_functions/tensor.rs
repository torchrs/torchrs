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
        let dims : Vec<isize> = input.size().iter().map(|v| *v as isize).collect();
        self.saved_trvalue.push(dims.into());
        let output = input.view(self.args.dims.clone());
        // XXX mark shared?
        vec![output]
    }
    fn backward(&mut self, grad_output: &mut OptTensorKindList) -> OptTensorKindList {
        let mut grad_output = grad_output.remove(0).unwrap();
        let dims: Vec<isize> = self.saved_trvalue.remove(0).into();
        println!("View backward");
        vec![grad_output.contiguous().view(dims).into(), None]
    }
}
#[derive(Clone)]
pub struct MaskedFillArgs {
    value: f64,
    inplace: bool,
}

impl_func_args_other!(MaskedFill, MaskedFillArgs);
impl FuncIntf for MaskedFill {
    fn forward(&mut self, input_list: &mut TensorKindList) -> TensorKindList {
        let (mut input, mut mask) = (input_list.remove(0), input_list.remove(0));

        self.saved_trvalue.push(input.size().into());
        let mut tensor = if self.args.inplace {
            self.mark_dirty(&vec![input.clone()]);
            input.clone()
        } else {
            input.copy()
        };
        self.save_for_backward(&vec![mask.clone()]);
        tensor.masked_fill_(mask.into(), self.args.value);
        vec![tensor]
    }
    fn backward(&mut self, grad_output: &mut OptTensorKindList) -> OptTensorKindList {
        unimplemented!();
        let mut grad_output = grad_output.remove(0).unwrap();
        let dims: Vec<isize> = self.saved_trvalue.remove(0).into();
        println!("MaskedFill backward");
        vec![grad_output.contiguous().view(dims).into(), None]
    }
}