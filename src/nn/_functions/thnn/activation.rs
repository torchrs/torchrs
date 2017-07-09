use autograd::{Function, FuncIntf, FuncDelegate, FIWrap, OptVarKindList, Variable, VariableArgs,
               VarKind};
use tensor::{TensorKindList, OptTensorKindList};

pub struct Threshold {
    delegate: Function,
    saved_tensors: Vec<::tensor::TensorKind>,
    threshold: f64,
    value: f64,
    inplace: bool,
}

impl Threshold {
    pub fn new(threshold: f64, value: f64, inplace: bool) -> FIWrap<Self> {
        if inplace {
            panic!("in-place processing requires value ({}) to not \
                    exceed threshold ({})",
                   value,
                   threshold);
        }

        FIWrap::new(Threshold {
                        delegate: Function::new(),
                        saved_tensors: Vec::new(),
                        threshold: threshold,
                        value: value,
                        inplace: inplace,
                    })
    }
}

impl_func_delegate!(Threshold);

impl FuncIntf for Threshold {
    fn forward(&mut self, input_: &mut TensorKindList) -> TensorKindList {
        let mut input = input_.remove(0);
        let mut backend = input.backend();
        let mut output = if self.inplace {
            self.mark_dirty(&vec![input.clone()]);
            input.clone()
        } else {
            input.new(()).resize_as_(&input)
        };
        // XXX check if training
        self.saved_tensors.push(input.clone());
        backend.Threshold_updateOutput(&mut input,
                                       &mut output,
                                       self.threshold,
                                       self.value,
                                       self.inplace);
        vec![output]
    }
    fn backward(&mut self, grad_output_list: &mut OptTensorKindList) -> OptTensorKindList {
        let mut grad_output = grad_output_list.remove(0).unwrap();
        let mut input = self.saved_tensors.remove(0);
        let needs_input_grad = self.needs_input_grad().clone();

        println!("Threshold backward");
        let grad_input = if needs_input_grad[0] {
            let mut grad_input = input.new(());
            let mut backend = input.backend();
            backend.Threshold_updateGradInput(&mut input,
                                              &mut grad_output,
                                              &mut grad_input,
                                              self.threshold,
                                              self.value,
                                              false);
            Some(grad_input)
        } else {
            None
        };
        vec![grad_input, None, None, None]
    }
}
