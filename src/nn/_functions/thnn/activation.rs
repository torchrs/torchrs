use autograd::{Function, FuncIntf, FuncDelegate, FIWrap, OptVarKindList};
use tensor::{TensorKindList, OptTensorKindList};

pub struct Threshold {
    delegate: Function,
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
        self.save_for_backward(&vec![input.clone()]);
        backend.Threshold_updateOutput(&mut input,
                                       &mut output,
                                       self.threshold,
                                       self.value,
                                       self.inplace);
        vec![output]
    }
    fn backward(&mut self, input: &mut OptTensorKindList) -> OptTensorKindList {
        /* Why are they doing backprop on a volatile variable? */
        unimplemented!()
    }
    fn backward_var(&mut self, input: &mut OptVarKindList) -> OptVarKindList {
        unimplemented!()
    }
}
