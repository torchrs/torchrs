use autograd::{Function, FuncIntf, FuncDelegate, FIWrap};
use tensor::{TensorKindList, OptTensorKindList, NewSelf};


impl_func!(LogSoftmax);
impl FuncIntf for LogSoftmax {
    fn forward(&mut self, input: &mut TensorKindList) -> TensorKindList {
        unimplemented!()
    }
    fn backward(&mut self, input: &mut OptTensorKindList) -> OptTensorKindList {
        unimplemented!()
    }
}


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
        let mut backend = input_[0].backend();
        let mut output = if self.inplace {
            self.mark_dirty(input_);
            input_[0].clone()
        } else {
            input_[0].new(())
        };
        self.save_for_backward(input_);
        backend.Threshold_updateOutput(&mut input_[0],
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
}
