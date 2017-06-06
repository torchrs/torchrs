#![allow(unused_variables)]
use autograd::{Function, FuncIntf, FuncDelegate, FIWrap};
use tensor::{TensorKindList, OptTensorKindList, TensorKind, New};

#[builder(pattern="owned")]
#[derive(Builder, Clone)]
pub struct DropoutArgs {
    #[builder(default="0.5")]
    pub p: f32,
    #[builder(default="false")]
    pub training: bool,
    #[builder(default="false")]
    pub inplace: bool,
}

impl Default for DropoutArgs {
    fn default() -> Self {
        DropoutArgsBuilder::default().build().unwrap()
    }
}

impl_func_args!(Dropout1d, DropoutArgs);
impl_func_args!(Dropout2d, DropoutArgs);

trait Noise: FuncIntf {
    fn make_noise(&self, input: &TensorKind) -> TensorKind;
}

trait Dropout: Noise + FuncIntf {
    fn dropout_forward(&mut self,
                       input: &mut TensorKindList,
                       args: &DropoutArgs)
                       -> TensorKindList {
        let output = if args.inplace {
            self.mark_dirty(input);
            input[0].clone()
        } else {
            input[0].copy()
        };
        let output = if args.p > 0. && args.training {
            let noise = self.make_noise(&input[0])
                .bernoulli_(&(1. - args.p).into())
                .div_(&(1. - args.p).into())
                .expand_as(&input[0]);
            self.save_for_backward(&vec![noise.clone()]);
            output.mult_(&noise)
        } else {
            output
        };
        vec![output]
    }
}

impl Dropout for Dropout1d {}
impl Dropout for Dropout2d {}
impl Noise for Dropout1d {
    fn make_noise(&self, input: &TensorKind) -> TensorKind {
        input.new_(())
    }
}
impl Noise for Dropout2d {
    fn make_noise(&self, input: &TensorKind) -> TensorKind {
        let mut v = vec![input.size()[0], input.size()[1]];
        for _ in 0..input.dim() - 2 {
            v.push(1)
        }
        input.new_(v)
    }
}
impl FuncIntf for Dropout1d {
    fn forward(&mut self, input: &mut TensorKindList) -> TensorKindList {
        let args = self.args.clone();
        self.dropout_forward(input, &args)
    }
    fn backward(&mut self, input: &mut OptTensorKindList) -> OptTensorKindList {
        unimplemented!()
    }
}
impl FuncIntf for Dropout2d {
    fn forward(&mut self, input: &mut TensorKindList) -> TensorKindList {
        let args = self.args.clone();
        self.dropout_forward(input, &args)
    }
    fn backward(&mut self, input: &mut OptTensorKindList) -> OptTensorKindList {
        unimplemented!()
    }
}
