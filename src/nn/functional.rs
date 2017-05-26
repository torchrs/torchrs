//use torchrs::nn::functional::{max_pool2d, relu, conv2d, dropout, dropout2d, linear, log_softmax};

use autograd::variable::Variable;
use autograd::{Conv2dFArgs, ConvNdArgs, ConvNd, FuncIntf};


#[builder(pattern="owned")]
#[derive(Builder)]
pub struct MaxPoolFArgs {
    #[builder(default="1")]
    stride: u32,
    #[builder(default="0")]
    padding: u32,
    #[builder(default="1")]
    dilation: u32,
    #[builder(default="false")]
    ceil_mode: bool,
}

impl Default for MaxPoolFArgs {
    fn default() -> Self {
        MaxPoolFArgsBuilder::default().build().unwrap()
    }
}

pub fn max_pool2d<T>(input: &Variable<T>,
                     kernel_size: (u32, u32),
                     args: MaxPoolFArgs)
                     -> Variable<T> {
    input.clone()
}

#[builder(pattern="owned")]
#[derive(Builder)]
pub struct DropoutFArgs {
    #[builder(default="0.5")]
    p: f32,
    #[builder(default="false")]
    training: bool,
    #[builder(default="false")]
    inplace: bool,
}

impl Default for DropoutFArgs {
    fn default() -> Self {
        DropoutFArgsBuilder::default().build().unwrap()
    }
}

pub fn dropout<T>(input: &Variable<T>, args: DropoutFArgs) -> Variable<T> {
    input.clone()
}

pub fn dropout2d<T>(input: &Variable<T>, args: DropoutFArgs) -> Variable<T> {
    input.clone()
}

pub fn conv2d<T:Default>(input: &mut Variable<T>, weight: &mut Variable<T>, args: &mut Conv2dFArgs<T>) -> Variable<T> {
    let mut v = match args.bias {
        Some(ref mut bias) => vec![input.clone(), weight.clone(), bias.clone()],
        None => vec![input.clone(), weight.clone()],
    };
    let mut convf = ConvNd::new(&ConvNdArgs::from(args));
    convf.f(&mut v)[0].clone()
}

pub fn relu<T>(input: &Variable<T>, inplace: bool) -> Variable<T> {
    input.clone()
}

pub fn log_softmax<T>(input: &Variable<T>) -> Variable<T> {
    input.clone()
}
