//use torchrs::nn::functional::{max_pool2d, relu, conv2d, dropout, dropout2d, linear, log_softmax};

use autograd::{Conv2dFArgs, ConvNdArgs, ConvNd, FuncIntf, FuncIntfKind, MaxPool2d, Dropout1d,
               Dropout2d, Threshold, LogSoftmax, NLLLoss, LinearF, Variable, VariableKind};

pub use autograd::{MaxPool2dArgs, DropoutArgs, NLLLossArgs};

pub fn max_pool2d<T>(input: &Variable<T>,
                     kernel_size: (u32, u32),
                     args: &MaxPool2dArgs)
                     -> Variable<T> {
    let mut pool_args = args.v.clone();
    pool_args.kernel_size = vec![kernel_size.0, kernel_size.1];
    MaxPool2d::new(&pool_args).f(&mut vec![input.clone()])[0].clone()
}

pub fn dropout<T>(input: &Variable<T>, args: &DropoutArgs) -> Variable<T> {
    if args.training == false {
        return input.clone();
    }
    Dropout1d::new(args).f(&mut vec![input.clone()])[0].clone()
}

pub fn dropout_<T>(input: &mut Variable<T>, args: &DropoutArgs) -> Variable<T> {
    if args.training == false {
        return input.clone();
    }
    Dropout1d::new(args).f(&mut vec![input.clone()])[0].clone()
}

pub fn dropout2d<T>(input: &Variable<T>, args: &DropoutArgs) -> Variable<T> {
    if args.training == false {
        return input.clone();
    }
    Dropout2d::new(args).f(&mut vec![input.clone()])[0].clone()
}

pub fn dropout2d_<T>(input: &mut Variable<T>, args: &DropoutArgs) -> Variable<T> {
    if args.training == false {
        return input.clone();
    }
    Dropout2d::new(args).f(&mut vec![input.clone()])[0].clone()
}

pub fn conv2d<T: Default>(input: &mut Variable<T>,
                          weight: &mut Variable<T>,
                          mut bias_: Option<&mut Variable<T>>,
                          args: &mut Conv2dFArgs)
                          -> Variable<T> {
    let mut v = match bias_ {
        Some(ref mut bias) => vec![input.clone(), weight.clone(), bias.clone()],
        None => vec![input.clone(), weight.clone()],
    };
    ConvNd::new(&ConvNdArgs::from(args)).f(&mut v)[0].clone()
}

pub fn linear<T>(input: &Variable<T>,
                 weight: &mut Variable<T>,
                 bias_: Option<&mut Variable<T>>)
                 -> Variable<T> {
    let mut v = if let Some(bias) = bias_ {
        vec![input.clone(), weight.clone(), bias.clone()]
    } else {
        vec![input.clone(), weight.clone()]
    };
    LinearF::new().f(&mut v)[0].clone()
}

pub fn relu<T>(input: &Variable<T>) -> Variable<T> {
    Threshold::new(0., 0., false).f(&mut vec![input.clone()])[0].clone()
}

pub fn relu_<T>(input: &mut Variable<T>) -> Variable<T> {
    Threshold::new(0., 0., true).f(&mut vec![input.clone()])[0].clone()
}

pub fn log_softmax<T>(input: &Variable<T>) -> Variable<T> {
    LogSoftmax::new().f(&mut vec![input.clone()])[0].clone()
}

pub fn nll_loss<T>(input: &Variable<T>, target: &Variable<i64>, args: &NLLLossArgs) -> Variable<T> {
    let mut kind_input = vec![VariableKind::from(input.clone()),
                              VariableKind::from(target.clone())];
    NLLLoss::new(args).fx(&mut kind_input).remove(0).into()

}
