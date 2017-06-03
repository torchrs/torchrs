use autograd::{Conv2dFArgs, ConvNdArgs, ConvNd, Dropout1d, Dropout2d, Threshold, LogSoftmax,
               NLLLoss, LinearF, Variable};

pub use autograd::{DropoutArgs, NLLLossArgs};
use nn::_functions::MaxPool2d;
pub use nn::_functions::MaxPool2dArgs;


pub fn max_pool2d<T: Copy>(input: &Variable<T>,
                           kernel_size: (u32, u32),
                           args: &MaxPool2dArgs)
                           -> Variable<T> {
    let mut pool_args = args.v.clone();
    pool_args.kernel_size = vec![kernel_size.0, kernel_size.1];
    MaxPool2d::new(&pool_args)
        .f(&mut vec![input.clone().into()])
        .remove(0)
        .into()
}

pub fn dropout<T: Copy>(input: &Variable<T>, args: &DropoutArgs) -> Variable<T> {
    if args.training == false {
        return input.clone();
    }
    Dropout1d::new(args)
        .f(&mut vec![input.clone().into()])
        .remove(0)
        .into()
}

pub fn dropout_<T: Copy>(input: &mut Variable<T>, args: &DropoutArgs) -> Variable<T> {
    if args.training == false {
        return input.clone();
    }
    Dropout1d::new(args)
        .f(&mut vec![input.clone().into()])
        .remove(0)
        .into()
}

pub fn dropout2d<T: Copy>(input: &Variable<T>, args: &DropoutArgs) -> Variable<T> {
    if args.training == false {
        return input.clone();
    }
    Dropout2d::new(args)
        .f(&mut vec![input.clone().into()])
        .remove(0)
        .into()
}

pub fn dropout2d_<T: Copy>(input: &mut Variable<T>, args: &DropoutArgs) -> Variable<T> {
    if args.training == false {
        return input.clone();
    }
    Dropout2d::new(args)
        .f(&mut vec![input.clone().into()])
        .remove(0)
        .into()
}

pub fn conv2d<T: Default + Copy>(input: &mut Variable<T>,
                                 weight: &mut Variable<T>,
                                 mut bias_: Option<&mut Variable<T>>,
                                 args: &mut Conv2dFArgs)
                                 -> Variable<T> {
    let mut v = match bias_ {
        Some(ref mut bias) => {
            vec![input.clone().into(),
                 weight.clone().into(),
                 bias.clone().into()]
        }
        None => vec![input.clone().into(), weight.clone().into()],
    };
    ConvNd::new(&ConvNdArgs::from(args))
        .f(&mut v)
        .remove(0)
        .into()
}

pub fn linear<T: Copy>(input: &Variable<T>,
                       weight: &mut Variable<T>,
                       bias_: Option<&mut Variable<T>>)
                       -> Variable<T> {
    let v = if let Some(bias) = bias_ {
        vec![input.clone(), weight.clone(), bias.clone()]
    } else {
        vec![input.clone(), weight.clone()]
    };
    let mut v = v.into_iter().map(|v| v.into()).collect();
    LinearF::new().f(&mut v).remove(0).into()
}

pub fn relu<T: Copy>(input: &Variable<T>) -> Variable<T> {
    Threshold::new(0., 0., false)
        .f(&mut vec![input.clone().into()])
        .remove(0)
        .into()
}

pub fn relu_<T: Copy>(input: &mut Variable<T>) -> Variable<T> {
    Threshold::new(0., 0., true)
        .f(&mut vec![input.clone().into()])
        .remove(0)
        .into()
}

pub fn log_softmax<T: Copy>(input: &Variable<T>) -> Variable<T> {
    LogSoftmax::new()
        .f(&mut vec![input.clone().into()])
        .remove(0)
        .into()
}

pub fn nll_loss<T: Copy>(input: &Variable<T>,
                         target: &Variable<i64>,
                         args: &NLLLossArgs)
                         -> Variable<T> {
    let mut kind_input = vec![input.clone().into(), target.clone().into()];
    NLLLoss::new(args).f(&mut kind_input).remove(0).into()

}
