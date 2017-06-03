use nn::{Module, ModuleStruct, ModDelegate, ModIntf, Parameter};
use autograd::Variable;
use nn::_functions::Conv2dFArgs;
use std::marker::PhantomData;
use nn::functional as F;

#[builder(pattern="owned")]
#[derive(Builder)]
pub struct Conv2dArgs<T: Default + Copy> {
    in_features: usize,
    out_features: usize,
    kernel_size: (usize, usize),
    #[builder(default="vec![1, 1]")]
    pub stride: Vec<u32>,
    #[builder(default="vec![0, 0]")]
    pub padding: Vec<u32>,
    #[builder(default="vec![1, 1]")]
    pub dilation: Vec<u32>,
    #[builder(default="1")]
    groups: u32,
    #[builder(default="true")]
    bias: bool,
    #[builder(default="PhantomData")]
    phantom: PhantomData<T>,
}

impl<T: Default + Copy> Conv2dArgsBuilder<T> {
    pub fn done(self) -> Conv2d<T> {
        let args = self.build().unwrap();
        Conv2d::new(args)
    }
}

#[derive(ModParse)]
pub struct Conv2d<T: Default + Copy> {
    delegate: Module<T>,
    weight: Parameter<T>,
    bias: Option<Parameter<T>>,
    #[ignore]
    args: Conv2dFArgs,
}

impl<T: Default + Copy> Conv2d<T> {
    pub fn build(in_features: usize,
                 out_features: usize,
                 kernel_size: (usize, usize))
                 -> Conv2dArgsBuilder<T> {
        Conv2dArgsBuilder::default()
            .in_features(in_features)
            .out_features(out_features)
            .kernel_size(kernel_size)
    }
    pub fn new(args: Conv2dArgs<T>) -> Conv2d<T> {
        let bias = if args.bias {
            Some(Parameter::new(vec![args.out_features]))
        } else {
            None
        };
        let fargs = Conv2dFArgs {
            kernel_size: vec![args.kernel_size.0, args.kernel_size.1],
            stride: args.stride.clone(),
            padding: args.padding.clone(),
            dilation: args.dilation.clone(),
            groups: args.groups,
        };
        Conv2d {
                delegate: Module::new(),
                weight: Parameter::new(vec![args.out_features, args.in_features]),
                bias: bias,
                args: fargs,
            }
            .init_module()
    }
}
impl_mod_delegate!(Conv2d);

impl<T: Default + Copy> ModIntf<T> for Conv2d<T> {
    fn forward(&mut self, input: &mut Variable<T>) -> Variable<T> {
        let bias = if let Some(ref mut biasp) = self.bias {
            Some(&mut biasp.v)
        } else {
            None
        };
        F::conv2d(input, &mut self.weight.v, bias, &mut self.args)
    }
}
