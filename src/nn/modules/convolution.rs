use nn::{Module, InitModuleStruct, GetFieldStruct, ModDelegate, ModIntf, Parameter};
use autograd::Variable;
use nn::_functions::Conv2dFArgs;
use std::marker::PhantomData;
use nn::functional as F;
use num;

#[builder(pattern="owned")]
#[derive(Builder)]
pub struct Conv2dArgs<T: Default + Copy + num::Num> {
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

impl<T: Default + Copy + num::Num> Conv2dArgsBuilder<T> {
    pub fn done(self) -> Conv2d<T> {
        let args = self.build().unwrap();
        Conv2d::new(args)
    }
}

#[derive(ModParse)]
pub struct Conv2d<T: Default + Copy + num::Num> {
    delegate: Module<T>,
    weight: Parameter<T>,
    bias: Option<Parameter<T>>,
    #[ignore]
    args: Conv2dFArgs,
}

impl<T: Default + Copy + num::Num> Conv2d<T> {
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

impl<T: Default + Copy + num::Num> ModIntf<T> for Conv2d<T> {
    fn forward(&mut self, input: &mut Variable<T>) -> Variable<T> {
        let bias = if let Some(ref mut biasp) = self.bias {
            Some(&mut biasp.v)
        } else {
            None
        };
        F::conv2d(input, &mut self.weight.v, bias, &mut self.args)
    }
}

/*
impl<T: Default + Copy> InitModuleStruct for Conv2d<T> {
    fn init_module(mut self) -> Self {
        self.delegate()._name = stringify ! ( Conv2d ).into();
        self.delegate.add_param(stringify ! ( weight ));
        if let Some(ref mut param) = self.bias {
            self.delegate.add_param(stringify ! ( bias ));
        };;        self
    }
}
impl<T: Default + Copy> GetFieldStruct<T> for Conv2d<T> {
    fn get_param(&mut self, name: &str) -> Option<i32> {
        match name {
            stringify ! ( weight ) => Some(self.weight.v.id),
            stringify ! ( bias ) => {
                if let Some(ref mut param) = self.bias {
                    Some(param.v.id)
                } else {
                    None
                }
            }
            _ => panic ! ( "unknown Parameter {}" , name ),
        }
    }
    fn get_module(&mut self, name: &str) -> &mut ModIntf<T> {
        match name {
            _ => self,
        }
    }
}
*/
