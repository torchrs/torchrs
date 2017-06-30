use nn::{Module, InitModuleStruct, GetFieldStruct, ModDelegate, ModIntf, Parameter};
use nn::functional as F;
use autograd::Variable;
use std::marker::PhantomData;
use tensor::NumLimits;

#[builder(pattern="owned")]
#[derive(Builder)]
pub struct LinearArgs<T: ::tensor::NumLimits> {
    in_features: usize,
    out_features: usize,
    #[builder(default="true")]
    bias: bool,
    #[builder(default="PhantomData")]
    phantom: PhantomData<T>,
}
impl<T: ::tensor::NumLimits> LinearArgsBuilder<T> {
    pub fn done(self) -> Linear<T> {
        let args = self.build().unwrap();
        Linear::new(args)
    }
}

#[derive(ModParse)]
pub struct Linear<T: NumLimits> {
    delegate: Module<T>,
    #[ignore]
    in_features: usize,
    #[ignore]
    out_features: usize,
    weight: Parameter<T>,
    bias: Option<Parameter<T>>,
}

impl<T: NumLimits> Linear<T> {
    pub fn build(in_features: usize, out_features: usize) -> LinearArgsBuilder<T> {
        LinearArgsBuilder::default()
            .in_features(in_features)
            .out_features(out_features)
    }
    fn reset_parameters(mut self) -> Self {
        let stdv: f64 = 1. / (self.in_features as f64).sqrt();
        self.weight.v.data().uniform_((-stdv, stdv));
        if let Some(ref mut bias) = self.bias {
            bias.v.data().uniform_((-stdv, stdv));
        }
        self
    }
    pub fn new(args: LinearArgs<T>) -> Linear<T> {
        let bias = if args.bias {
            Some(Parameter::new((args.out_features)))
        } else {
            None
        };
        Linear {
                delegate: Module::new(),
                in_features: args.in_features,
                out_features: args.out_features,
                weight: Parameter::new((args.out_features, args.in_features)),
                bias: bias,
            }
            .init_module()
            .reset_parameters()
    }
}
impl_mod_delegate!(Linear);

impl<T: NumLimits> ModIntf<T> for Linear<T> {
    fn forward(&mut self, input: &mut Variable<T>) -> Variable<T> {
        let bias = if let Some(ref mut bias) = self.bias {
            Some(&mut bias.v)
        } else {
            None
        };
        F::linear(&input, &mut self.weight.v, bias)
    }
}
