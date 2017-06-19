use optim::*;
use num;

pub struct SGD<'a, T: Copy + 'a> {
    optimizer: Optimizer<'a, T>,
}

impl<'a, T: Copy + 'a> SGD<'a, T> {
    pub fn new(params: ParamIter<'a, T>, defaults: HashMap<&'static str, OptimOpts>) -> Self {
        let mut sgd_defaults = map_opt!{"lr" => OptimOpts::Required, "momentum" => 0.0, "dampening" => 0.0,
                 "weight_decay"=> 0.0, "nesterov"=>false};

        for (ref key, ref value) in defaults {
            let cloned = value.clone();
            sgd_defaults.insert(key, cloned);
        }
        SGD { optimizer: Optimizer::new(params, sgd_defaults) }
    }
}


impl<'a, T : 'a + Copy + From<OptimOpts> + num::Zero + PartialEq> OptIntf<'a, T> for SGD<'a, T> {
    fn optimizer(&mut self) -> &mut Optimizer<'a, T> {
        &mut self.optimizer
    }
    fn step(&mut self) {
        let group = &self.optimizer.defaults;
        let params = self.optimizer.params.clone();
        let weight_decay : T = group["weight_decay"].clone().into();
        let momentum : T = group["momentum"].clone().into();
        let dampening : T = group["dampening"].clone().into();
        let nesterov : bool = group["nesterov"].clone().into();

        for p in params {
            let mut v = &mut p.v;
            let mut d_p = if let Some(ref mut grad) = *v.grad() {
                grad.data().clone() as ::tensor::Tensor<T>
            } else {
                continue;
            };
            if !weight_decay.is_zero() {
                d_p.addt_(weight_decay, v.data());
            }
        }
    }
}
