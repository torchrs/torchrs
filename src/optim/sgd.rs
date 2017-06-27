use optim::*;
use std::ops::Neg;
use num;

pub struct SGD {
    optimizer: Optimizer,
}

impl SGD {
    pub fn new(defaults: HashMap<&'static str, OptimVal>) -> Self {
        let mut sgd_defaults = map_opt!{"lr" => OptimVal::Required,
            "momentum" => 0.0, "dampening" => 0.0,
            "weight_decay"=> 0.0, "nesterov"=>false};

        for (ref key, ref value) in defaults {
            let cloned = value.clone();
            sgd_defaults.insert(key, cloned);
        }
        SGD { optimizer: Optimizer::new(sgd_defaults) }
    }
}

impl<T: ::tensor::NumLimits + From<OptimVal> + Neg<Output = T>> OptIntf<T> for SGD {
    fn optimizer(&mut self) -> &mut Optimizer {
        &mut self.optimizer
    }
    fn step(&mut self, model: &mut ModIntf<T>) {
        let (group, states) = (&self.optimizer.defaults, &mut self.optimizer.state);
        let weight_decay: T = group["weight_decay"].clone().into();
        let momentum: T = group["momentum"].clone().into();
        let dampening: T = group["dampening"].clone().into();
        let nesterov: bool = group["nesterov"].clone().into();
        let lr: T = group["lr"].clone().into();

        model.apply_parameters(&mut |v| {
            let mut d_p = if let Some(ref mut grad) = *v.grad() {
                grad.data().clone() as ::tensor::Tensor<T>
            } else {
                return;
            };
            if !weight_decay.is_zero() {
                d_p.addt_(weight_decay, v.data());
            }
            if !momentum.is_zero() {
                let mut state = &mut states[v.id];
                let mut buf: Tensor<T>;
                if !state.contains_key("momentum_buffer") {
                    buf = d_p.copy();
                    state.insert("momentum_buffer", buf.clone().into());
                } else {
                    buf = state["momentum_buffer"].clone().into();
                    buf.mul_(momentum).addt_(T::one() - dampening, &d_p);
                }
                if nesterov {
                    d_p = d_p.addt(momentum, &buf);
                } else {
                    d_p = buf;
                }

            }
            v.data().addt_(-lr, &d_p);
        });
    }
}
