use optim::*;
use std::ops::Neg;
use std::marker::PhantomData;
use nn::{ModRefMut, ModIntf};
use num;

pub struct SGD<'a, M: 'a, T: Copy + 'static> {
    optimizer: Optimizer<'a, M, T>,
}

impl<'a, T: Copy, M: ModIntf<T> + 'a> SGD<'a, M, T> {
    pub fn new(model: ModRefMut<'a, M>, defaults: HashMap<&'static str, OptimVal>) -> Self {
        let mut sgd_defaults = map_opt!{"lr" => OptimVal::Required, "momentum" => 0.0, "dampening" => 0.0,
                 "weight_decay"=> 0.0, "nesterov"=>false};

        for (ref key, ref value) in defaults {
            let cloned = value.clone();
            sgd_defaults.insert(key, cloned);
        }
        SGD { optimizer: Optimizer::new(model, sgd_defaults) }
    }
}

impl<'a, T : Copy + From<OptimVal> + num::Num + num::Float + Neg<Output=T>, M: ModIntf<T> + 'a> OptIntf<'a, M, T> for SGD<'a, M, T> {
    fn optimizer(&mut self) -> &mut Optimizer<'a, M, T> {
        &mut self.optimizer
    }
    fn step(&mut self) {
        let group = &self.optimizer.defaults;
        let params = (*self.optimizer.model).parameters();
        let weight_decay : T = group["weight_decay"].clone().into();
        let momentum : T = group["momentum"].clone().into();
        let dampening : T = group["dampening"].clone().into();
        let nesterov : bool = group["nesterov"].clone().into();
        let lr : T = group["lr"].clone().into();

        for mut p in params {
            let mut v = &mut p.v;
            let mut d_p = if let Some(ref mut grad) = *v.grad() {
                grad.data().clone() as ::tensor::Tensor<T>
            } else {
                continue;
            };
            if !weight_decay.is_zero() {
                d_p.addt_(weight_decay, v.data());
            }
            if !momentum.is_zero() {
                let mut state = &mut self.optimizer.state[v.id];
                let mut buf : Tensor<T>;
                if !state.contains_key("momentum_buffer") {
                    buf = d_p.copy();
                    state.insert("momentum_buffer", buf.clone().into());
                } else {
                    buf = state["momentum_buffer"].clone().into();
                    buf.mul_(momentum).addt_(T::one() - dampening,  &d_p);
                }
                if nesterov {
                    d_p = d_p.addt(momentum, &buf);
                } else {
                    d_p = buf;
                }

            }
            v.data().addt_(-lr, &d_p);
        }
    }
}
