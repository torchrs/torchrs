extern crate torchrs;
#[macro_use]
extern crate modparse_derive;


use torchrs::nn::{Module, Conv2d, Linear};
use torchrs::nn::modules::module::*;
use torchrs::autograd::variable::Variable;
use torchrs::tensor::Tensor;

use torchrs::nn::functional::{max_pool2d, relu, dropout, dropout2d, log_softmax, MaxPool2dArgs,
                              DropoutArgs};
#[derive(ModParse)]
struct Net {
    delegate: Module<f32>,
    conv1: Conv2d<f32>,
    conv2: Conv2d<f32>,
    fc1: Linear<f32>,
    fc2: Linear<f32>,
}

impl Net {
    pub fn new() -> Net {
        let mut t = Net {
            delegate: Module::new(),
            conv1: Conv2d::build(1, 10, 5).done(),
            conv2: Conv2d::build(10, 20, 5).done(),
            fc1: Linear::build(320, 50).done(),
            fc2: Linear::build(50, 10).done(),
        };
        t.init_module();
        t
    }
}
// The forward operations could take on one of two implementations.
// The first supporting a near verbatim version of the python
// implementation, and the second supporting a slightly more
// idiomatic to Rust method chaining.

// a) as a near verbatim implementation of the python version
impl ModIntf<f32> for Net {
    fn forward(&mut self, args: &mut Variable<f32>) -> Variable<f32> {
        let training = self.delegate.training;
        let pool_val = MaxPool2dArgs::default();
        let dropout_val = DropoutArgs::default();
        let mut x = relu(&max_pool2d(&self.conv1.f(args), (2, 2), &pool_val));
        let mut x = relu(&max_pool2d(&dropout2d(&self.conv2.f(&mut x), &dropout_val),
                                     (2, 2),
                                     &pool_val));
        let mut x = x.view(&[-1, 320]);
        let x = relu(&self.fc1.f(&mut x));
        let mut x = dropout(&x, &dropout_val);
        let x = self.fc2.f(&mut x);
        log_softmax(&x)
    }
    fn forwardv(&mut self, input: &mut Vec<Variable<f32>>) -> Vec<Variable<f32>> {
        panic!("not valid")
    }

    fn delegate(&mut self) -> &mut Module<f32> {
        &mut self.delegate
    }
}

fn train(model: &mut Net, epoch: u32) {
    model.train()
}

fn test(model: &mut Net, epoch: u32) {
    model.eval()
}


fn main() {
    let mut model = Net::new();
    train(&mut model, 10);
    test(&mut model, 10)
}
