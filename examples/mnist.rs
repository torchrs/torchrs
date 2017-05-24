extern crate torchrs;

use torchrs::nn::{Module, Conv2d, Linear};


use torchrs::nn::functional::{max_pool2d, relu, dropout, dropout2d, log_softmax};
#[derive(Serialize, Deserialize, Debug, ModuleParse)]
struct Net<'a> {
    delegate: nn::Module<'a>,
    conv1: Conv2d<'a>,
    conv2: Conv2d<'a>,
    fc1: Linear<'a>,
    fc2: Linear<'a>,
}

impl Net<'a> {
    pub fn new() -> Net<'a> {
        let t = Net {
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
impl <'a>ModIntf<'a> for Net<'a> {
    fn forward(&mut self, args: &[&mut Tensor]) -> [&mut Tensor] {
        let training = self.delegate.training;
        let x = relu(max_pool2d(self.conv1(&args[0]), 2), false);
        let x = relu(max_pool2d(dropout2d(self.conv2(&x), training, 0.5), 2), false);
        let x = x.view(-1, 320);
        let x = relu(self.fc1(&x), false);
        let x = dropout(&x, training, 0.5);
        let x = self.fc2(&x);
        [log_softmax(&x)]
    }
 }

fn main () {

}