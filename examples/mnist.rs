extern crate getopts;
#[macro_use]
extern crate modparse_derive;
#[macro_use]
extern crate derive_builder;

#[macro_use]
extern crate torchrs;


use torchrs::autograd::{Variable, VariableArgs, VarAccess};
use torchrs::tensor::Tensor;
use torchrs::optim;

use torchrs::macros::*;
use torchrs::nn;
use torchrs::nn::{ModuleStruct, ModIntf, ModDelegate, Module};
use torchrs::nn::functional as F;
use torchrs::utils::data as D;

use getopts::Options;
use std::env;

fn MNIST() -> D::Dataset<D::Batch<f32, i64>> {
    unimplemented!()
}

#[derive(Clone, Builder)]
struct NetArgs {
    #[builder(default="64")]
    batch_size: usize,
    #[builder(default="1000")]
    test_batch_size: usize,
    #[builder(default="10")]
    epochs: u32,
    #[builder(default="0.01")]
    lr: f32,
    #[builder(default="0.5")]
    momentum: f32,
    #[builder(default="true")]
    cuda: bool,
    #[builder(default="1")]
    seed: usize,
    #[builder(default="10")]
    log_interval: usize,
}

impl Default for NetArgs {
    fn default() -> Self {
        NetArgsBuilder::default().build().unwrap()
    }
}

fn parse_args() -> NetArgs {
    let cmd_args: Vec<String> = env::args().collect();
    let mut args = NetArgs::default();
    let mut opts = Options::new();
    unimplemented!();
    /*
    	* XXX do parsey stuff
    	*/
    args
}

#[derive(ModParse)]
struct Net<T: Default + Copy> {
    delegate: nn::Module<T>,
    conv1: nn::Conv2d<T>,
    conv2: nn::Conv2d<T>,
    fc1: nn::Linear<T>,
    fc2: nn::Linear<T>,
}

impl<T: Default + Copy> Net<T> {
    pub fn new() -> Net<T> {
        Net {
                delegate: nn::Module::new(),
                conv1: nn::Conv2d::build(1, 10, (5, 5)).done(),
                conv2: nn::Conv2d::build(10, 20, (5, 5)).done(),
                fc1: nn::Linear::build(320, 50).done(),
                fc2: nn::Linear::build(50, 10).done(),
            }
            .init_module()
    }
}
impl_mod_delegate!(Net);

// The forward operations could take on one of two implementations.
// The first supporting a near verbatim version of the python
// implementation, and the second supporting a slightly more
// idiomatic to Rust method chaining.

// a) as a near verbatim implementation of the python version
impl ModIntf<f32> for Net<f32> {
    fn forward(&mut self, args: &mut Variable<f32>) -> Variable<f32> {
        let training = self.delegate.training;
        let pool_val = F::MaxPool2dArgs::default();
        let dropout_val = F::DropoutArgs::default();
        let mut x = F::relu(&F::max_pool2d(&self.conv1.f(args), (2, 2), &pool_val));
        let mut x = F::relu(&F::max_pool2d(&F::dropout2d(&self.conv2.f(&mut x), &dropout_val),
                                           (2, 2),
                                           &pool_val));
        let mut x = x.view(&[-1, 320]);
        let x = F::relu(&self.fc1.f(&mut x));
        let mut x = F::dropout(&x, &dropout_val);
        let x = self.fc2.f(&mut x);
        F::log_softmax(&x)
    }
}

fn train(model: &mut Net<f32>,
         args: &NetArgs,
         train_loader: &D::BatchLoader<f32, i64>,
         epoch: u32,
         optimizer: &mut optim::OptIntf) {
    model.train();
    for (batch_idx, (ref data, ref target)) in train_loader.iter().enumerate() {
        let (mut data, target) = if args.cuda {
            (Variable::new(data.cuda(None).clone()), Variable::new(target.cuda(None).clone()))
        } else {
            (Variable::new(data.clone()), Variable::new(target.clone()))
        };
        optimizer.zero_grad();
        let output = model.f(&mut data);
        let mut loss = F::nll_loss(&output, &target, &F::NLLLossArgs::default());
        loss.backward();
        optimizer.step();
        if batch_idx % args.log_interval == 0 {
            println!("Train Epoch: {} [{}/{} ({:.0}%)]\tLoss: {:.6}",
                     epoch,
                     batch_idx * data.data().len(),
                     train_loader.dataset.len(),
                     100. * (batch_idx as f32) / train_loader.len() as f32,
                     loss.data()[0]);
        }
    }
}

fn test(model: &mut Net<f32>, args: &NetArgs, test_loader: &D::BatchLoader<f32, i64>) {
    model.eval();
    let mut test_loss = 0.;
    let mut correct = 0;
    for (ref data, ref target) in test_loader.iter() {
        let varargs = VariableArgs {
            volatile: true,
            ..Default::default()
        };
        let (mut data, mut target) = if args.cuda {
            (Variable::new_args(data.cuda(None).clone(), &varargs),
             Variable::new(target.cuda(None).clone()))
        } else {
            (Variable::new_args(data.clone(), &varargs), Variable::new(target.clone()))
        };
        let mut output = model.f(&mut data);
        test_loss += F::nll_loss(&output, &target, &F::NLLLossArgs::default());
        let pred = output.data().max_reduce(1).1;
        correct += pred.eq_tensor(&*target.data()).cpu().sum() as u32;
    }
    test_loss /= test_loader.len() as f32;
    println!("\nTest set: Average loss: {:.4}, Accuracy: {}/{} ({:.0}%)\n",
             test_loss,
             correct,
             test_loader.dataset.len(),
             100. * (correct as f32) / (test_loader.dataset.len() as f32))

}

fn main() {
    // no data loaders yet so demonstrate with Vec placeholder
    let train_loader: D::BatchLoader<f32, i64> = D::DataLoader::new(MNIST(),
                                                                    D::DataLoaderArgs::default());
    let test_loader: D::BatchLoader<f32, i64> = D::DataLoader::new(MNIST(),
                                                                   D::DataLoaderArgs::default());
    let args = parse_args();
    let mut optimizer = optim::SGD::new();

    let mut model = Net::new();
    for epoch in 1..args.epochs + 1 {
        train(&mut model, &args, &train_loader, epoch, &mut optimizer);
        test(&mut model, &args, &test_loader);
    }
}
