#[macro_use]
extern crate modparse_derive;
#[macro_use]
extern crate derive_builder;
#[macro_use]
extern crate torchrs;
#[macro_use]
extern crate clap;

use torchrs::autograd::{Variable, VariableArgs, VarAccess};
use torchrs::optim;
use torchrs::optim::OptimVal;
use torchrs::nn;
use torchrs::nn::{InitModuleStruct, GetFieldStruct, ModIntf, ModDelegate, Module};
use torchrs::nn::functional as F;
use torchrs::utils::data as D;
use torchrs::utils::torchvision::{datasets, transforms};
use torchrs::tensor;

use clap::{Arg, App};
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
    #[builder(default="false")]
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
    let mut args = NetArgs::default();
    let matches = App::new("Torch.rs MNIST Example")
        .version("0.1")
        .author("Some Guy")
        .about("recognize handwritten digits")
        .arg(Arg::with_name("batch-size").takes_value(true))
        .arg(Arg::with_name("test-batch-size").takes_value(true))
        .arg(Arg::with_name("epochs").takes_value(true))
        .arg(Arg::with_name("lr").takes_value(true))
        .arg(Arg::with_name("momentum").takes_value(true))
        .arg(Arg::with_name("no-cuda").takes_value(false))
        .arg(Arg::with_name("seed").takes_value(true))
        .arg(Arg::with_name("log-interval").takes_value(true))
        .get_matches();
    let (b_size, tb_size, epochs, lr, momentum, seed, log_int) =
        (value_t!(matches.value_of("batch-size"), usize).ok(),
         value_t!(matches.value_of("test-batch-size"), usize).ok(),
         value_t!(matches.value_of("epochs"), u32).ok(),
         value_t!(matches.value_of("lr"), f32).ok(),
         value_t!(matches.value_of("momentum"), f32).ok(),
         value_t!(matches.value_of("seed"), usize).ok(),
         value_t!(matches.value_of("log-interval"), usize).ok());
    if let Some(b_size) = b_size {
        args.batch_size = b_size
    }
    if let Some(tb_size) = tb_size {
        args.test_batch_size = tb_size
    }
    if let Some(epochs) = epochs {
        args.epochs = epochs
    }
    if let Some(lr) = lr {
        args.lr = lr
    }
    if let Some(momentum) = momentum {
        args.momentum = momentum
    }
    if let Some(seed) = seed {
        args.seed = seed
    }
    if let Some(log_int) = log_int {
        args.log_interval = log_int
    }
    //XXX CUDA?
    args
}

#[derive(ModParse)]
struct Net<T: tensor::NumLimits> {
    delegate: nn::Module<T>,
    conv1: nn::Conv2d<T>,
    conv2: nn::Conv2d<T>,
    fc1: nn::Linear<T>,
    fc2: nn::Linear<T>,
}

impl<T: tensor::NumLimits> Net<T> {
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

impl<T: tensor::NumLimits> ModIntf<T> for Net<T> {
    fn forward(&mut self, args: &mut Variable<T>) -> Variable<T> {
        let pool_val = F::MaxPool2dArgs::default();
        let mut dropout_val = F::DropoutArgs::default();
        dropout_val.training = self.delegate.training;
        let mut x = F::relu(F::max_pool2d(self.conv1.f(args.clone()), (2, 2), &pool_val));
        x = F::relu(F::max_pool2d(F::dropout2d(self.conv2.f(x), &dropout_val),
                                  (2, 2),
                                  &pool_val));
        x = x.view([-1, 320]);
        x = F::relu(self.fc1.f(x));
        x = F::dropout(x, &dropout_val);
        x = self.fc2.f(x);
        F::log_softmax(x)
    }
}

fn train(model: &mut Net<f32>,
         args: &NetArgs,
         train_loader: &D::BatchLoader<f32, i64>,
         epoch: u32,
         optimizer: &mut optim::OptIntf<f32>) {
    model.train();
    for (batch_idx, (ref data, ref target)) in train_loader.iter().enumerate() {
        let (mut data, target) = if args.cuda {
            (Variable::new(data.cuda(None).clone()), Variable::new(target.cuda(None).clone()))
        } else {
            (Variable::new(data.clone()), Variable::new(target.clone()))
        };
        optimizer.zero_grad(model);
        let output = model.f(data.clone());
        let mut loss = F::nll_loss(output, target, &F::NLLLossArgs::default());
        loss.backward();
        optimizer.step(model);
        model.free_graph();
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
        let (data, mut target) = if args.cuda {
            (Variable::new_args(data.cuda(None).clone(), &varargs),
             Variable::new(target.cuda(None).clone()))
        } else {
            (Variable::new_args(data.clone(), &varargs), Variable::new(target.clone()))
        };
        let mut output = model.f(data);
        test_loss += F::nll_loss(output.clone(), target.clone(), &F::NLLLossArgs::default());
        let pred = output.data().max_reduce(1).1;
        correct += pred.eq_tensor(&*target.data()).cpu().sum::<u32>();
    }
    test_loss /= test_loader.len() as f32;
    println!("\nTest set: Average loss: {:.4}, Accuracy: {}/{} ({:.0}%)\n",
             test_loss,
             correct,
             test_loader.dataset.len(),
             100. * (correct as f32) / (test_loader.dataset.len() as f32))

}

fn main() {
    let args = parse_args();
    let train_loader: D::BatchLoader<f32, i64> = D::DataLoader::build()
        .batch_size(args.batch_size)
        .done(datasets::MNIST::<f32>::build("../data")
                  .download(false)
                  .done(None));

    let test_loader: D::BatchLoader<f32, i64> = D::DataLoader::build()
        .batch_size(args.batch_size)
        .done(datasets::MNIST::<f32>::build("../data")
                  .train(false)
                  .done(None));
    let mut model = Net::new();
    let mut optimizer = optim::SGD::new(map_opt!{"lr" => args.lr, "momentum" => args.momentum});
    for epoch in 1..args.epochs + 1 {
        train(&mut model, &args, &train_loader, epoch, &mut optimizer);
        test(&mut model, &args, &test_loader);
    }
}
