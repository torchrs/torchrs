extern crate torchrs;
#[macro_use]
extern crate modparse_derive;

use torchrs::autograd::{Variable, BackwardArgs};
use torchrs::tensor::Tensor;
use torchrs::optim;

use torchrs::nn;
use torchrs::nn::{ModuleStruct, ModIntf};
use torchrs::nn::functional::{dropout2d, MaxPool2dArgs, DropoutArgs};
use torchrs::nn::functional as F;
use std::vec;
use std::slice;



type Batch<dT, tT> = (Tensor<dT>, Tensor<tT>);
type Dataset<dT, tT> = Vec<Batch<dT, tT>>;
struct DataLoader<dT, tT> {
    dataset: Dataset<dT, tT>,
}

impl<dT, tT> DataLoader<dT, tT> {
    pub fn new() -> Self {
        DataLoader { dataset: Vec::new() }
    }
    pub fn iter(&self) -> slice::Iter<Batch<dT, tT>> {
        self.dataset.as_slice().iter()
    }
    pub fn iter_mut(&mut self) -> slice::IterMut<Batch<dT, tT>> {
        self.dataset.as_mut_slice().iter_mut()
    }
    pub fn len(&self) -> usize {
        self.dataset.len()
    }
}

#[derive(Default)]
struct NetArgs {
    log_interval: usize,
    cuda: bool,
}

#[derive(ModParse)]
struct Net {
    delegate: nn::Module<f32>,
    conv1: nn::Conv2d<f32>,
    conv2: nn::Conv2d<f32>,
    fc1: nn::Linear<f32>,
    fc2: nn::Linear<f32>,
}

impl Net {
    pub fn new() -> Net {
        let mut t = Net {
            delegate: nn::Module::new(),
            conv1: nn::Conv2d::build(1, 10, 5).done(),
            conv2: nn::Conv2d::build(10, 20, 5).done(),
            fc1: nn::Linear::build(320, 50).done(),
            fc2: nn::Linear::build(50, 10).done(),
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
        let mut x = F::relu(&F::max_pool2d(&self.conv1.f(args), (2, 2), &pool_val));
        let mut x = F::relu(&F::max_pool2d(&dropout2d(&self.conv2.f(&mut x), &dropout_val),
                                           (2, 2),
                                           &pool_val));
        let mut x = x.view(&[-1, 320]);
        let x = F::relu(&self.fc1.f(&mut x));
        let mut x = F::dropout(&x, &dropout_val);
        let x = self.fc2.f(&mut x);
        F::log_softmax(&x)
    }
    fn forwardv(&mut self, input: &mut Vec<Variable<f32>>) -> Vec<Variable<f32>> {
        panic!("not valid")
    }

    fn delegate(&mut self) -> &mut nn::Module<f32> {
        &mut self.delegate
    }
}

fn train(model: &mut Net,
         args: &NetArgs,
         train_loader: &DataLoader<f32, i64>,
         epoch: u32,
         optimizer: &mut optim::OptIntf) {
    model.train();
    for (batch_idx, &(ref data, ref target)) in train_loader.iter().enumerate() {
        let (mut data, target) = if args.cuda {
            (Variable::new(data.cuda().clone()), Variable::new(target.cuda().clone()))
        } else {
            (Variable::new(data.clone()), Variable::new(target.clone()))
        };
        optimizer.zero_grad();
        let output = model.f(&mut data);
        let mut loss = F::nll_loss(&output, &target, &F::NLLLossArgs::default());
        loss.backward(&BackwardArgs::default());
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

fn test(model: &mut Net, args: &NetArgs, data: &DataLoader<f32, i64>, epoch: u32) {
    model.eval()
}


fn main() {
    let epochs = 10;
    // no data loaders yet so demonstrate with Vec placeholder
    let train_loader: DataLoader<f32, i64> = DataLoader::new();
    let test_loader: DataLoader<f32, i64> = DataLoader::new();
    let args = NetArgs::default();
    let mut optimizer = optim::SGD::new();

    let mut model = Net::new();
    for epoch in 1..epochs + 1 {
        train(&mut model, &args, &train_loader, epoch, &mut optimizer);
        test(&mut model, &args, &test_loader, epoch);
    }
}
