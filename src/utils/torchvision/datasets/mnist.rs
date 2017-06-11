#![allow(deprecated)]
use utils::data as D;
use std::ops::Index;

type Sample = (Vec<u8>, u32);

pub struct MNIST {
    pub root: String,
    pub train: bool,
    pub transform: Option<fn(Vec<u8>)>,
}

#[builder(pattern="owned")]
#[derive(Clone, Builder)]
pub struct MNISTArgs {
    root: String,
    #[builder(default="true")]
    train: bool,
    #[builder(default="false")]
    download: bool,
    #[builder(default="None")]
    transform: Option<fn(Vec<u8>)>,
}

impl MNISTArgsBuilder {
    pub fn done(self) -> MNIST {
        let args = self.build().unwrap();
        MNIST::new(&args)
    }
}

impl MNIST {
    pub fn build(root: &str) -> MNISTArgsBuilder {
        MNISTArgsBuilder::default().root(root.into())
    }
    pub fn new(args: &MNISTArgs) -> Self {
        MNIST {
            root: args.root.clone(),
            train: args.train,
            transform: args.transform,
        }
    }
}

impl Index<usize> for MNIST {
    type Output = Sample;
    fn index(&self, idx: usize) -> &Self::Output {
        unimplemented!()
    }
}

impl D::DatasetIntf<Sample> for MNIST {
    fn len(&self) -> usize {
        if self.train { 60000 } else { 10000 }
    }
    fn iter(&self) -> Box<Iterator<Item = Sample>> {
        unimplemented!();
    }
}
