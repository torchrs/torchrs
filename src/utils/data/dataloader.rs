use utils::data::{Dataset, SamplerIntf, RandomSampler, SequentialSampler, Sampler};
use std::slice;
use tensor::Tensor;
use torch;
use ::*;

pub type Batch<dT, tT> = (Tensor<dT>, Tensor<tT>);
pub type BatchLoader<dT: 'static, tT: 'static> = DataLoader<Batch<dT, tT>>;


pub struct DataLoader<T: Clone + 'static> {
    pub dataset: Dataset<T>,
    batch_size: usize,
    num_workers: usize,
    pin_memory: bool,
    drop_last: bool,
    sampler: Sampler,
}

#[derive(Builder)]
pub struct DataLoaderArgs {
    #[builder(default="1")]
    pub batch_size: usize,
    #[builder(default="0")]
    pub num_workers: usize,
    #[builder(default="false")]
    pub pin_memory: bool,
    #[builder(default="false")]
    pub drop_last: bool,
    #[builder(default="false")]
    pub shuffle: bool,
    #[builder(default="None")]
    pub sampler: Option<Sampler>,
}

impl Default for DataLoaderArgs {
    fn default() -> Self {
        DataLoaderArgsBuilder::default().build().unwrap()
    }
}

impl<T: Clone + 'static> DataLoader<T> {
    pub fn new(dataset: Dataset<T>, args: DataLoaderArgs) -> Self {
        let sampler = match args.sampler {
            Some(sampler) => sampler,
            None => {
                if args.shuffle {
                    RandomSampler::new(dataset.len())
                } else {
                    SequentialSampler::new(dataset.len())
                }
            }
        };
        DataLoader {
            dataset: dataset,
            batch_size: args.batch_size,
            num_workers: args.num_workers,
            pin_memory: args.pin_memory,
            drop_last: args.drop_last,
            sampler: sampler,
        }
    }
    pub fn iter(&self) -> Box<Iterator<Item = T>> {
        unimplemented!();
        //self.dataset.iter()
    }
    pub fn len(&self) -> usize {
        self.dataset.len()
    }
}
