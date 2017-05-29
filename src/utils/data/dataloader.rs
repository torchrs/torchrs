use utils::data::{Dataset, SamplerIntf, RandomSampler, SequentialSampler, Sampler};
use std::slice;
use tensor::Tensor;

pub type Batch<dT, tT> = (Tensor<dT>, Tensor<tT>);

pub struct DataLoader<T: Clone> {
    pub dataset: Dataset<T>,
    batch_size: usize,
    num_workers: usize,
    collate_fn: fn(batch: T) -> T,
    pin_memory: bool,
    drop_last: bool,
    sampler: Sampler<T>,
}

#[derive(Builder)]
pub struct DataLoaderArgs<T> {
    #[builder(default="1")]
    pub batch_size: usize,
    #[builder(default="0")]
    pub num_workers: usize,
    #[builder(default="default_collate")]
    pub collate_fn: fn(batch: T) -> T,
    #[builder(default="false")]
    pub pin_memory: bool,
    #[builder(default="false")]
    pub drop_last: bool,
    #[builder(default="false")]
    pub shuffle: bool,
    #[builder(default="None")]
    pub sampler: Option<Sampler<T>>,
}


fn default_collate<T>(batch: T) -> T
    where T: Clone
{
    batch.clone()
}

impl<T: Clone + 'static> DataLoader<T> {
    pub fn new(dataset: Dataset<T>, args: DataLoaderArgs<T>) -> Self {
        let sampler = match args.sampler {
            Some(sampler) => sampler,
            None => {
                if args.shuffle {
                    RandomSampler::new(dataset.clone())
                } else {
                    SequentialSampler::new(dataset.clone())
                }
            }
        };
        DataLoader {
            dataset: dataset,
            batch_size: args.batch_size,
            num_workers: args.num_workers,
            collate_fn: args.collate_fn,
            pin_memory: args.pin_memory,
            drop_last: args.drop_last,
            sampler: sampler,
        }
    }
}

pub trait DataLoaderIntf<T> {
    fn iter<'a>(&self) -> slice::Iter<'a, T>;
    fn len(&self) -> usize;
}
