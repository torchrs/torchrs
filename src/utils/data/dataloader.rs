use utils::data::{DatasetIntfRef, RandomSampler, SequentialSampler, Sampler};
use std::vec;
use tensor::Tensor;

pub type Batch<Dt, Tt> = (Tensor<Dt>, Tensor<Tt>);
pub type BatchLoader<Dt, Tt> = DataLoader<Batch<Dt, Tt>>;


pub struct DataLoader<T: Clone> {
    pub dataset: DatasetIntfRef<T>,
    pub batch_size: usize,
    pub num_workers: usize,
    pub pin_memory: bool,
    pub drop_last: bool,
    pub sampler: Sampler,
}

#[derive(Builder)]
#[builder(pattern="owned")]
pub struct DataLoaderArgs<T> {
    #[builder(default="1")]
    pub batch_size: usize,
    #[builder(default="0")]
    pub num_workers: usize,
    #[builder(default="false")]
    pub pin_memory: bool,
    #[builder(default="false")]
    pub drop_last: bool,
    #[builder(default="false")]
    shuffle: bool,
    #[builder(default="None")]
    pub sampler: Option<Sampler>,
    #[builder(default="::std::marker::PhantomData")]
    phantom: ::std::marker::PhantomData<T>,
}

impl<T: Clone + Default> Default for DataLoaderArgs<T> {
    fn default() -> Self {
        DataLoaderArgsBuilder::default().build().unwrap()
    }
}

pub struct DataLoaderIter<T: Clone> {
    dataset: DatasetIntfRef<T>,
    batch_size: usize,
    chunk_iter: vec::IntoIter<Vec<usize>>,
}

impl<T: Clone> DataLoaderIter<T> {
    pub fn new(loader: &DataLoader<T>) -> Self {
        DataLoaderIter {
            dataset: loader.dataset.clone(),
            batch_size: loader.batch_size,
            chunk_iter: loader.sampler.data().into_iter(),
        }
    }
}

impl<T: Clone> Iterator for DataLoaderIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        match self.chunk_iter.next() {
            Some(ref v) if v.len() == self.batch_size => Some(self.dataset.collate(v.clone())),
            _ => None,
        }
    }
}

impl<T: Default + Clone + 'static> DataLoaderArgsBuilder<T> {
    pub fn done(self, dataset: DatasetIntfRef<T>) -> DataLoader<T> {
        let args = self.build().unwrap();
        DataLoader::new(dataset, args)
    }
}

impl<T: Clone + 'static + Default> DataLoader<T> {
    pub fn build() -> DataLoaderArgsBuilder<T> {
        DataLoaderArgsBuilder::default()
    }
    pub fn new(dataset: DatasetIntfRef<T>, args: DataLoaderArgs<T>) -> Self {
        let sampler = match args.sampler {
            Some(sampler) => sampler,
            None => {
                if args.shuffle {
                    RandomSampler::new(dataset.len(), args.batch_size)
                } else {
                    SequentialSampler::new(dataset.len(), args.batch_size)
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
        Box::new(DataLoaderIter::new(self))
    }
    pub fn len(&self) -> usize {
        self.dataset.len()
    }
}
