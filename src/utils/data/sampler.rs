use utils::data::Dataset;
use std::slice;
use std::rc::Rc;

#[derive(Clone)]
pub struct Sampler<T> {
    value: Rc<SamplerIntf<T>>,
}
impl<T> Sampler<T> {
    pub fn iter<'a>(&self) -> slice::Iter<'a, T> {
        self.value.iter()
    }
    pub fn len(&self) -> usize {
        self.value.len()
    }
}

pub trait SamplerIntf<T> {
    fn iter<'a>(&self) -> slice::Iter<'a, T>;
    fn len(&self) -> usize;
}

pub struct SequentialSampler<T: Clone> {
    data_source: Dataset<T>,
}

impl<T: Clone + 'static> SequentialSampler<T> {
    pub fn new<'a>(dataset: Dataset<T>) -> Sampler<T> {
        Sampler { value: Rc::new(SequentialSampler { data_source: dataset }) }
    }
}

impl<T: Clone> SamplerIntf<T> for SequentialSampler<T> {
    fn iter<'a>(&self) -> slice::Iter<'a, T> {
        panic!("implement")
    }

    fn len(&self) -> usize {
        panic!("implement")
    }
}

pub struct RandomSampler<T: Clone> {
    data_source: Dataset<T>,
}

impl<T: Clone> SamplerIntf<T> for RandomSampler<T> {
    fn iter<'a>(&self) -> slice::Iter<'a, T> {
        panic!("implement")
    }

    fn len(&self) -> usize {
        panic!("implement")
    }
}

impl<T: Clone + 'static> RandomSampler<T> {
    pub fn new(dataset: Dataset<T>) -> Sampler<T> {
        Sampler { value: Rc::new(RandomSampler { data_source: dataset.clone() }) }
    }
}

pub struct WeightedRandomSampler<T: Clone> {
    data_source: Dataset<T>,
}
