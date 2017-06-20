use std::marker::PhantomData;
use utils::data::Dataset;
use std::slice;
use std::rc::Rc;

#[derive(Clone)]
pub struct Sampler {
    value: Rc<SamplerIntf>,
}
impl Sampler {
    pub fn iter(&self) -> slice::Iter<usize> {
        self.value.iter()
    }
    pub fn len(&self) -> usize {
        self.value.len()
    }
}

fn indices(len: usize) -> Vec<usize> {
    let mut v: Vec<usize> = Vec::with_capacity(len);
    v.iter_mut().enumerate().map(|(i, _)| i).collect()
}

pub trait SamplerIntf {
    fn iter<'a>(&self) -> slice::Iter<usize>;
    fn len(&self) -> usize;
}

pub struct SequentialSampler {
    indices: Vec<usize>,
}

impl SequentialSampler {
    pub fn new(len: usize) -> Sampler {
        Sampler { value: Rc::new(SequentialSampler { indices: indices(len) }) }
    }
}

impl SamplerIntf for SequentialSampler {
    fn iter(&self) -> slice::Iter<usize> {
        unimplemented!()
    }

    fn len(&self) -> usize {
        unimplemented!()
    }
}

pub struct RandomSampler {
    indices: Vec<usize>,
}

impl SamplerIntf for RandomSampler {
    fn iter(&self) -> slice::Iter<usize> {
        self.indices.iter()
    }

    fn len(&self) -> usize {
        unimplemented!()
    }
}

fn randidx(len: usize) -> usize {
    ::rand::random::<usize>() % len
}

impl RandomSampler {
    pub fn new(len: usize) -> Sampler {
        let mut indices = indices(len);
        for _ in 0..len {
            let (a, b) = (randidx(len), randidx(len));
            let tmp = indices[a];
            indices[a] = indices[b];
            indices[b] = tmp;
        }

        Sampler { value: Rc::new(RandomSampler { indices: indices }) }
    }
}

/*
pub struct WeightedRandomSampler<'a, T: Clone + 'a, R:Clone + 'a> {
    data_source: Dataset<'a, T, R>,
}
*/
