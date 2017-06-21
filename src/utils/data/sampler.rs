use std::rc::Rc;

#[derive(Clone)]
pub struct Sampler {
    value: Rc<SamplerIntf>,
    batch_size: usize,
}

impl Sampler {
    pub fn data(&self) -> Vec<Vec<usize>> {
        self.value
            .data()
            .chunks(self.batch_size)
            .map(|v| v.to_vec())
            .collect()
    }
}

fn indices(len: usize) -> Vec<usize> {
    let mut v: Vec<usize> = Vec::with_capacity(len);
    v.iter_mut().enumerate().map(|(i, _)| i).collect()
}

pub trait SamplerIntf {
    fn data(&self) -> Vec<usize>;
}

pub struct SequentialSampler {
    indices: Vec<usize>,
}

impl SequentialSampler {
    pub fn new(len: usize, batch_size: usize) -> Sampler {
        Sampler {
            value: Rc::new(SequentialSampler { indices: indices(len) }),
            batch_size: batch_size,
        }
    }
}

impl SamplerIntf for SequentialSampler {
    fn data(&self) -> Vec<usize> {
        self.indices.clone()
    }
}

pub struct RandomSampler {
    indices: Vec<usize>,
}

impl SamplerIntf for RandomSampler {
    fn data(&self) -> Vec<usize> {
        self.randomize()
    }
}

impl RandomSampler {
    fn randidx(len: usize) -> usize {
        ::rand::random::<usize>() % len
    }
    fn randomize(&self) -> Vec<usize> {
        let (len, mut indices) = (self.indices.len(), self.indices.clone());
        for _ in 0..len {
            let (a, b) = (Self::randidx(len), Self::randidx(len));
            let tmp = indices[a];
            indices[a] = indices[b];
            indices[b] = tmp;
        }
        indices
    }

    pub fn new(len: usize, batch_size: usize) -> Sampler {
        Sampler {
            value: Rc::new(RandomSampler { indices: indices(len) }),
            batch_size: batch_size,
        }
    }
}

/*
pub struct WeightedRandomSampler<'a, T: Clone + 'a, R:Clone + 'a> {
    data_source: Dataset<'a, T, R>,
}
*/
