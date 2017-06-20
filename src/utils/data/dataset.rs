use std::rc::Rc;
use std::marker::PhantomData;

#[derive(Clone)]
pub struct Dataset<T: Clone + 'static, R: Clone + 'static> {
    value: Rc<DatasetIntf<Sample = T, Batch = R>>,
}
impl<T: Clone + 'static, R: Clone + 'static> Dataset<T, R> {
    pub fn new(arg: Rc<DatasetIntf<Sample = T, Batch = R>>) -> Self {
        Dataset { value: arg }
    }
    pub fn len(&self) -> usize {
        self.value.len()
    }
    pub fn iter(&self) -> Box<Iterator<Item = T>> {
        Box::new(self.value.iter())
    }
}

pub trait DatasetIntf {
    type Batch: Clone;
    type Sample: Clone;
    fn len(&self) -> usize;
    fn iter(&self) -> Box<Iterator<Item = Self::Sample>>;
    fn index(&mut self, idx: usize) -> Self::Sample;
    fn collate(&self, sample: Vec<&Self::Sample>) -> Self::Batch;
}
