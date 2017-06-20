use std::rc::Rc;
use std::marker::PhantomData;

#[derive(Clone)]
pub struct Dataset<T: Clone + 'static> {
    value: Rc<DatasetIntf<Batch = T>>,
}
impl<T: Clone + 'static> Dataset<T> {
    pub fn new(arg: Rc<DatasetIntf<Batch = T>>) -> Self {
        Dataset { value: arg }
    }
    pub fn len(&self) -> usize {
        self.value.len()
    }
}

pub trait DatasetIntf {
    type Batch: Clone;
    fn len(&self) -> usize;
    fn collate(&self, sample: Vec<usize>) -> Self::Batch;
}
