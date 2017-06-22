use std::rc::Rc;

#[derive(Clone)]
pub struct Dataset<T: Clone> {
    value: Rc<DatasetIntf<Batch = T>>,
}
impl<T: Clone> Dataset<T> {
    pub fn new(arg: Rc<DatasetIntf<Batch = T>>) -> Self {
        Dataset { value: arg }
    }
    pub fn len(&self) -> usize {
        self.value.len()
    }
    pub fn collate(&self, sample: Vec<usize>) -> T {
        self.value.collate(sample)
    }
}

pub trait DatasetIntf {
    type Batch: Clone;
    fn len(&self) -> usize;
    fn collate(&self, sample: Vec<usize>) -> Self::Batch;
}
