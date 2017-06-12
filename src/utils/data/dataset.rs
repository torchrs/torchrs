use std::rc::Rc;
use std::ops::Index;

#[derive(Clone)]
pub struct Dataset<T: Clone> {
    value: Rc<DatasetIntf<T, Output = T>>,
}
impl<T: Clone> Dataset<T> {
    pub fn new(arg: Rc<DatasetIntf<T, Output = T>>) -> Self {
        Dataset { value: arg }
    }
    pub fn len(&self) -> usize {
        self.value.len()
    }
    pub fn iter(&self) -> Box<Iterator<Item = T>> {
        self.value.iter()
    }
}

pub trait DatasetIntf<T>: Index<usize, Output = T> {
    fn len(&self) -> usize;
    fn iter(&self) -> Box<Iterator<Item = T>>;
}
