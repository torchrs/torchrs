use std::ops::Index;
use std::rc::Rc;

#[derive(Clone)]
pub struct Dataset<T: Clone> {
    value: Rc<DatasetIntf<T, Output = T>>,
}
impl<T: Clone> Dataset<T> {
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
