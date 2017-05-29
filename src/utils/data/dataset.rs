use std::ops::Index;
use std::rc::Rc;

#[derive(Clone)]
pub struct Dataset<T: Clone> {
    value: Rc<DatasetIntf<T, Output = T>>,
}

pub trait DatasetIntf<T>: Index<usize, Output = T> {
    fn len(&self) -> usize;
}
