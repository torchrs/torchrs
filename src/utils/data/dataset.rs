use std::rc::Rc;
use std::marker::PhantomData;

#[derive(Clone)]
pub struct Dataset<T: Clone + 'static, R: Clone + 'static> {
    value: Rc<DatasetIntf<T, CollateOutput = R>>,
}
impl<T: Clone + 'static, R: Clone + 'static> Dataset<T, R> {
    pub fn new(arg: Rc<DatasetIntf<T, CollateOutput = R>>) -> Self {
        Dataset { value: arg }
    }
    pub fn len(&self) -> usize {
        self.value.len()
    }
    pub fn iter(&self) -> Box<Iterator<Item = R>> {
        Box::new(self.value.iter())
    }
}

pub trait DatasetIntf<T: Clone> {
    type CollateOutput: Clone;
    fn len(&self) -> usize;
    fn iter(&self) -> Box<Iterator<Item = Self::CollateOutput>>;
    fn index(&mut self, idx: usize) -> T;
    fn collate(&self, sample: &Vec<&T>) -> Self::CollateOutput;
}
