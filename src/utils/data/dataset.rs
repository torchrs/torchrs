
pub type DatasetIntfRef<T> = ::std::rc::Rc<DatasetIntf<Batch = T>>;
pub trait DatasetIntf {
    type Batch: Clone;
    fn len(&self) -> usize;
    fn collate(&self, sample: Vec<usize>) -> Self::Batch;
}
