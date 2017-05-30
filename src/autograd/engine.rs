
use autograd::Variable;
use tensor::Tensor;

#[derive(Default, Clone)]
pub struct ExecutionEngine {}

impl ExecutionEngine {
    pub fn run_backward<T>(arg: &mut Variable<T>,
                           gradient: &mut Tensor<T>,
                           retain_variables: bool) {
        unimplemented!()
    }
}
