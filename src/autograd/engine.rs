
use autograd::{Variable, Function, FuncId};
use tensor::Tensor;
use std::collections::{HashSet, HashMap};

#[derive(Default, Clone)]
pub struct ExecutionEngine {}

type FnRefs = HashMap<Function, u32>;
type FnDependencies = HashMap<Function, FnRefs>;

impl ExecutionEngine {
    fn _compute_dependencies(function: FuncId) -> FnDependencies {
        unimplemented!()
    }
    fn _free_backward_dependency(dependencies: FnDependencies,
                                 prev_fn: FuncId,
                                 fn_: FuncId,
                                 arg_id: u32)
                                 -> usize {
        unimplemented!();
    }
    fn _is_ready_for_backward(dependencies: FnDependencies, function: FuncId) -> bool {
        unimplemented!()
    }
    fn _add_grad<T>(need_copy: HashSet<Function>,
                    prev_grad: HashMap<u32, &Tensor<T>>,
                    output_nr: u32,
                    d_prev_grad: &Tensor<T>) {
        unimplemented!()
    }
    pub fn run_backward<T>(arg: &mut Variable<T>,
                           gradient: &mut Tensor<T>,
                           retain_variables: bool) {
        unimplemented!()
    }
}
