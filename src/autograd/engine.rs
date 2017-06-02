
// silence warnings while still a WIP
#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(unused_assignments)]

use autograd::{Variable, Function, FuncId, FuncIntf};
use tensor::{Tensor, TensorKind};
use std::collections::{HashSet, HashMap, VecDeque};

pub struct ExecutionEngine {}



type FnRefs = HashMap<Function, u32>;
type FnDependencies = HashMap<Function, FnRefs>;

impl ExecutionEngine {
    fn _compute_dependencies(function: &Function) -> FnDependencies {
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
    pub fn run_backward<T: Copy>(var: &mut Variable<T>, grad: TensorKind, retain_variables: bool) {
        let grad_fn;
        match var.grad_fn() {
            Some(v) => grad_fn = v,
            None => {
                let grad_tensor = Tensor::<T>::from(grad);
                var._do_backward(grad_tensor[0]);
                return;
            }
        }
        let mut ready = VecDeque::new();
        ready.push_back((grad_fn.clone(), vec![grad]));
        //let mut need_copy = HashSet::new();

        let dependencies = Self::_compute_dependencies(&grad_fn);
        while !ready.is_empty() {
            let (func, grad) = ready.pop_front().unwrap();
            let grad_input = func.clone()._do_backward(&grad, retain_variables);
        }
    }
}
