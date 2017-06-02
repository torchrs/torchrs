
// silence warnings while still a WIP
#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(unused_assignments)]
#![allow(unused_imports)]

use autograd::{Variable, Function, FuncId, FuncIntf, RootKind};
use tensor::{Tensor, TensorKind, TensorKindList};
use std::collections::{HashSet, HashMap, VecDeque};
use itertools;

pub struct ExecutionEngine {}



type FnRefs = HashMap<FuncId, u32>;
type FnDependencies = HashMap<FuncId, FnRefs>;

impl ExecutionEngine {
    fn _compute_dependencies(function: &Function) -> FnDependencies {
        unimplemented!()
    }
    fn _free_backward_dependency(dependencies: &FnDependencies,
                                 prev_func: &Function,
                                 func: &Function,
                                 arg_id: i32)
                                 -> usize {
        unimplemented!();
    }
    fn _is_ready_for_backward(dependencies: &FnDependencies, function: &Function) -> bool {
        for ref deps in dependencies[&function.id].iter() {
            if deps.1 > &0 {
                return false;
            }
        }
        return true;
    }
    fn _add_grad(need_copy: &mut HashSet<TensorKind>,
                 prev_grad: &Vec<Option<TensorKind>>,
                 output_nr: usize,
                 d_prev_grad: &TensorKind) {
        unimplemented!()
    }
    pub fn run_backward<T: Copy>(var: &mut Variable<T>, grad: TensorKind, retain_variables: bool) {
        let grad_fn;
        match var.grad_fn() {
            Some(v) => grad_fn = v,
            None => {
                let grad_tensor = Tensor::<T>::from(grad);
                var._do_backward(&grad_tensor);
                return;
            }
        }
        let mut ready = VecDeque::new();
        ready.push_back((grad_fn.clone(), vec![grad]));
        let mut need_copy: HashSet<TensorKind> = HashSet::new();
        let mut not_ready: HashMap<FuncId, Vec<Option<TensorKind>>> = HashMap::new();

        let dependencies = Self::_compute_dependencies(&grad_fn);
        while !ready.is_empty() {
            let (mut func, grad) = ready.pop_front().unwrap();
            let grad_input = func._do_backward(&grad, retain_variables);
            for (&(ref prev_func_, ref arg_id), ref d_prev_func) in
                itertools::zip(func.previous_functions(), grad_input) {
                if !prev_func_.requires_grad() {
                    continue;
                }
                let prev_func = match prev_func_ {
                    &RootKind::RootVar(ref v) => {
                        v.clone()._do_backward(&d_prev_func);
                        return;
                    }
                    &RootKind::RootFunc(ref f) => f,
                };
                let output_nr =
                    Self::_free_backward_dependency(&dependencies, prev_func, &func, *arg_id);
                let is_ready = Self::_is_ready_for_backward(&dependencies, prev_func);
                if is_ready {
                    let prev_grad = if not_ready.contains_key(&prev_func.id) {
                        let prev_grad = not_ready[&prev_func.id].clone();
                        Self::_add_grad(&mut need_copy, &prev_grad, output_nr, &d_prev_func);
                        prev_grad
                            .iter()
                            .map(|sv| if let &Some(ref v) = sv {
                                     v.clone()
                                 } else {
                                     panic!("found None")
                                 })
                            .collect()
                    } else {
                        vec![d_prev_func.clone()]
                    };
                    ready.push_front((prev_func.clone(), prev_grad))
                } else {
                    let prev_grad = if not_ready.contains_key(&prev_func.id) {
                        not_ready[&prev_func.id].clone()
                    } else {
                        prev_func.output_ids().iter().map(|_| None).collect()
                    };
                    Self::_add_grad(&mut need_copy, &prev_grad, output_nr, &d_prev_func);
                    not_ready.insert(prev_func.id, prev_grad.clone());
                }
            }
        }
    }
}
