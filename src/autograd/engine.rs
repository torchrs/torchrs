
// silence warnings while still a WIP
#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(unused_assignments)]
#![allow(unused_imports)]

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::cell::{Cell, RefCell};
use std::ops::{Index, IndexMut};

// Pythonesque Counter implementation
// XXX Move to a separate module
static ZERO: usize = 0;
pub struct Counter<T: Hash + Eq + Clone> {
    pub map: HashMap<T, Cell<usize>>,
}
impl<T: Hash + Eq + Clone> Counter<T> {
    pub fn new() -> Self {
        Counter { map: HashMap::new() }
    }
    pub fn len(&self) -> usize {
        self.map.len()
    }
    pub fn remove(&mut self, idx: &T) {
        self.map.remove(idx);
    }
}
impl<T: Hash + Eq + Clone> Index<T> for Counter<T> {
    type Output = usize;
    fn index(&self, idx: T) -> &Self::Output {
        if self.map.contains_key(&idx) {
            let cntp = self.map[&idx].as_ptr();
            unsafe { &*cntp }
        } else {
            //map.insert(idx, Cell::new(0));
            //let mut cntp = map[&idx].as_ptr();
            //unsafe {& *cntp}
            &ZERO
        }
    }
}
impl<T: Hash + Eq + Clone> IndexMut<T> for Counter<T> {
    fn index_mut(&mut self, idx: T) -> &mut Self::Output {
        if self.map.contains_key(&idx) {
            let cntp = self.map[&idx].as_ptr();
            unsafe { &mut *cntp }
        } else {
            self.map.insert(idx.clone(), Cell::new(0));
            let cntp = self.map[&idx].as_ptr();
            unsafe { &mut *cntp }
        }
    }
}


#[allow(non_snake_case)]
pub mod ExecutionEngine {
    use autograd::{Variable, Function, FuncId, FuncIntf, RootKind, VarId};
    use tensor::{Tensor, TensorKind, TensorKindList};
    use std::collections::{HashSet, HashMap, VecDeque};
    use std::cell::RefCell;
    use itertools;
    use super::Counter;

    type FnRefs = RefCell<Vec<Counter<FuncId>>>;
    type FnDependencies = HashMap<FuncId, FnRefs>;

    fn fn_ref_init(count: usize) -> FnRefs {
        let mut v = Vec::new();
        for _ in 0..count {
            v.push(Counter::new())
        }
        RefCell::new(v)
    }

    fn _compute_dependencies(function: &Function) -> FnDependencies {
        let mut dependencies = FnDependencies::new();
        let mut seen: HashSet<FuncId> = HashSet::new();
        let mut queue = VecDeque::new();
        seen.insert(function.id);
        queue.push_back(function);
        while !queue.is_empty() {
            let func = queue.pop_front().unwrap();
            for &(ref prev_func_, ref arg_id) in func.previous_functions().iter() {
                let prev_func = match prev_func_ {
                    &RootKind::RootVar(_) => continue,
                    &RootKind::RootFunc(ref f) => f,
                };
                if !dependencies.contains_key(&prev_func.id) {
                    dependencies.insert(prev_func.id, fn_ref_init(prev_func.output_ids().len()));
                }
                let output_idx = prev_func.output_ids()[arg_id];
                let mut fnrefs = dependencies[&prev_func.id].borrow_mut();
                fnrefs[output_idx][func.id] += 1;
                if !seen.contains(&prev_func.id) {
                    queue.push_back(prev_func);
                    seen.insert(prev_func.id);
                }
            }
        }
        dependencies
    }
    fn _free_backward_dependency(dependencies: &FnDependencies,
                                 prev_func: &Function,
                                 func: &Function,
                                 arg_id: i32)
                                 -> usize {
        let mut deps = dependencies[&prev_func.id].borrow_mut();
        let output_idx = prev_func.output_ids()[&arg_id];
        let mut output_deps = &mut deps[output_idx];
        output_deps[func.id] -= 1;
        if output_deps[func.id] == 0 {
            output_deps.remove(&func.id)
        }
        output_idx
    }
    fn _is_ready_for_backward(dependencies: &FnDependencies, function: &Function) -> bool {
        for ref deps in dependencies[&function.id].borrow().iter() {
            if deps.len() > 0 {
                return false;
            }
        }
        return true;
    }
    fn _add_grad(need_copy: &mut HashSet<TensorKind>,
                 prev_grad: &mut Vec<Option<TensorKind>>,
                 output_nr: usize,
                 d_prev_func: &TensorKind) {
        // We can't match and operate on the vector at
        // the same time because that would be performing
        // a mutable borrow in the middle of an immutable
        // borrow so we use a boolean
        // d_prev_func is just used as a placeholder so that
        // both arms match
        let (mut grad_tensor, matched) = match prev_grad[output_nr] {
            Some(ref t) => (t.clone(), true),
            None => (d_prev_func.clone(), false),
        };
        if matched {
            if need_copy.contains(&grad_tensor) {
                need_copy.remove(&grad_tensor);
                grad_tensor = grad_tensor.copy();
                // we perform the add before the assignment
                // in order to avoid an extra clone since
                // creation of the Option and subsequent
                // assignment moves the grad_tensor
                grad_tensor.add_(d_prev_func);
                prev_grad[output_nr] = Some(grad_tensor);
            } else {
                grad_tensor.add_(d_prev_func);
            }
        } else {
            // We need to clone twice here as the compiler
            // can't determine the lifetime dependency
            // between the two
            need_copy.insert(d_prev_func.clone());
            prev_grad[output_nr] = Some(d_prev_func.clone());
        }

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

        let dependencies = _compute_dependencies(&grad_fn);
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
                let output_nr = _free_backward_dependency(&dependencies, prev_func, &func, *arg_id);
                let is_ready = _is_ready_for_backward(&dependencies, prev_func);
                if is_ready {
                    let prev_grad = if not_ready.contains_key(&prev_func.id) {
                        let mut prev_grad = not_ready[&prev_func.id].clone();
                        _add_grad(&mut need_copy, &mut prev_grad, output_nr, &d_prev_func);
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
                    let mut prev_grad = if not_ready.contains_key(&prev_func.id) {
                        not_ready[&prev_func.id].clone()
                    } else {
                        prev_func.output_ids().iter().map(|_| None).collect()
                    };
                    _add_grad(&mut need_copy, &mut prev_grad, output_nr, &d_prev_func);
                    not_ready.insert(prev_func.id, prev_grad.clone());
                }
            }
        }
    }
}
