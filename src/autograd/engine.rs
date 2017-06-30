#[allow(non_snake_case)]
pub mod ExecutionEngine {
    use autograd::{Variable, Function, FuncId, RootKind, VarKind};
    use tensor::NumLimits;
    use std::collections::{HashSet, HashMap, VecDeque};
    use std::cell::RefCell;
    use itertools;
    use utils::unsafe_lib::Counter;

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
    fn _add_grad<T>(need_copy: &mut HashSet<VarKind>,
                    prev_grad: &mut Vec<Option<VarKind>>,
                    output_nr: usize,
                    d_prev_func: &VarKind)
        where T: NumLimits
    {
        // We can't match and operate on the vector at
        // the same time because that would be performing
        // a mutable borrow in the middle of an immutable
        // borrow so we take a temporary
        let grad_tensor_ = prev_grad[output_nr].clone();
        if let Some(mut grad_tensor) = grad_tensor_ {
            if need_copy.contains(&grad_tensor) {
                need_copy.remove(&grad_tensor);
                grad_tensor = grad_tensor.copy();
                // we perform the add before the assignment
                // in order to avoid an extra clone since
                // creation of the Option and subsequent
                // assignment moves the grad_tensor
                grad_tensor.addt_::<T>(T::one(), &d_prev_func);
                prev_grad[output_nr] = Some(grad_tensor.clone());
            } else {
                grad_tensor.addt_::<T>(T::one(), &d_prev_func);
            }
        } else {
            // We need to clone twice here as the compiler
            // can't determine the lifetime dependency
            // between the two
            need_copy.insert(d_prev_func.clone());
            prev_grad[output_nr] = Some(d_prev_func.clone());
        }

    }
    pub fn run_backward<T: NumLimits>(var: &mut Variable<T>,
                                      grad: VarKind,
                                      retain_variables: bool) {
        let grad_fn;
        match var.grad_fn() {
            Some(v) => grad_fn = v,
            None => {
                var._do_backward(&mut grad.into());
                return;
            }
        }
        let mut ready = VecDeque::new();
        ready.push_back((grad_fn.clone(), vec![Some(grad)]));
        let mut need_copy: HashSet<VarKind> = HashSet::new();
        let mut not_ready: HashMap<FuncId, Vec<Option<VarKind>>> = HashMap::new();

        let dependencies = _compute_dependencies(&grad_fn);
        while !ready.is_empty() {
            let (mut func, mut grad) = ready.pop_front().unwrap();
            let grad_input = func._do_backward(&mut grad, retain_variables);
            for (&(ref prev_func_, ref arg_id), ref d_prev_func_) in
                itertools::zip(func.previous_functions(), grad_input) {
                if !prev_func_.requires_grad() {
                    continue;
                }
                let d_prev_func = match *d_prev_func_ {
                    Some(ref f) => f,
                    None => continue,
                };
                let prev_func = match prev_func_ {
                    &RootKind::RootVar(ref v) => {
                        v.clone()._do_backward(&mut Some(d_prev_func.clone()));
                        return;
                    }
                    &RootKind::RootFunc(ref f) => f,
                };
                let output_nr = _free_backward_dependency(&dependencies, prev_func, &func, *arg_id);
                let is_ready = _is_ready_for_backward(&dependencies, prev_func);
                if is_ready {
                    let prev_grad = if not_ready.contains_key(&prev_func.id) {
                        let mut prev_grad = not_ready[&prev_func.id].clone();
                        _add_grad::<T>(&mut need_copy, &mut prev_grad, output_nr, &d_prev_func);
                        prev_grad
                    } else {
                        vec![Some(d_prev_func.clone())]
                    };
                    ready.push_front((prev_func.clone(), prev_grad))
                } else {
                    let mut prev_grad = if not_ready.contains_key(&prev_func.id) {
                        not_ready[&prev_func.id].clone()
                    } else {
                        prev_func.output_ids().iter().map(|_| None).collect()
                    };
                    _add_grad::<T>(&mut need_copy, &mut prev_grad, output_nr, &d_prev_func);
                    not_ready.insert(prev_func.id, prev_grad.clone());
                }
            }
        }
    }
}
