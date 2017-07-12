use autograd::{Variable, VarKind, VariableArgs, VarAccess};
use tensor::NumLimits;
use itertools::zip;



fn _make_grads<T: NumLimits>(outputs: &Vec<Variable<T>>,
                             grads: &Vec<Option<Variable<T>>>,
                             create_graph: Option<bool>)
                             -> (Vec<Option<VarKind>>, bool) {
    let create_graph = match create_graph {
        Some(val) => val,
        None => {
            grads
                .iter()
                .any(|v| if let Some(ref grad) = *v {
                         !grad.is_volatile()
                     } else {
                         false
                     })
        }
    };
    let mut new_grads: Vec<Option<VarKind>> = Vec::new();
    for (ref out, ref grad_) in zip(outputs, grads) {
        if let &&Some(ref grad) = grad_ {
            new_grads.push(Some(grad.clone().into()));
        } else {
            if out.requires_grad() {
                if out.numel() != 1 {
                    panic!("grad can be implicitly created only for scalar outputs");
                }
                let data = out.data_borrow();
                let args = VariableArgs::build().volatile(!create_graph).done();
                let new_grad =
                    Variable::new_args(data.new(()).resize_as_(data).fill_(T::one()).clone(),
                                       &args);
                new_grads.push(Some(new_grad.into()));
            } else {
                new_grads.push(None);
            }
        }
    }

    (new_grads, create_graph)
}

//       Computes the sum of gradients of given variables w.r.t. graph leaves.
//
//    The graph is differentiated using the chain rule. If any of ``variables``
//    are non-scalar (i.e. their data has more than one element) and require
//    gradient, the function additionaly requires specifying ``grad_variables``.
//    It should be a sequence of matching length, that contains gradient of
//    the differentiated function w.r.t. corresponding variables (``None`` is an
//    acceptable value for all variables that don't need gradient tensors).
//
//    This function accumulates gradients in the leaves - you might need to zero
//    them before calling it.
//
//    Arguments:
//        variables (sequence of Variable): Variables of which the derivative will be
//            computed.
//        grad_variables (sequence of (Tensor, Variable or None)): Gradients w.r.t.
//            each element of corresponding variables.  Any tensors will be
//            automatically converted to Variables that are volatile unless
//            ``create_graph`` is True.  None values can be specified for scalar
//            Variables or ones that don't require grad. If a None value would
//            be acceptable for all grad_variables, then this argument is optional.
//        retain_graph (bool, optional): If False, the graph used to compute the grad
//            will be freed. Note that in nearly all cases setting this option to True
//            is not needed and often can be worked around in a much more efficient
//            way. Defaults to the value of ``create_graph``.
//        create_graph (bool, optional): If true, graph of the derivative will
//            be constructed, allowing to compute higher order derivative products.
//            Defaults to False, unless ``grad_variables`` contains at least one
//            non-volatile Variable.
//

pub fn backward<T: NumLimits>(variables: &mut Vec<Variable<T>>,
                              grad_variables: &Vec<Variable<T>>,
                              retain_graph: Option<bool>,
                              create_graph: Option<bool>) {
    let grad_variables = if grad_variables.len() > 0 {
        grad_variables.iter().map(|v| Some(v.clone())).collect()
    } else {
        let mut v = Vec::new();
        for _ in 0..variables.len() {
            v.push(None);
        }
        v
    };
    let (grad_variables, create_graph) = _make_grads(variables, &grad_variables, create_graph);
    let retain_graph = if let Some(value) = retain_graph {
        value
    } else {
        create_graph
    };
    ::autograd::ExecutionEngine::run_backward(variables, grad_variables, retain_graph);
}
