use autograd::{Variable, VariableArgs, VarAccess};
use tensor::{NumLimits, Tensor};
use std::slice::{Iter, IterMut};
use torch;
use itertools::zip;

type Var64List = Vec<Variable<f64>>;

type Layer = fn(&Var64List) -> Var64List;
type PartialLayer = FnMut(&Var64List) -> Tensor<f64>;

pub fn contiguous(v: &Var64List) -> Var64List {
    v.iter().map(|v| v.contiguous()).collect()
}

pub fn get_numerical_jacobian(func: &mut PartialLayer,
                              input: &Var64List,
                              target: &Var64List,
                              eps: Option<f64>)
                              -> Vec<Tensor<f64>> {

    let eps = match eps {
        Some(e) => e,
        None => 1e-3,
    };
    let input_ = contiguous(input);
    let input = &input_;
    let output_size = func(input).numel();
    // XXX look at when this may not be a flat vector
    let jacobian: Vec<Tensor<f64>> = target
        .iter()
        .map(|v| ::torch::zeros([v.numel(), output_size]))
        .collect();

    let x_tensors: Vec<Tensor<f64>> = target
        .iter()
        .filter_map(|v| if v.requires_grad() {
                        Some(v.data_borrow().clone())
                    } else {
                        None
                    })
        .collect();
    let j_tensors = jacobian.clone();

    let mut outa = torch::double_tensor(output_size);
    let mut outb = torch::double_tensor(output_size);

    for (ref x_tensor_, ref mut d_tensor) in zip(x_tensors, j_tensors) {
        let x_tensor: &Tensor<f64> = x_tensor_;
        let mut flat_tensor = x_tensor.view([-1]);
        for i in 0..flat_tensor.numel() {
            let orig = flat_tensor[i];
            flat_tensor[i] = orig - eps;
            outa.copy_(&func(input));
            flat_tensor[i] = orig + eps;
            outb.copy_(&func(input));
            flat_tensor[i] = orig;

            outb.addt_(-1., &outa).div_(2. * eps);
            d_tensor.s([i as isize]).copy_(&outb);
        }
    }
    jacobian
}

pub fn zero_gradients(input: &mut Var64List) {
    for v in input {
        if let &mut Some(ref mut grad) = v.grad() {
            grad.detach_();
            grad.data().zero_();
        }
    }
}

pub fn iter_gradients(input: &mut Var64List) -> Vec<Option<Tensor<f64>>> {
    input
        .iter_mut()
        .map(|v| if v.requires_grad() {
                 if let &mut Some(ref mut grad) = v.grad() {
                     Some(grad.data().clone())
                 } else {
                     None
                 }
             } else {
                 None
             })
        .collect()
}

pub fn get_analytical_jacobian(input: &mut Var64List, output: &Variable<f64>) -> Vec<Tensor<f64>> {
    let mut jacobian: Vec<Tensor<f64>> = input
        .iter()
        .map(|v| ::torch::zeros([v.numel(), output.numel()]))
        .collect();
    let grad_output: Tensor<f64> = torch::zeros(output.data_borrow().size());
    let mut flat_grad_output = grad_output.view([-1]);

    for i in 0..flat_grad_output.numel() {
        flat_grad_output.zero_();
        flat_grad_output[i] = 1.;
        zero_gradients(input);
        for (ref mut jacobian_x, ref d_x_) in zip(jacobian.iter_mut(), iter_gradients(input)) {
            let d_x_: &Option<Tensor<f64>> = d_x_;
            if let &Some(ref d_x) = d_x_ {
                jacobian_x.s([-1, i as isize]).copy_(d_x);
            }

        }

    }
    jacobian
}

#[derive(Debug, Clone, Default)]
pub struct GradValues {
    pub eps: Option<f64>,
    pub atol: Option<f64>,
    pub rtol: Option<f64>,
}

//
//       Check gradients computed via small finite differences
//       against analytical gradients
//
//    The check between numerical and analytical has the same behaviour as
//    numpy.allclose https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html
//    meaning it check that
//        absolute(a - n) <= (atol + rtol * absolute(n))
//    is true for all elements of analytical jacobian a and numerical jacobian n.
//
//    Args:
//        func: function that takes Vec<Variable> inputs and returns
//            a Vec<Variable>
//        inputs: Vec<Variable>
//        eps: perturbation for finite differences
//        atol: absolute tolerance
//        rtol: relative tolerance
//
//    Returns:
//        True if all differences satisfy allclose condition
pub fn gradcheck(func: Layer, inputs: &mut Var64List, values: GradValues) -> bool {
    let eps = match values.eps {
        Some(e) => e,
        None => 1e-6,
    };
    let atol = match values.atol {
        Some(e) => e,
        None => 1e-5,
    };
    let rtol = match values.rtol {
        Some(e) => e,
        None => 1e-3,
    };
    let output = func(inputs);
    for (i, ref o) in output.iter().enumerate() {
        if !o.requires_grad() {
            continue;
        }
        let mut f = move |input: &Var64List| func(input)[i].data().clone();
        let analytical = get_analytical_jacobian(inputs, o);
        let numerical = get_numerical_jacobian(&mut f, inputs, inputs, Some(eps));
        for (ref a, ref n) in zip(analytical, numerical) {
            if !zip(a.sub(n).abs().iter(), n.abs().mul(rtol).add(atol).iter())
                   .all(|(a, b)| a <= b) {
                return false;
            }
        }
    }
    zero_gradients(inputs);
    let mut output = func(inputs);
    let args = VariableArgs::build().volatile(true).done();

    let grads = output
        .iter()
        .map(|o| Variable::new_args(o.data_borrow().new(o.size()).zero_(), &args))
        .collect();
    torch::autograd::backward(&mut output, &grads, None, None);
    true
}

//       Check gradients of gradients computed via small finite differences
//       against analytical gradients
//    This function checks that backpropagating through the gradients computed
//    to the given grad_outputs are correct.
//
//    The check between numerical and analytical has the same behaviour as
//    numpy.allclose https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html
//    meaning it check that
//        absolute(a - n) <= (atol + rtol * absolute(n))
//    is true for all elements of analytical gradient a and numerical gradient n.
//
//    Args:
//        func: Python function that takes Variable inputs and returns
//            a tuple of Variables
//        inputs: tuple of Variables
//        grad_outputs: tuple of Variables
//        eps: perturbation for finite differences
//        atol: absolute tolerance
//        rtol: relative tolerance
//
//    Returns:
//        True if all differences satisfy allclose condition
/*
pub fn gradgradcheck(func: Layer,
                 inputs: &mut Var64List,
                 grad_outputs: &mut Var64List,
                 values: GradValues)
                 -> bool {
    let mut f = move |input: &Var64List| {
        let outputs = func(input);
        let inputs = inputs.iter().map(|v| )



                 }

*/
