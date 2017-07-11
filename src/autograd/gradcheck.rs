use autograd::Variable;
use tensor::Tensor;
use torch;
use itertools::zip;

type Var64List = Vec<Variable<f64>>;


pub fn get_numerical_jacobian(func: fn(&Variable<f64>) -> Variable<f64>,
                              input: &Variable<f64>,
                              target: &Var64List,
                              eps: Option<f64>)
                              -> Vec<Tensor<f64>> {

    let eps = match eps {
        Some(e) => e,
        None => 1e-3,
    };
    let input = input.contiguous();
    let output_size = func(&input).numel();
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
            outa.copy_(func(&input).data_borrow());
            flat_tensor[i] = orig + eps;
            outb.copy_(func(&input).data_borrow());
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
