use autograd::Variable;
use tensor::Tensor;
use torch;
use itertools::zip;


pub fn get_numerical_jacobian(func: fn(&Variable<f64>) -> Variable<f64>,
                              input: &Variable<f64>,
                              target: &Vec<Variable<f64>>,
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
            d_tensor.s([i]).copy_(&outb);
        }
    }
    jacobian
}
