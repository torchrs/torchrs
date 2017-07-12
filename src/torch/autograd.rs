use autograd::{Variable, VariableArgs, VarAccess};
use tensor::NumLimits;
use itertools::zip;



fn _make_grads<T : NumLimits>(outputs: &Vec<Variable<T>>, grads: &Vec<Option<Variable<T>>>, create_graph: Option<bool>) -> (Vec<Option<Variable<T>>>, bool) {
	let create_graph = match create_graph {
		Some(val) => val,
		None => grads.iter().any(|v| if let Some(ref grad) = *v {!grad.is_volatile()} else {false})
	};
	let mut new_grads : Vec<Option<Variable<T>>> = Vec::new();
	for (ref out, ref grad_) in zip(outputs, grads) {
		if let &&Some(ref grad) = grad_ {
			new_grads.push(Some(grad.clone()));
		} else {
			if out.requires_grad() {
				if out.numel() != 1 {
					panic!("grad can be implicitly created only for scalar outputs");
				}
				let data = out.data_borrow();
				let args = VariableArgs::build().volatile(!create_graph).done();
				let new_grad = Variable::new_args(data.new(()).resize_as_(data).fill_(T::one()).clone(), &args);
				new_grads.push(Some(new_grad));
			} else {
				new_grads.push(None);
			}
		}
	}

	(new_grads, create_graph)
}

pub fn backward<T: NumLimits>(variables: Vec<Variable<T>>, grad_variables: Option<Vec<Variable<T>>>, retain_graph: Option<bool>, create_graph: Option<bool>, retain_variables: Option<bool>) {
	let grad_variables = if let Some(grads) = grad_variables {
		grads.iter().map(|v| Some(v.clone())).collect()
	} else {
		let mut v = Vec::new();
		for _ in 0..variables.len() {
			v.push(None);
		}
		v
	};

}


