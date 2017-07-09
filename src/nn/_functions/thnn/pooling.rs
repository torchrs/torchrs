use autograd::{Function, FuncIntf, FuncDelegate, FIWrap};
use tensor::{TensorKindList, OptTensorKindList, TensorKind};

#[builder(pattern="owned")]
#[derive(Builder, Clone)]
pub struct MaxPoolFArgs {
    // just for code re-use
    #[builder(default="vec![1]")]
    pub kernel_size: Vec<i32>,
    pub stride: Vec<i32>,
    pub padding: Vec<i32>,
    pub dilation: Vec<i32>,
    #[builder(default="false")]
    pub ceil_mode: bool,
    #[builder(default="false")]
    pub return_indices: bool,
}

pub struct MaxPool1dArgs {
    pub v: MaxPoolFArgs,
}
pub struct MaxPool2dArgs {
    pub v: MaxPoolFArgs,
}
pub struct MaxPool3dArgs {
    pub v: MaxPoolFArgs,
}

impl Default for MaxPool1dArgs {
    fn default() -> Self {
        let args = MaxPoolFArgsBuilder::default()
            .stride(vec![0])
            .padding(vec![0])
            .dilation(vec![1])
            .build()
            .unwrap();
        MaxPool1dArgs { v: args }
    }
}

impl Default for MaxPool2dArgs {
    fn default() -> Self {
        let args = MaxPoolFArgsBuilder::default()
            .stride(vec![0])
            .padding(vec![0, 0])
            .dilation(vec![1, 1])
            .build()
            .unwrap();
        MaxPool2dArgs { v: args }
    }
}

impl Default for MaxPool3dArgs {
    fn default() -> Self {
        let args = MaxPoolFArgsBuilder::default()
            .stride(vec![0])
            .padding(vec![0, 0, 0])
            .dilation(vec![1, 1, 1])
            .build()
            .unwrap();
        MaxPool3dArgs { v: args }
    }
}

impl_func_args!(MaxPool2d, MaxPoolFArgs);

impl FuncIntf for MaxPool2d {
    fn forward(&mut self, input: &mut TensorKindList) -> TensorKindList {
        if self.args.stride[0] == 0 {
            self.args.stride = self.args.kernel_size.clone()
        }
        let p = self.args.clone();
        let mut input = input.remove(0);
        let mut backend = input.backend();
        let mut indices: TensorKind = ::torch::long_tensor(input.size()).into();
        let mut output = input.new(()).resize_as_(&input);
        backend.SpatialDilatedMaxPooling_updateOutput(&mut input,
                                                      &mut output,
                                                      &mut indices,
                                                      p.kernel_size[1],
                                                      p.kernel_size[0],
                                                      p.stride[1],
                                                      p.stride[0],
                                                      p.padding[1],
                                                      p.padding[0],
                                                      p.dilation[1],
                                                      p.dilation[0],
                                                      p.ceil_mode);
        let v = if p.return_indices {
            self.save_for_backward(&vec![input, indices.clone()]);
            self.mark_non_differentiable(&vec![indices.clone()]);
            vec![output, indices]
        } else {
            self.save_for_backward(&vec![input]);
            self.saved_tensors.push(indices);
            vec![output]
        };
        v
    }
    fn backward(&mut self, grad_output_list: &mut OptTensorKindList) -> OptTensorKindList {
        println!("MaxPool2d backward");
        let mut saved_tensors = self.saved_tensors();
        let mut grad_output = grad_output_list.remove(0).unwrap();
        let (mut input, mut indices) = if self.args.return_indices {
            (saved_tensors.remove(0), saved_tensors.remove(0))
        } else {
            (saved_tensors.remove(0), self.saved_tensors.remove(0))
        };
        let mut grad_input = grad_output.new(());
        let mut backend = input.backend();
        let p = &self.args;
        backend.SpatialDilatedMaxPooling_updateGradInput(&mut input, &mut grad_output, &mut grad_input, &mut indices,
                                                         p.kernel_size[1], p.kernel_size[0],
                                                         p.stride[1], p.stride[0],
                                                         p.padding[1], p.padding[0],
                                                         p.dilation[1], p.dilation[0],
                                                         p.ceil_mode);
        vec![Some(grad_input)]
    }
}
