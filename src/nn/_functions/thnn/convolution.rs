use autograd::{Function, FuncIntf, FuncDelegate, Variable, FIWrap};
use tensor::{TensorKindList, OptTensorKindList, TensorKind};
use ::*;
use nn::backends::backend::*;

#[builder(pattern="owned")]
#[derive(Builder, Clone)]
pub struct ConvNdArgs {
    stride: Vec<i32>,
    padding: Vec<i32>,
    dilation: Vec<i32>,
    kernel_size: Vec<i32>,
    #[builder(default="false")]
    transposed: bool,
    output_padding: Vec<i32>,
    groups: u32,
    #[builder(default="false")]
    benchmark: bool,
    #[builder(default="false")]
    cudnn_enabled: bool,
}

#[builder(pattern="owned")]
#[derive(Builder)]
pub struct Conv1dFArgs {
    #[builder(default="vec![1]")]
    pub kernel_size: Vec<i32>,
    #[builder(default="vec![1]")]
    stride: Vec<i32>,
    #[builder(default="vec![0]")]
    padding: Vec<i32>,
    #[builder(default="vec![1]")]
    dilation: Vec<i32>,
    #[builder(default="1")]
    groups: u32,
}

impl Default for Conv1dFArgs {
    fn default() -> Self {
        Conv1dFArgsBuilder::default().build().unwrap()
    }
}

#[builder(pattern="owned")]
#[derive(Builder)]
pub struct Conv2dFArgs {
    #[builder(default="1")]
    pub in_features: usize,
    #[builder(default="1")]
    pub out_features: usize,
    #[builder(default="vec![1, 1]")]
    pub kernel_size: Vec<i32>,
    #[builder(default="vec![1, 1]")]
    pub stride: Vec<i32>,
    #[builder(default="vec![0, 0]")]
    pub padding: Vec<i32>,
    #[builder(default="vec![1, 1]")]
    pub dilation: Vec<i32>,
    #[builder(default="1")]
    pub groups: u32,
}

impl Default for Conv2dFArgs {
    fn default() -> Self {
        Conv2dFArgsBuilder::default().build().unwrap()
    }
}

#[builder(pattern="owned")]
#[derive(Builder)]
pub struct Conv3dFArgs {
    #[builder(default="vec![1, 1, 1]")]
    pub kernel_size: Vec<i32>,
    #[builder(default="vec![1, 1, 1]")]
    stride: Vec<i32>,
    #[builder(default="vec![0, 0, 0]")]
    padding: Vec<i32>,
    #[builder(default="vec![1, 1, 1]")]
    dilation: Vec<i32>,
    #[builder(default="1")]
    groups: u32,
}

impl Default for Conv3dFArgs {
    fn default() -> Self {
        Conv3dFArgsBuilder::default().build().unwrap()
    }
}

impl<'a> From<&'a mut Conv2dFArgs> for ConvNdArgs {
    fn from(input: &'a mut Conv2dFArgs) -> Self {
        ConvNdArgsBuilder::default()
            .kernel_size(input.kernel_size.clone())
            .stride(input.stride.clone())
            .padding(input.padding.clone())
            .dilation(input.dilation.clone())
            .groups(input.groups)
            .output_padding(vec![0, 0])
            .build()
            .unwrap()
    }
}
impl ConvNdArgs {
    fn is_dilated(&self) -> bool {
        self.dilation.iter().any(|v| *v != 1)
    }
}


impl_func_args!(ConvNd, ConvNdArgs);

fn compute_output(input: &mut TensorKind,
                  weight: &mut TensorKind,
                  bias: &mut Option<TensorKind>,
                  columns: &mut TensorKind,
                  ones: &mut TensorKind,
                  args: &ConvNdArgs)
                  -> TensorKind {
    let mut output = input.new(()).resize_as_(input);
    let dim = input.size().len();
    let dilated = args.is_dilated();
    let mut backend = input.backend();
    if dilated {
        if !args.transposed && dim == 4 {
            backend.SpatialDilatedConvolution_updateOutput(input,
                                                           &mut output,
                                                           weight,
                                                           bias,
                                                           columns,
                                                           ones,
                                                           args.kernel_size[1],
                                                           args.kernel_size[0],
                                                           args.stride[1],
                                                           args.stride[0],
                                                           args.padding[1],
                                                           args.padding[0],
                                                           args.dilation[1],
                                                           args.dilation[0])
        } else if !args.transposed && dim == 5 {
            backend.VolumetricDilatedConvolution_updateOutput(input,
                                                              &mut output,
                                                              weight,
                                                              bias,
                                                              columns,
                                                              ones,
                                                              args.kernel_size[0],
                                                              args.kernel_size[2],
                                                              args.kernel_size[1],
                                                              args.stride[0],
                                                              args.stride[2],
                                                              args.stride[1],
                                                              args.padding[0],
                                                              args.padding[2],
                                                              args.padding[1],
                                                              args.dilation[0],
                                                              args.dilation[2],
                                                              args.dilation[1])
        } else {
            panic!("unsupported ConvNd parameters")
        }
    } else {
        if args.transposed {
            /* !dilated && transposed */
            if dim == 4 {
                backend.SpatialFullConvolution_updateOutput(input,
                                                            &mut output,
                                                            weight,
                                                            bias,
                                                            columns,
                                                            ones,
                                                            args.kernel_size[1],
                                                            args.kernel_size[0],
                                                            args.stride[1],
                                                            args.stride[0],
                                                            args.padding[1],
                                                            args.padding[0],
                                                            args.output_padding[1],
                                                            args.output_padding[0]);
            } else if dim == 5 {
                backend.VolumetricFullConvolution_updateOutput(input,
                                                               &mut output,
                                                               weight,
                                                               bias,
                                                               columns,
                                                               ones,
                                                               args.stride[0],
                                                               args.stride[2],
                                                               args.stride[1],
                                                               args.padding[0],
                                                               args.padding[2],
                                                               args.padding[1],
                                                               args.output_padding[0],
                                                               args.output_padding[2],
                                                               args.output_padding[1]);
            } else {
                panic!("unsupported ConvNd parameters")
            }
        } else {
            /* !transposed */

            /* !dilated && !transposed */
            if dim == 4 {
                backend.SpatialConvolutionMM_updateOutput(input,
                                                          &mut output,
                                                          weight,
                                                          bias,
                                                          columns,
                                                          ones,
                                                          args.kernel_size[1],
                                                          args.kernel_size[0],
                                                          args.stride[1],
                                                          args.stride[0],
                                                          args.padding[1],
                                                          args.padding[0]);
            } else if dim == 5 && input.is_cuda() {
                backend.VolumetricConvolution_updateOutput(input,
                                                           &mut output,
                                                           weight,
                                                           bias,
                                                           columns,
                                                           ones,
                                                           args.stride[0],
                                                           args.stride[2],
                                                           args.stride[1],
                                                           args.padding[0],
                                                           args.padding[2],
                                                           args.padding[1]);
            } else if dim == 5 {
                backend.VolumetricConvolutionMM_updateOutput(input,
                                                             &mut output,
                                                             weight,
                                                             bias,
                                                             columns,
                                                             args.kernel_size[0],
                                                             args.kernel_size[2],
                                                             args.kernel_size[1],
                                                             args.stride[0],
                                                             args.stride[2],
                                                             args.stride[1],
                                                             args.padding[0],
                                                             args.padding[2],
                                                             args.padding[1]);
            } else {
                panic!("unsupported ConvNd parameters")
            }
        }
    }
    output
}

fn compute_grad_input(input: &mut TensorKind,
                      grad_output: &mut TensorKind,
                      weight: &mut TensorKind,
                      columns: &mut TensorKind,
                      ones: &mut TensorKind,
                      args: &ConvNdArgs)
                      -> TensorKind {
    let size = input.size();
    let dim = size.len();
    let mut grad_input = input.new(size);
    let dilated = args.is_dilated();
    let mut backend = input.backend();

    if dilated && !args.transposed {
        /* dilated && !transposed */
        if dim == 4 {
            backend.SpatialDilatedConvolution_updateGradInput(input,
                                                              grad_output,
                                                              &mut grad_input,
                                                              weight,
                                                              columns,
                                                              args.kernel_size[1],
                                                              args.kernel_size[0],
                                                              args.stride[1],
                                                              args.stride[0],
                                                              args.padding[1],
                                                              args.padding[0],
                                                              args.dilation[1],
                                                              args.dilation[0]);
        } else if dim == 5 {
            backend.VolumetricDilatedConvolution_updateGradInput(input,
                                                                 grad_output,
                                                                 &mut grad_input,
                                                                 weight,
                                                                 columns,
                                                                 args.kernel_size[0],
                                                                 args.kernel_size[2],
                                                                 args.kernel_size[1],
                                                                 args.stride[0],
                                                                 args.stride[2],
                                                                 args.stride[1],
                                                                 args.padding[0],
                                                                 args.padding[2],
                                                                 args.padding[1],
                                                                 args.dilation[0],
                                                                 args.dilation[2],
                                                                 args.dilation[1]);
        } else {
            panic!("unsupported ConvNdBackward parameters")
        }
    } else if dilated && args.transposed {
        panic!("unsupported ConvNdBackward parameters")
    } else if !dilated && args.transposed {
        /* !dilated && transposed */
        if dim == 4 {
            backend.SpatialFullConvolution_updateGradInput(input,
                                                           grad_output,
                                                           &mut grad_input,
                                                           weight,
                                                           columns,
                                                           args.kernel_size[1],
                                                           args.kernel_size[0],
                                                           args.stride[1],
                                                           args.stride[0],
                                                           args.padding[1],
                                                           args.padding[0],
                                                           args.output_padding[1],
                                                           args.output_padding[0]);
        } else if dim == 5 {
            backend.VolumetricFullConvolution_updateGradInput(input,
                                                              grad_output,
                                                              &mut grad_input,
                                                              weight,
                                                              columns,
                                                              ones,
                                                              args.stride[0],
                                                              args.stride[2],
                                                              args.stride[1],
                                                              args.padding[0],
                                                              args.padding[2],
                                                              args.padding[1],
                                                              args.output_padding[0],
                                                              args.output_padding[2],
                                                              args.output_padding[1]);
        } else {
            panic!("unsupported ConvNdBackward parameters")
        }
    } else
    /* !transposed */
    {
        /* !dilated && !transposed */
        if dim == 4 {
            backend.SpatialConvolutionMM_updateGradInput(input,
                                                         grad_output,
                                                         &mut grad_input,
                                                         weight,
                                                         columns,
                                                         ones,
                                                         args.kernel_size[1],
                                                         args.kernel_size[0],
                                                         args.stride[1],
                                                         args.stride[0],
                                                         args.padding[1],
                                                         args.padding[0]);
        } else if dim == 5 && input.is_cuda() {
            backend.VolumetricConvolution_updateGradInput(input,
                                                          grad_output,
                                                          &mut grad_input,
                                                          weight,
                                                          columns,
                                                          args.stride[0],
                                                          args.stride[2],
                                                          args.stride[1],
                                                          args.padding[0],
                                                          args.padding[2],
                                                          args.padding[1]);
        } else if dim == 5 {
            backend.VolumetricConvolutionMM_updateGradInput(input,
                                                            grad_output,
                                                            &mut grad_input,
                                                            weight,
                                                            columns,
                                                            ones,
                                                            args.kernel_size[0],
                                                            args.kernel_size[2],
                                                            args.kernel_size[1],
                                                            args.stride[0],
                                                            args.stride[2],
                                                            args.stride[1],
                                                            args.padding[0],
                                                            args.padding[2],
                                                            args.padding[1]);
        } else {
            panic!("unsupported ConvNdBackward parameters")
        }
    }

    grad_input
}
fn compute_grad_params(input: &mut TensorKind,
                       grad_output: &mut TensorKind,
                       weight: &mut TensorKind,
                       bias: &mut Option<TensorKind>,
                       columns: &mut TensorKind,
                       ones: &mut TensorKind,
                       args: &ConvNdArgs)
                       -> (TensorKind, Option<TensorKind>) {
    let dim = input.size().len();
    let dilated = args.is_dilated();
    let backend = input.backend();

    let mut grad_weight = weight.new(()).resize_as_(weight);
    grad_weight.zero_();
    let mut backend = input.backend();
    let mut grad_bias = None;
    /*
    if let &mut Some(ref bias) = bias {
        if args.should_compute_output(2) {
        	let new_bias = bias.new(()).resize_as_(&bias);
        	new_bias.zero_();
            grad_bias = Some(new_bias);
        }
    }
*/
    if dim != 4 && dim != 5 {
        panic!("unsupported ConvNdBackward dimension: {}", dim)
    }
    if dilated && args.transposed {
        /* dilated && transposed */
        /* NOT IMPLEMENTED */
        panic!("dilated AND transposed not supported as ConvNdBackward parameters")
    } else if dilated && !args.transposed {
        /* dilated && !transposed */
        if dim == 4 {
            backend.SpatialDilatedConvolution_accGradParameters(input,
                                                                grad_output,
                                                                &mut grad_weight,
                                                                &mut grad_bias,
                                                                columns,
                                                                ones,
                                                                args.kernel_size[1],
                                                                args.kernel_size[0],
                                                                args.stride[1],
                                                                args.stride[0],
                                                                args.padding[1],
                                                                args.padding[0],
                                                                args.dilation[1],
                                                                args.dilation[0],
                                                                1.0);
        } else if dim == 5 {
            backend.VolumetricDilatedConvolution_accGradParameters(input,
                                                                   grad_output,
                                                                   &mut grad_weight,
                                                                   &mut grad_bias,
                                                                   columns,
                                                                   ones,
                                                                   args.kernel_size[0],
                                                                   args.kernel_size[2],
                                                                   args.kernel_size[1],
                                                                   args.stride[0],
                                                                   args.stride[2],
                                                                   args.stride[1],
                                                                   args.padding[0],
                                                                   args.padding[2],
                                                                   args.padding[1],
                                                                   args.dilation[0],
                                                                   args.dilation[2],
                                                                   args.dilation[1],
                                                                   1.0);
        }
    } else if !dilated && args.transposed {
        /* !dilated && transposed */
        if dim == 4 {
            backend.SpatialFullConvolution_accGradParameters(input,
                                                             grad_output,
                                                             &mut grad_weight,
                                                             &mut grad_bias,
                                                             columns,
                                                             ones,
                                                             args.kernel_size[1],
                                                             args.kernel_size[0],
                                                             args.stride[1],
                                                             args.stride[0],
                                                             args.padding[1],
                                                             args.padding[0],
                                                             args.output_padding[1],
                                                             args.output_padding[0],
                                                             1.0);
        } else if dim == 5 {
            backend.VolumetricFullConvolution_accGradParameters(input,
                                                                grad_output,
                                                                &mut grad_weight,
                                                                &mut grad_bias,
                                                                columns,
                                                                ones,
                                                                args.stride[0],
                                                                args.stride[2],
                                                                args.stride[1],
                                                                args.padding[0],
                                                                args.padding[2],
                                                                args.padding[1],
                                                                args.output_padding[0],
                                                                args.output_padding[2],
                                                                args.output_padding[1],
                                                                1.0);
        }
    } else {
        /* !dilated && !transposed */
        if dim == 4 {
            backend.SpatialConvolutionMM_accGradParameters(input,
                                                           grad_output,
                                                           &mut grad_weight,
                                                           &mut grad_bias,
                                                           columns,
                                                           ones,
                                                           args.kernel_size[1],
                                                           args.kernel_size[0],
                                                           args.stride[1],
                                                           args.stride[0],
                                                           args.padding[1],
                                                           args.padding[0],
                                                           1.0);
        } else if dim == 5 && input.is_cuda() {
            backend.VolumetricConvolution_accGradParameters(input,
                                                            grad_output,
                                                            &mut grad_weight,
                                                            &mut grad_bias,
                                                            columns,
                                                            ones,
                                                            args.stride[0],
                                                            args.stride[2],
                                                            args.stride[1],
                                                            args.padding[0],
                                                            args.padding[2],
                                                            args.padding[1],
                                                            1.0);
        } else if dim == 5 {
            backend.VolumetricConvolutionMM_accGradParameters(input,
                                                              grad_output,
                                                              &mut grad_weight,
                                                              &mut grad_bias,
                                                              columns,
                                                              args.kernel_size[0],
                                                              args.kernel_size[2],
                                                              args.kernel_size[1],
                                                              args.stride[0],
                                                              args.stride[2],
                                                              args.stride[1],
                                                              args.padding[0],
                                                              args.padding[2],
                                                              args.padding[1],
                                                              1.0);
        }
    }
    (grad_weight, grad_bias)
}

fn view3d(t: TensorKind) -> TensorKind {
    unimplemented!()
}
fn view4d(t: TensorKind) -> TensorKind {
    unimplemented!()
}

impl ConvNd {
    fn view1d_as_2d(&mut self) {
        unimplemented!()
    }
    fn conv_forward_apply(&mut self, inputs: &mut TensorKindList) -> TensorKindList {

        let mut input = inputs.remove(0);
        let mut weight = inputs.remove(0);
        let mut ones = input.new(()).resize_as_(&input);
        let mut columns = input.new(()).resize_as_(&input);
        let mut save_list = vec![input.clone(), weight.clone()];
        let mut bias = if inputs.len() > 2 {
            let b = inputs.remove(0);
            save_list.push(b.clone());
            Some(b)
        } else {
            None
        };
        self.save_for_backward(&save_list);
        self.saved_tensors.push(ones.clone());
        self.saved_tensors.push(columns.clone());
        let k = input.size().len();
        if k == 3 {
            self.view1d_as_2d();
            input = view4d(input);
            weight = view4d(weight);
        }
        // let weight_size = weight.size(); -- NOT YET
        /* ......... */
        let mut output = compute_output(&mut input,
                                        &mut weight,
                                        &mut bias,
                                        &mut columns,
                                        &mut ones,
                                        &self.args);
        if k == 3 {
            output = view3d(output);
        };
        vec![output]
    }
}

impl FuncIntf for ConvNd {
    fn forward(&mut self, input: &mut TensorKindList) -> TensorKindList {
        // we don't support groups yet
        assert_eq!(self.args.groups, 1);
        self.conv_forward_apply(input)
    }
    fn backward(&mut self, mut input: &mut OptTensorKindList) -> OptTensorKindList {
        // run native code here
        unimplemented!()
    }
}
