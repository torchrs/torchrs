use tensor::{Tensor, TensorKind, NumLimits};

pub fn byte_tensor_kind<u8>(arg: u8) -> TensorKind {
    unimplemented!()
}
pub fn byte_tensor_kind<T>(arg: T) -> TensorKind where T: AsRef<Vec<u8>> {
	unimplemented!()
}
pub fn byte_tensor_kind<T>(arg: T) -> TensorKind where T: AsRef<Vec<Vec<u8>>> {
    unimplemented!()
}

pub fn float_tensor_kind<T>(arg: T) -> TensorKind {
    unimplemented!()
}
pub fn long_tensor_kind<T>(arg: T) -> TensorKind {
    unimplemented!()
}
pub fn byte_tensor<T>(arg: T) -> Tensor<u8> {
    unimplemented!()
}
pub fn float_tensor<T>(arg: T) -> Tensor<f32> {
    unimplemented!()
}
pub fn double_tensor<T>(arg: T) -> Tensor<f64> {
    unimplemented!()
}
pub fn long_tensor<T>(arg: T) -> Tensor<i64>  where T: AsRef<Vec<i64>> {
	unimplemented!()
}
pub fn tensor<S, f32>(arg: S) -> Tensor<f32> where S: AsRef<Vec<Tensor<f32>>>, f32: ::tensor::tensor::NumLimits {
    unimplemented!()
}
