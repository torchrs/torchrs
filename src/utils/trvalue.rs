use tensor::{Tensor, TensorKind, NumLimits};
use autograd::VarKind;

#[derive(Clone, Debug)]
pub enum TRVal {
    Bool(bool),
    Int(i32),
    Float(f32),
    Tensor(TensorKind),
    Variable(VarKind),
    Dims(Vec<usize>),
    ViewDims(Vec<isize>),
    Required,
}

impl From<f32> for TRVal {
    fn from(input: f32) -> Self {
        TRVal::Float(input)
    }
}
impl From<i32> for TRVal {
    fn from(input: i32) -> Self {
        TRVal::Int(input)
    }
}
impl From<bool> for TRVal {
    fn from(input: bool) -> Self {
        TRVal::Bool(input)
    }
}
impl From<TensorKind> for TRVal {
    fn from(input: TensorKind) -> Self {
        TRVal::Tensor(input)
    }
}
impl From<Vec<usize>> for TRVal {
    fn from(input: Vec<usize>) -> Self {
        TRVal::Dims(input)
    }
}
impl From<Vec<isize>> for TRVal {
    fn from(input: Vec<isize>) -> Self {
        TRVal::ViewDims(input)
    }
}
impl<T: NumLimits> From<Tensor<T>> for TRVal {
    fn from(input: Tensor<T>) -> Self {
        TRVal::Tensor(input.into())
    }
}
impl From<TRVal> for bool {
    fn from(input: TRVal) -> Self {
        match input {
            self::TRVal::Bool(x) => x.clone(),
            _ => unimplemented!(),
        }
    }
}
impl From<TRVal> for f32 {
    fn from(input: TRVal) -> Self {
        match input {
            self::TRVal::Float(x) => x.clone(),
            _ => unimplemented!(),
        }
    }
}
impl<T: NumLimits> From<TRVal> for Tensor<T> {
    fn from(input: TRVal) -> Self {
        match input {
            self::TRVal::Tensor(x) => x.clone().into(),
            _ => unimplemented!(),
        }
    }
}
impl From<TRVal> for Vec<usize> {
    fn from(input: TRVal) -> Self {
        match input {
            self::TRVal::Dims(x) => x.clone(),
            _ => unimplemented!(),
        }
    }
}
impl From<TRVal> for Vec<isize> {
    fn from(input: TRVal) -> Self {
        match input {
            self::TRVal::ViewDims(x) => x.clone(),
            _ => {println!("{:?}", input); unimplemented!()},
        }
    }
}
