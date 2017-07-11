#![allow(unused_variables)]
use tensor::*;
use std::cell::RefMut;
use autograd::{Variable, VariableArgs, VarAccess, VarKind};


impl<T: NumLimits> Variable<T> {
    pub fn abs(&self) -> Self {
        unimplemented!()
    }
    pub fn abs_(self) -> Self {
        unimplemented!()
    }
    pub fn acos(&self) -> Self {
        unimplemented!()
    }
    pub fn acos_(self) -> Self {
        unimplemented!()
    }
    pub fn add(&self, rhs: T) -> Self {
        let mut v = self.copy();
        v.data().clone().add_(rhs);
        v
    }
    pub fn add_(self, rhs: T) -> Self {
        self.data_borrow().clone().add_(rhs);
        self
    }
    pub fn addt(&self, val: T, rhs: &Self) -> Self {
        let mut v = self.copy();
        v.data().clone().addt_(val, rhs.data_borrow());
        v
    }
    pub fn addt_(&mut self, val: T, rhs: &Self) -> &mut Self {
        self.data().addt_(val, &rhs.data_borrow());
        self
    }
    pub fn addbmm(&self, beta: T, alpha: T, tensor1: &Self, tensor2: &Self) -> Self {
        unimplemented!()
    }
    pub fn addbmm_(self, beta: T, alpha: T, tensor1: &Self, tensor2: &Self) -> Self {
        unimplemented!()
    }
    pub fn addcdiv(&self, value: T, tensor1: &Self, tensor2: &Self) -> Self {
        unimplemented!()
    }
    pub fn addcdiv_(self, value: T, tensor1: &Self, tensor2: &Self) -> Self {
        unimplemented!()
    }
    pub fn addcmul(&self, value: T, tensor1: &Self, tensor2: &Self) -> Self {
        unimplemented!()
    }
    pub fn addcmul_(self, value: T, tensor1: &Self, tensor2: &Self) -> Self {
        unimplemented!()
    }
    pub fn addmm(&self, beta: T, alpha: T, tensor1: &Self, tensor2: &Self) -> Self {
        unimplemented!()
    }
    pub fn addmm_(self, beta: T, alpha: T, tensor1: &Self, tensor2: &Self) -> Self {
        unimplemented!()
    }
    pub fn addmv(&self, beta: T, alpha: T, tensor1: &Self, vec: &Self) -> Self {
        unimplemented!()
    }
    pub fn addmv_(self, beta: T, alpha: T, tensor1: &Self, vec: &Self) -> Self {
        unimplemented!()
    }
    pub fn addr(&self, beta: T, alpha: T, vec1: &Self, vec2: &Self) -> Self {
        unimplemented!()
    }
    pub fn addr_(self, beta: T, alpha: T, vec1: &Self, vec2: &Self) -> Self {
        unimplemented!()
    }
    pub fn asin(&self) -> Self {
        unimplemented!()
    }
    pub fn asin_(self) -> Self {
        unimplemented!()
    }
    pub fn atan(&self) -> Self {
        unimplemented!()
    }
    pub fn atan2(&self) -> Self {
        unimplemented!()
    }
    pub fn atan2_(self) -> Self {
        unimplemented!()
    }
    pub fn baddbmm(&self, beta: T, alpha: T, tensor1: &Self, tensor2: &Self) -> Self {
        unimplemented!()
    }
    pub fn baddbmm_(self, beta: T, alpha: T, tensor1: &Self, tensor2: &Self) -> Self {
        unimplemented!()
    }
    pub fn bernoulli(&self, p: T) -> Self {
        unimplemented!()
    }
    pub fn bernoulli_(self, p: T) -> Self {
        unimplemented!()
    }
    pub fn bmm(&self, other: &Self) -> Self {
        unimplemented!()
    }
    pub fn byte(self) -> Self {
        unimplemented!()
    }
    //
    // cauchy_
    //
    pub fn ceil(&self) -> Self {
        unimplemented!()
    }
    pub fn ceil_(self) -> Self {
        unimplemented!()
    }
    pub fn char(self) -> Self {
        unimplemented!()
    }
    pub fn chunk(&self, n_chunks: usize, dim: usize) -> Vec<Self> {
        unimplemented!()
    }
    pub fn clamp(&self, min: f32, max: f32) -> Self {
        unimplemented!()
    }
    pub fn clamp_(self, min: f32, max: f32) -> Self {
        unimplemented!()
    }
    pub fn contiguous(&self) -> Self {
        unimplemented!()
    }
    // perform deep copy
    pub fn copy(&self) -> Self {
        let new = Variable::new_args(self.data_borrow().copy(), &VariableArgs::default());
        let (inner, new_inner) = (self.access(), new.access());
        new_inner.copy_refs(inner);
        new
    }
    pub fn copy_(self, src: &Self) -> Self {
        unimplemented!()
    }
    pub fn copy_async_(self, src: &Self) -> Self {
        unimplemented!()
    }
    pub fn cos(&self) -> Self {
        unimplemented!()
    }
    pub fn cos_(self) -> Self {
        unimplemented!()
    }
    pub fn cosh(&self) -> Self {
        unimplemented!()
    }
    pub fn cosh_(self) -> Self {
        unimplemented!()
    }
    pub fn cpu(&self) -> Self {
        unimplemented!()
    }
    pub fn cross(&self, dim: Option<i32>) -> Self {
        unimplemented!()
    }
    pub fn cuda(&self, deewce: Option<i32>) -> Self {
        unimplemented!()
    }
    pub fn cuda_async(&self, device: Option<i32>) -> Self {
        unimplemented!()
    }
    pub fn diag(&self, diag: u32) -> Self {
        unimplemented!()
    }
    pub fn dim(&self) -> i32 {
        unimplemented!()
    }
    pub fn dist(&self, other: &Self, p: u32) -> f32 {
        unimplemented!()
    }
    pub fn div(&self, value: T) -> Self {

        unimplemented!()
    }
    pub fn div_(self, value: T) -> Self {
        unimplemented!()
    }
    pub fn divt(&self, value: &Self) -> Self {

        unimplemented!()
    }
    pub fn divt_(self, value: &Self) -> Self {
        unimplemented!()
    }
    pub fn dot(&self, other: &Self) -> Self {
        unimplemented!()
    }
    pub fn double(&self) -> Self {
        unimplemented!()
    }
    pub fn eig(&self, eigenvectors: bool) -> (Self, Self) {
        unimplemented!()
    }
    pub fn element_size(&self) -> i32 {
        unimplemented!()
    }
    pub fn eq_tensor(&self, other: &Self) -> Tensor<u8> {
        unimplemented!()
    }
    pub fn eq_tensor_(&self, other: &Self) -> Self {
        unimplemented!()
    }
    pub fn exp(&self) -> Self {
        unimplemented!()
    }
    pub fn exp_(self) -> Self {
        unimplemented!()
    }
    pub fn expand(&self, dims: &[u32]) -> Self {
        unimplemented!()
    }
    pub fn expand_as(&self, tensor: &Self) -> Self {
        unimplemented!()
    }
    pub fn fill_(self) -> Self {
        unimplemented!()
    }
    pub fn float(self) -> Self {
        unimplemented!()
    }
    pub fn floor(&self) -> Self {
        unimplemented!()
    }
    pub fn floor_(self) -> Self {
        unimplemented!()
    }
    pub fn fmod(&self, divisor: T) -> Self {
        unimplemented!()
    }
    pub fn fmod_(self, divisor: T) -> Self {
        unimplemented!()
    }
    pub fn frac(&self) -> Self {
        unimplemented!()
    }
    pub fn frac_(self) -> Self {
        unimplemented!()
    }
    pub fn gather(&self, dim: i32, index: Tensor<i64>) {
        unimplemented!()
    }
    pub fn ge_tensor(&self, other: &Self) -> Tensor<u8> {
        unimplemented!()
    }
    pub fn ge_tensor_(&self, other: &Self) -> Self {
        unimplemented!()
    }
    pub fn gels(&self, other: &Self) -> Self {
        unimplemented!();
    }
    pub fn gt_tensor(&self, other: &Self) -> Tensor<u8> {
        unimplemented!()
    }
    pub fn gt_tensor_(&self, other: &Self) -> Self {
        unimplemented!()
    }
    pub fn half(self) -> Self {
        unimplemented!()
    }
    pub fn id(&self) -> usize {
        self.data_borrow().inner() as usize
    }
    pub fn index_masked(&self, m: &Tensor<u8>) -> Self {
        unimplemented!()
    }
    pub fn index_add_(self, dim: i32, index: Tensor<i64>, tensor: &Self) -> Self {
        unimplemented!()
    }
    pub fn index_copy_(self, dim: i32, index: Tensor<i64>, tensor: &Self) -> Self {
        unimplemented!()
    }
    pub fn index_fill_(self, dim: i32, index: Tensor<i64>, val: f32) -> Self {
        unimplemented!()
    }
    pub fn index_select(&self, dim: i32, index: Tensor<i64>) -> Self {
        unimplemented!()
    }
    pub fn inner_impl(&self) -> RefMut<TIArg<T>> {
        self.data_borrow().value.borrow_mut()
    }
    pub fn inner(&self) -> *mut ::std::os::raw::c_void {
        unimplemented!()
    }
    pub fn int(self) -> Self {
        unimplemented!()
    }
    pub fn is_cuda(&self) -> bool {
        unimplemented!()
    }
    pub fn is_pinned(&self) -> bool {
        unimplemented!()
    }
    pub fn is_set_to(&self, tensor: &Self) -> bool {
        unimplemented!()
    }
    pub fn is_signed(&self) -> bool {
        unimplemented!()
    }
    pub fn kthvalue(&self, k: i32, dim: Option<i32>) -> (Self, Tensor<i64>) {
        unimplemented!()
    }
    pub fn le_tensor(&self, other: &Self) -> Tensor<u8> {
        unimplemented!()
    }
    pub fn le_tensor_(&self, other: &Self) -> Self {
        unimplemented!()
    }
    pub fn lerp(&self, start: &Self, end: &Self, weight: f32) -> Self {
        unimplemented!()
    }
    pub fn lerp_(&self, start: &Self, end: &Self, weight: f32) -> Self {
        unimplemented!()
    }
    pub fn log(&self) -> Self {
        unimplemented!()
    }
    pub fn log_(self) -> Self {
        unimplemented!()
    }
    pub fn log1p(&self) -> Self {
        unimplemented!()
    }
    pub fn log1p_(self) -> Self {
        unimplemented!()
    }
    //
    // log_normal(...)
    //
    pub fn long(self) -> Self {
        unimplemented!()
    }
    pub fn lt_tensor(&self, other: &Self) -> Tensor<u8> {
        unimplemented!()
    }
    pub fn lt_tensor_(self, other: &Self) -> Self {
        unimplemented!()
    }
    //
    // map_
    //
    pub fn masked_copy_(self, mask: Tensor<u8>, source: &Self) -> Self {
        unimplemented!()
    }
    pub fn masked_select(&self, mask: Tensor<u8>) -> Self {
        unimplemented!()
    }
    pub fn max(&self) -> T {
        unimplemented!()
    }
    pub fn max_reduce(&self, dim: i32) -> (Self, Tensor<i64>) {
        unimplemented!()
    }
    pub fn mean(&self) -> T {
        unimplemented!()
    }
    pub fn mean_reduce(&self, dim: i32) -> (Self, Tensor<i64>) {
        unimplemented!()
    }
    //
    // median
    //
    pub fn min(&self) -> T {
        unimplemented!()
    }
    pub fn min_reduce(&self, dim: i32) -> (Self, Tensor<i64>) {
        unimplemented!()
    }
    pub fn mm(&self, rhs: &Self) -> Self {
        unimplemented!()
    }
    //
    // mode
    //
    pub fn mul(&self, rhs: T) -> Self {
        unimplemented!()
    }
    pub fn mul_(self, rhs: T) -> Self {
        unimplemented!()
    }
    pub fn mult(&self, rhs: &Self) -> Self {
        unimplemented!()
    }
    pub fn mult_(self, rhs: &Self) -> Self {
        unimplemented!()
    }
    //
    // multinomial
    //
    pub fn mv(&self, vec: &Self) -> Self {
        unimplemented!()
    }
    pub fn narrow(&self, dim: i32, start: i32, length: i32) -> Self {
        unimplemented!()
    }
    pub fn ndimension(&self) -> i32 {
        unimplemented!()
    }
    pub fn ne_tensor(&self, other: &Self) -> Tensor<u8> {
        unimplemented!()
    }
    pub fn ne_tensor_(self, other: &Self) -> Self {
        unimplemented!()
    }
    pub fn neg(&self) -> Self {
        unimplemented!()
    }
    pub fn neg_(self) -> Self {
        unimplemented!()
    }
    pub fn nonzero(&self) -> Tensor<i64> {
        unimplemented!()
    }
    pub fn norm(&self, p: i32) -> f32 {
        unimplemented!()
    }
    //
    // normal_
    //
    pub fn numel(&self) -> usize {
        self.data_borrow().numel()
    }
    //
    // numpy() (need native tensor equivalent - rust-ndarray?)
    //
    //
    // orgqr
    //
    // ormqr
    //
    pub fn permute(&self, dims: &[u32]) -> Self {
        unimplemented!()
    }
    pub fn pin_memory(&mut self) -> Self {
        unimplemented!()
    }
    //
    // potrf
    //
    //
    // potri
    //
    //
    // potrs
    //
    pub fn pow(&self) -> Self {
        unimplemented!()
    }
    pub fn pow_(self) -> Self {
        unimplemented!()
    }
    pub fn prod(&self) -> f32 {
        unimplemented!()
    }
    //
    // pstrf
    //
    //
    // qr
    //
    //
    // random_
    //
    pub fn reciprocal(&self) -> Self {
        unimplemented!()
    }
    pub fn reciprocal_(self) -> Self {
        unimplemented!()
    }
    pub fn remainder(&self, divisor: T) -> Self {
        unimplemented!()
    }
    pub fn remainder_(self, divisor: T) -> Self {
        unimplemented!()
    }
    //
    // renorm
    //
    //
    // renorm_
    //
    pub fn repeat(&self, sizes: &[i32]) -> Self {
        // NB: copies data
        unimplemented!()
    }
    pub fn resize_(self, sizes: &[i32]) -> Self {
        unimplemented!()
    }
    pub fn resize_as_(self, tensor: &Self) -> Self {
        unimplemented!()
    }
    pub fn round(&self) -> Self {
        unimplemented!()
    }
    pub fn round_(self) -> Self {
        unimplemented!()
    }
    pub fn rsqrt(&self) -> Self {
        unimplemented!()
    }
    pub fn rsqrt_(self) -> Self {
        unimplemented!()
    }
    pub fn scatter_(self, dim: i32, index: Tensor<i64>, src: &Self) -> Self {
        unimplemented!()
    }
    pub fn select(&self, dim: i32, index: i32) -> Self {
        unimplemented!()
    }
    //
    // set_
    //
    //
    // share_memory_
    //
    pub fn short(self) -> Self {
        unimplemented!()
    }
    pub fn sigmoid(&self) -> Self {
        unimplemented!()
    }
    pub fn sigmoid_(self) -> Self {
        unimplemented!()
    }
    pub fn sign(&self) -> Self {
        unimplemented!()
    }
    pub fn sign_(self) -> Self {
        unimplemented!()
    }
    pub fn sin(&self) -> Self {
        unimplemented!()
    }
    pub fn sin_(self) -> Self {
        unimplemented!()
    }
    pub fn sinh(&self) -> Self {
        unimplemented!()
    }
    pub fn sinh_(self) -> Self {
        unimplemented!()
    }
    pub fn size(&self) -> Vec<usize> {
        unimplemented!()
    }
    pub fn sort(&self, dim: Option<i32>, descending: bool) -> (Self, Tensor<i64>) {
        unimplemented!()
    }
    pub fn sqrt(&self) -> Self {
        unimplemented!()
    }
    pub fn sqrt_(self) -> Self {
        unimplemented!()
    }
    pub fn squeeze(&self, dim: Option<i32>) -> Self {
        unimplemented!()
    }
    pub fn squeeze_(self, dim: Option<i32>) -> Self {
        unimplemented!()
    }
    pub fn std(&self) -> f32 {
        unimplemented!()
    }
    //
    // storage
    //
    //
    // storage_offset
    //
    pub fn stride(&self) -> Vec<i32> {
        unimplemented!()
    }
    pub fn sub(&self, rhs: &Self) -> Self {
        unimplemented!()
    }
    pub fn sub_(self, rhs: &Self) -> Self {
        unimplemented!()
    }
    pub fn sum(&self) -> f32 {
        unimplemented!()
    }
    pub fn sum_reduce(&self, dim: i32, keepdim: bool) -> Self {
        unimplemented!()
    }
    pub fn svd(&self, some: bool) -> (Self, Self, Self) {
        unimplemented!()
    }
    //
    // symeig
    //
    pub fn t(&self) -> Self {
        unimplemented!()
    }
    pub fn t_(self) -> Self {
        unimplemented!()
    }
    pub fn tan(&self) -> Self {
        unimplemented!()
    }
    pub fn tan_(self) -> Self {
        unimplemented!()
    }
    pub fn tanh(&self) -> Self {
        unimplemented!()
    }
    pub fn tanh_(self) -> Self {
        unimplemented!()
    }
    //
    // tolist
    //
    pub fn topk(k: i32, dim: Option<i32>, largest: bool, sorted: bool) -> (Self, Tensor<i64>) {
        unimplemented!()
    }
    pub fn trace(&self) -> Self {
        unimplemented!()
    }
    pub fn transpose(&self, dim0: i32, dim1: i32) -> Self {
        unimplemented!()
    }
    pub fn transpose_(&self, dim0: i32, dim1: i32) -> Self {
        unimplemented!()
    }
    //
    // tril
    //
    //
    // tril_
    //
    //
    // triu
    //
    //
    // tril_
    //
    //
    // trtrs
    //
    pub fn trunc(&self) -> Self {
        unimplemented!()
    }
    pub fn trunc_(self) -> Self {
        unimplemented!()
    }
    pub fn type_as(&self, tensor: &Self) -> Self {
        unimplemented!()
    }
    pub fn typecast(&self, new_type: TensorType, async: bool) -> Self {
        unimplemented!()
    }
    pub fn unfold(&self, dim: i32, size: i32, step: i32) -> Self {
        unimplemented!()
    }
    pub fn uniform_(self, range: (i32, i32)) -> Self {
        unimplemented!()
    }
    pub fn unsqueeze(&self, dim: i32) -> Self {
        unimplemented!()
    }
    pub fn unsqueeze_(self, dim: i32) -> Self {
        unimplemented!()
    }
    pub fn var(&self) -> f32 {
        unimplemented!()
    }
    pub fn view<D>(&self, dims: D) -> Self
        where D: AsRef<[isize]>
    {
        use autograd::{View, ViewArgs};
        let mut input = vec![self.clone().into()];
        let args = ViewArgs { dims: dims.as_ref().to_vec() };
        View::new(&args).f(&mut input).remove(0).into()
    }
    pub fn view_as(&self, tensor: &Self) -> Self {
        let dims: Vec<isize> = tensor.size().iter().map(|v| *v as isize).collect();
        self.view(dims)
    }
    pub fn zero_(mut self) -> Self {
        self.data().zero_();
        self
    }
}


impl VarKind {
    pub fn abs(&self) -> Self {
        unimplemented!()
    }
    pub fn abs_(self) -> Self {
        unimplemented!()
    }
    pub fn acos(&self) -> Self {
        unimplemented!()
    }
    pub fn acos_(self) -> Self {
        unimplemented!()
    }
    pub fn add<T: NumLimits>(&self, rhs: T) -> Self {
        let mut v = self.copy();
        unimplemented!();
        //v.data().clone().add_(&rhs);
        v
    }
    pub fn add_<T: NumLimits>(self, rhs: T) -> Self {
        //self.data_borrow().clone().add_(rhs);
        self
    }
    pub fn addt<T: NumLimits>(&self, val: T, rhs: &Self) -> Self {
        let mut v = self.copy();
        //v.data().clone().addt_(val, rhs.data_borrow());
        v
    }
    pub fn addt_<T: NumLimits>(&mut self, val: T, rhs: &Self) -> &mut Self {
        self.data_borrow().addt_(val, &rhs.data_borrow());
        self
    }
    pub fn addbmm<T: NumLimits>(&self, beta: T, alpha: T, tensor1: &Self, tensor2: &Self) -> Self {
        unimplemented!()
    }
    pub fn addbmm_<T: NumLimits>(self, beta: T, alpha: T, tensor1: &Self, tensor2: &Self) -> Self {
        unimplemented!()
    }
    pub fn addcdiv<T: NumLimits>(&self, value: T, tensor1: &Self, tensor2: &Self) -> Self {
        unimplemented!()
    }
    pub fn addcdiv_<T: NumLimits>(self, value: T, tensor1: &Self, tensor2: &Self) -> Self {
        unimplemented!()
    }
    pub fn addcmul<T: NumLimits>(&self, value: T, tensor1: &Self, tensor2: &Self) -> Self {
        unimplemented!()
    }
    pub fn addcmul_<T: NumLimits>(self, value: T, tensor1: &Self, tensor2: &Self) -> Self {
        unimplemented!()
    }
    pub fn addmm<T: NumLimits>(&self, beta: T, alpha: T, tensor1: &Self, tensor2: &Self) -> Self {
        unimplemented!()
    }
    pub fn addmm_<T: NumLimits>(self, beta: T, alpha: T, tensor1: &Self, tensor2: &Self) -> Self {
        unimplemented!()
    }
    pub fn addmv<T: NumLimits>(&self, beta: T, alpha: T, tensor1: &Self, vec: &Self) -> Self {
        unimplemented!()
    }
    pub fn addmv_<T: NumLimits>(self, beta: T, alpha: T, tensor1: &Self, vec: &Self) -> Self {
        unimplemented!()
    }
    pub fn addr<T: NumLimits>(&self, beta: T, alpha: T, vec1: &Self, vec2: &Self) -> Self {
        unimplemented!()
    }
    pub fn addr_<T: NumLimits>(self, beta: T, alpha: T, vec1: &Self, vec2: &Self) -> Self {
        unimplemented!()
    }
    pub fn asin(&self) -> Self {
        unimplemented!()
    }
    pub fn asin_(self) -> Self {
        unimplemented!()
    }
    pub fn atan(&self) -> Self {
        unimplemented!()
    }
    pub fn atan2(&self) -> Self {
        unimplemented!()
    }
    pub fn atan2_(self) -> Self {
        unimplemented!()
    }
    pub fn baddbmm<T: NumLimits>(&self, beta: T, alpha: T, tensor1: &Self, tensor2: &Self) -> Self {
        unimplemented!()
    }
    pub fn baddbmm_<T: NumLimits>(self, beta: T, alpha: T, tensor1: &Self, tensor2: &Self) -> Self {
        unimplemented!()
    }
    pub fn bernoulli<T: NumLimits>(&self, p: T) -> Self {
        unimplemented!()
    }
    pub fn bernoulli_<T: NumLimits>(self, p: T) -> Self {
        unimplemented!()
    }
    pub fn bmm(&self, other: &Self) -> Self {
        unimplemented!()
    }
    pub fn byte(self) -> Self {
        unimplemented!()
    }
    //
    // cauchy_
    //
    pub fn ceil(&self) -> Self {
        unimplemented!()
    }
    pub fn ceil_(self) -> Self {
        unimplemented!()
    }
    pub fn char(self) -> Self {
        unimplemented!()
    }
    pub fn chunk(&self, n_chunks: usize, dim: usize) -> Vec<Self> {
        unimplemented!()
    }
    pub fn clamp(&self, min: f32, max: f32) -> Self {
        unimplemented!()
    }
    pub fn clamp_(self, min: f32, max: f32) -> Self {
        unimplemented!()
    }
    pub fn contiguous(&self) -> Self {
        unimplemented!()
    }
    // perform deep copy
    pub fn copy(&self) -> Self {
        unimplemented!();
        /*
        let new = Variable::new_args(self.data_borrow().copy(), &VariableArgs::default());
        let (inner, new_inner) = (self.access(), new.access());
        new_inner.copy_refs(inner);
        new
*/
    }
    pub fn copy_(self, src: &Self) -> Self {
        unimplemented!()
    }
    pub fn copy_async_(self, src: &Self) -> Self {
        unimplemented!()
    }
    pub fn cos(&self) -> Self {
        unimplemented!()
    }
    pub fn cos_(self) -> Self {
        unimplemented!()
    }
    pub fn cosh(&self) -> Self {
        unimplemented!()
    }
    pub fn cosh_(self) -> Self {
        unimplemented!()
    }
    pub fn cpu(&self) -> Self {
        unimplemented!()
    }
    pub fn cross(&self, dim: Option<i32>) -> Self {
        unimplemented!()
    }
    pub fn cuda(&self, device: Option<i32>) -> Self {
        unimplemented!()
    }
    pub fn cuda_async(&self, device: Option<i32>) -> Self {
        unimplemented!()
    }
    pub fn diag(&self, diag: u32) -> Self {
        unimplemented!()
    }
    pub fn dim(&self) -> i32 {
        unimplemented!()
    }
    pub fn dist(&self, other: &Self, p: u32) -> f32 {
        unimplemented!()
    }
    pub fn div<T: NumLimits>(&self, value: T) -> Self {

        unimplemented!()
    }
    pub fn div_<T: NumLimits>(self, value: T) -> Self {
        unimplemented!()
    }
    pub fn divt(&self, value: &Self) -> Self {

        unimplemented!()
    }
    pub fn divt_(self, value: &Self) -> Self {
        unimplemented!()
    }
    pub fn dot(&self, other: &Self) -> Self {
        unimplemented!()
    }
    pub fn double(&self) -> Self {
        unimplemented!()
    }
    pub fn eig(&self, eigenvectors: bool) -> (Self, Self) {
        unimplemented!()
    }
    pub fn element_size(&self) -> i32 {
        unimplemented!()
    }
    pub fn eq_tensor(&self, other: &Self) -> Tensor<u8> {
        unimplemented!()
    }
    pub fn eq_tensor_(&self, other: &Self) -> Self {
        unimplemented!()
    }
    pub fn exp(&self) -> Self {
        unimplemented!()
    }
    pub fn exp_(self) -> Self {
        unimplemented!()
    }
    pub fn expand(&self, dims: &[u32]) -> Self {
        unimplemented!()
    }
    pub fn expand_as(&self, tensor: &Self) -> Self {
        unimplemented!()
    }
    pub fn fill_(self) -> Self {
        unimplemented!()
    }
    pub fn float(self) -> Self {
        unimplemented!()
    }
    pub fn floor(&self) -> Self {
        unimplemented!()
    }
    pub fn floor_(self) -> Self {
        unimplemented!()
    }
    pub fn fmod<T: NumLimits>(&self, divisor: T) -> Self {
        unimplemented!()
    }
    pub fn fmod_<T: NumLimits>(self, divisor: T) -> Self {
        unimplemented!()
    }
    pub fn frac(&self) -> Self {
        unimplemented!()
    }
    pub fn frac_(self) -> Self {
        unimplemented!()
    }
    pub fn gather(&self, dim: i32, index: Tensor<i64>) {
        unimplemented!()
    }
    pub fn ge_tensor(&self, other: &Self) -> Tensor<u8> {
        unimplemented!()
    }
    pub fn ge_tensor_(&self, other: &Self) -> Self {
        unimplemented!()
    }
    pub fn gels(&self, other: &Self) -> Self {
        unimplemented!();
    }
    pub fn gt_tensor(&self, other: &Self) -> Tensor<u8> {
        unimplemented!()
    }
    pub fn gt_tensor_(&self, other: &Self) -> Self {
        unimplemented!()
    }
    pub fn half(self) -> Self {
        unimplemented!()
    }
    pub fn id(&self) -> usize {
        self.data_borrow().inner() as usize
    }
    pub fn index_masked(&self, m: &Tensor<u8>) -> Self {
        unimplemented!()
    }
    pub fn index_add_(self, dim: i32, index: Tensor<i64>, tensor: &Self) -> Self {
        unimplemented!()
    }
    pub fn index_copy_(self, dim: i32, index: Tensor<i64>, tensor: &Self) -> Self {
        unimplemented!()
    }
    pub fn index_fill_(self, dim: i32, index: Tensor<i64>, val: f32) -> Self {
        unimplemented!()
    }
    pub fn index_select(&self, dim: i32, index: Tensor<i64>) -> Self {
        unimplemented!()
    }
    pub fn inner_impl<T: NumLimits>(&self) -> RefMut<TIArg<T>> {
        unimplemented!()
        //self.data_borrow().value.borrow_mut()
    }
    pub fn inner(&self) -> *mut ::std::os::raw::c_void {
        unimplemented!()
    }
    pub fn int(self) -> Self {
        unimplemented!()
    }
    pub fn is_cuda(&self) -> bool {
        unimplemented!()
    }
    pub fn is_pinned(&self) -> bool {
        unimplemented!()
    }
    pub fn is_set_to(&self, tensor: &Self) -> bool {
        unimplemented!()
    }
    pub fn is_signed(&self) -> bool {
        unimplemented!()
    }
    pub fn kthvalue(&self, k: i32, dim: Option<i32>) -> (Self, Tensor<i64>) {
        unimplemented!()
    }
    pub fn le_tensor(&self, other: &Self) -> Tensor<u8> {
        unimplemented!()
    }
    pub fn le_tensor_(&self, other: &Self) -> Self {
        unimplemented!()
    }
    pub fn lerp(&self, start: &Self, end: &Self, weight: f32) -> Self {
        unimplemented!()
    }
    pub fn lerp_(&self, start: &Self, end: &Self, weight: f32) -> Self {
        unimplemented!()
    }
    pub fn log(&self) -> Self {
        unimplemented!()
    }
    pub fn log_(self) -> Self {
        unimplemented!()
    }
    pub fn log1p(&self) -> Self {
        unimplemented!()
    }
    pub fn log1p_(self) -> Self {
        unimplemented!()
    }
    //
    // log_normal(...)
    //
    pub fn long(self) -> Self {
        unimplemented!()
    }
    pub fn lt_tensor(&self, other: &Self) -> Tensor<u8> {
        unimplemented!()
    }
    pub fn lt_tensor_(self, other: &Self) -> Self {
        unimplemented!()
    }
    //
    // map_
    //
    pub fn masked_copy_(self, mask: Tensor<u8>, source: &Self) -> Self {
        unimplemented!()
    }
    pub fn masked_select(&self, mask: Tensor<u8>) -> Self {
        unimplemented!()
    }
    pub fn max<T: NumLimits>(&self) -> T {
        unimplemented!()
    }
    pub fn max_reduce(&self, dim: i32) -> (Self, Tensor<i64>) {
        unimplemented!()
    }
    pub fn mean<T: NumLimits>(&self) -> T {
        unimplemented!()
    }
    pub fn mean_reduce(&self, dim: i32) -> (Self, Tensor<i64>) {
        unimplemented!()
    }
    //
    // median
    //
    pub fn min<T: NumLimits>(&self) -> T {
        unimplemented!()
    }
    pub fn min_reduce(&self, dim: i32) -> (Self, Tensor<i64>) {
        unimplemented!()
    }
    pub fn mm(&self, rhs: &Self) -> Self {
        unimplemented!()
    }
    //
    // mode
    //
    pub fn mul<T: NumLimits>(&self, rhs: T) -> Self {
        unimplemented!()
    }
    pub fn mul_<T: NumLimits>(self, rhs: T) -> Self {
        unimplemented!()
    }
    pub fn mult(&self, rhs: &Self) -> Self {
        unimplemented!()
    }
    pub fn mult_(self, rhs: &Self) -> Self {
        unimplemented!()
    }
    //
    // multinomial
    //
    pub fn mv(&self, vec: &Self) -> Self {
        unimplemented!()
    }
    pub fn narrow(&self, dim: i32, start: i32, length: i32) -> Self {
        unimplemented!()
    }
    pub fn ndimension(&self) -> i32 {
        unimplemented!()
    }
    pub fn ne_tensor(&self, other: &Self) -> Tensor<u8> {
        unimplemented!()
    }
    pub fn ne_tensor_(self, other: &Self) -> Self {
        unimplemented!()
    }
    pub fn neg(&self) -> Self {
        unimplemented!()
    }
    pub fn neg_(self) -> Self {
        unimplemented!()
    }
    pub fn nonzero(&self) -> Tensor<i64> {
        unimplemented!()
    }
    pub fn norm(&self, p: i32) -> f32 {
        unimplemented!()
    }
    //
    // normal_
    //
    pub fn numel(&self) -> i32 {
        unimplemented!()
    }
    //
    // numpy() (need native tensor equivalent - rust-ndarray?)
    //
    //
    // orgqr
    //
    // ormqr
    //
    pub fn permute(&self, dims: &[u32]) -> Self {
        unimplemented!()
    }
    pub fn pin_memory(&mut self) -> Self {
        unimplemented!()
    }
    //
    // potrf
    //
    //
    // potri
    //
    //
    // potrs
    //
    pub fn pow(&self) -> Self {
        unimplemented!()
    }
    pub fn pow_(self) -> Self {
        unimplemented!()
    }
    pub fn prod(&self) -> f32 {
        unimplemented!()
    }
    //
    // pstrf
    //
    //
    // qr
    //
    //
    // random_
    //
    pub fn reciprocal(&self) -> Self {
        unimplemented!()
    }
    pub fn reciprocal_(self) -> Self {
        unimplemented!()
    }
    pub fn remainder<T: NumLimits>(&self, divisor: T) -> Self {
        unimplemented!()
    }
    pub fn remainder_<T: NumLimits>(self, divisor: T) -> Self {
        unimplemented!()
    }
    //
    // renorm
    //
    //
    // renorm_
    //
    pub fn repeat(&self, sizes: &[i32]) -> Self {
        // NB: copies data
        unimplemented!()
    }
    pub fn resize_(self, sizes: &[i32]) -> Self {
        unimplemented!()
    }
    pub fn resize_as_(self, tensor: &Self) -> Self {
        unimplemented!()
    }
    pub fn round(&self) -> Self {
        unimplemented!()
    }
    pub fn round_(self) -> Self {
        unimplemented!()
    }
    pub fn rsqrt(&self) -> Self {
        unimplemented!()
    }
    pub fn rsqrt_(self) -> Self {
        unimplemented!()
    }
    pub fn scatter_(self, dim: i32, index: Tensor<i64>, src: &Self) -> Self {
        unimplemented!()
    }
    pub fn select(&self, dim: i32, index: i32) -> Self {
        unimplemented!()
    }
    //
    // set_
    //
    //
    // share_memory_
    //
    pub fn short(self) -> Self {
        unimplemented!()
    }
    pub fn sigmoid(&self) -> Self {
        unimplemented!()
    }
    pub fn sigmoid_(self) -> Self {
        unimplemented!()
    }
    pub fn sign(&self) -> Self {
        unimplemented!()
    }
    pub fn sign_(self) -> Self {
        unimplemented!()
    }
    pub fn sin(&self) -> Self {
        unimplemented!()
    }
    pub fn sin_(self) -> Self {
        unimplemented!()
    }
    pub fn sinh(&self) -> Self {
        unimplemented!()
    }
    pub fn sinh_(self) -> Self {
        unimplemented!()
    }
    pub fn size(&self) -> Vec<usize> {
        unimplemented!()
    }
    pub fn sort(&self, dim: Option<i32>, descending: bool) -> (Self, Tensor<i64>) {
        unimplemented!()
    }
    pub fn sqrt(&self) -> Self {
        unimplemented!()
    }
    pub fn sqrt_(self) -> Self {
        unimplemented!()
    }
    pub fn squeeze(&self, dim: Option<i32>) -> Self {
        unimplemented!()
    }
    pub fn squeeze_(self, dim: Option<i32>) -> Self {
        unimplemented!()
    }
    pub fn std(&self) -> f32 {
        unimplemented!()
    }
    //
    // storage
    //
    //
    // storage_offset
    //
    pub fn stride(&self) -> Vec<i32> {
        unimplemented!()
    }
    pub fn sub(&self, rhs: &Self) -> Self {
        unimplemented!()
    }
    pub fn sub_(self, rhs: &Self) -> Self {
        unimplemented!()
    }
    pub fn sum(&self) -> f32 {
        unimplemented!()
    }
    pub fn sum_reduce(&self, dim: i32, keepdim: bool) -> Self {
        unimplemented!()
    }
    pub fn svd(&self, some: bool) -> (Self, Self, Self) {
        unimplemented!()
    }
    //
    // symeig
    //
    pub fn t(&self) -> Self {
        unimplemented!()
    }
    pub fn t_(self) -> Self {
        unimplemented!()
    }
    pub fn tan(&self) -> Self {
        unimplemented!()
    }
    pub fn tan_(self) -> Self {
        unimplemented!()
    }
    pub fn tanh(&self) -> Self {
        unimplemented!()
    }
    pub fn tanh_(self) -> Self {
        unimplemented!()
    }
    //
    // tolist
    //
    pub fn topk(k: i32, dim: Option<i32>, largest: bool, sorted: bool) -> (Self, Tensor<i64>) {
        unimplemented!()
    }
    pub fn trace(&self) -> Self {
        unimplemented!()
    }
    pub fn transpose(&self, dim0: i32, dim1: i32) -> Self {
        unimplemented!()
    }
    pub fn transpose_(&self, dim0: i32, dim1: i32) -> Self {
        unimplemented!()
    }
    //
    // tril
    //
    //
    // tril_
    //
    //
    // triu
    //
    //
    // tril_
    //
    //
    // trtrs
    //
    pub fn trunc(&self) -> Self {
        unimplemented!()
    }
    pub fn trunc_(self) -> Self {
        unimplemented!()
    }
    pub fn type_as(&self, tensor: &Self) -> Self {
        unimplemented!()
    }
    pub fn typecast(&self, new_type: TensorType, async: bool) -> Self {
        unimplemented!()
    }
    pub fn unfold(&self, dim: i32, size: i32, step: i32) -> Self {
        unimplemented!()
    }
    pub fn uniform_(self, range: (i32, i32)) -> Self {
        unimplemented!()
    }
    pub fn unsqueeze(&self, dim: i32) -> Self {
        unimplemented!()
    }
    pub fn unsqueeze_(self, dim: i32) -> Self {
        unimplemented!()
    }
    pub fn var(&self) -> f32 {
        unimplemented!()
    }
    pub fn view<D>(&self, dims: D) -> Self
        where D: AsRef<[isize]>
    {
        unimplemented!()
    }
    pub fn view_as(&self, tensor: &Self) -> Self {
        unimplemented!()
    }
    pub fn zero_(self) -> Self {
        unimplemented!()
    }
}
