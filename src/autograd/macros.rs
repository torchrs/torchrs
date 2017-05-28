use autograd::FuncDelegate;


#[macro_export]
macro_rules! impl_func_delegate {
	($name:ident) => (
		impl<T> FuncDelegate<T> for $name<T> {
		    fn delegate(&mut self) -> &mut Function<T> {
    		    &mut self.delegate
    		}
		}

	)
}
