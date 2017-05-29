
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

#[macro_export]
macro_rules! impl_mod_delegate {
	($name:ident) => (
		impl<T:Default> ModDelegate<T> for $name<T> {
		    fn delegate(&mut self) -> &mut Module<T> {
    		    &mut self.delegate
    		}
		}

	)
}
