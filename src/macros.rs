
#[macro_export]
macro_rules! impl_func_delegate {
	($name:ident) => (
		impl FuncDelegate for $name {
		    fn delegate(&mut self) -> &mut Function {
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
