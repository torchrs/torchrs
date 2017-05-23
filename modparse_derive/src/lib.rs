#![crate_type = "proc-macro"]
#![recursion_limit = "192"]
#![feature(trace_macros)]
#![feature(log_syntax)]


extern crate syn;
extern crate proc_macro;
#[macro_use]
extern crate quote;

use proc_macro::TokenStream;

enum Kind {
    Module,
    Parameter,
    ModIntf,
}

#[proc_macro_derive(ModParse, attributes(ignore))]
pub fn parse(input: TokenStream) -> TokenStream {
    let source = input.to_string();
    let mut ast = syn::parse_derive_input(&source).expect("failed to parse rust syntax");
    let gen = impl_parse(&mut ast);
    gen.parse().expect("failed to serialize into rust syntax")
}
fn is_type(ty: &syn::Ty, name: & 'static str) -> bool {
    if let syn::Ty::Path(_, ref path) = *ty {
        if path.segments.iter().any(|seg| seg.ident.as_ref() == name) { 
            true  
        } else {
            false
        }
    } else {false}
}

fn option_type(ty: &syn::Ty) -> &syn::Ty {
    if let syn::Ty::Path(_, ref path) = *ty {
        if let Some(seg) = 
            path.segments.iter().find(|seg| seg.ident.as_ref() == "Option") {
            if let syn::PathParameters::AngleBracketed(ref data) = seg.parameters {
            // assume simple nesting for now
                &data.types[0]
            } else {
                panic!("can't parse Option segment parameters {:?}", seg.parameters);
            }
        } else {
            panic!("Option not found!")
        } 
    } else {
        panic!("not path")
    }
}

fn get_type(ty: &syn::Ty) -> Kind {
        if is_type(&ty, "Option") {
            get_type(option_type(ty))
        } else if is_type(&ty, "Module") {
            Kind::Module
        } else if is_type(&ty, "Parameter"){
            Kind::Parameter
        } else {
            Kind::ModIntf
        }
}

fn impl_parse(ast: &mut syn::DeriveInput) -> quote::Tokens {
    use syn::{Body, VariantData};

    let ref mut variants = match ast.body {
        Body::Struct(VariantData::Struct(ref vars)) => vars,
        _ => panic!("#[derive(Parse)] is only defined for braced structs"),
    };

    let name = &ast.ident;
    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();
    let atmatch = variants.iter()
        .filter_map(|field| {
            if field.attrs.iter().any(|attr| attr.name() == "ignore") ||
                is_type(& field.ty, "Module")
            { return None;}
        	let field_name = field.ident.as_ref().clone();
            let is_option = is_type(&field.ty, "Option");
            get_type(&field.ty);
            match get_type(&field.ty) {
                Kind::Parameter if is_option => 
                    Some( quote! { 
                        if let Some(ref mut param) = self. #field_name {
                            self.delegate.add_param(stringify!(#field_name), param);
                        };
                    } ),
                Kind::Parameter => 
                    Some( quote! { self.delegate.add_param(stringify!(#field_name), &mut self. #field_name)  } ),
                Kind::ModIntf if is_option => 
                Some ( quote! {
                    if let Some(ref module) = self. #field_name {
                        self.delegate.add_module(&mut module ); 
                    };
                } ),
                Kind::ModIntf => 
                    Some ( quote! { self.delegate.add_module(&mut self. #field_name ); } ),
                _ => panic!("bad match {:?}", field.ty),
        	}
        });
    let foo = quote! {
        impl #impl_generics ModuleStruct<'a> for #name  #ty_generics  #where_clause {
            fn init_module(&mut self) {
            	self.delegate._name = stringify!(#name);
				#(#atmatch);* 
				;
            }
        }
    };
    println!("parse is {}", foo);
    foo
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {}
}
