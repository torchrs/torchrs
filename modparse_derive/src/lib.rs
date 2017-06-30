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
fn is_type(ty: &syn::Ty, name: &'static str) -> bool {
    if let syn::Ty::Path(_, ref path) = *ty {
        if path.segments.iter().any(|seg| seg.ident.as_ref() == name) {
            true
        } else {
            false
        }
    } else {
        false
    }
}

fn option_type(ty: &syn::Ty) -> &syn::Ty {
    if let syn::Ty::Path(_, ref path) = *ty {
        if let Some(seg) = path.segments
               .iter()
               .find(|seg| seg.ident.as_ref() == "Option") {
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
    } else if is_type(&ty, "Parameter") {
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
    // XXX figure out how to avoid this 
    let used_variants0 = variants.iter().filter(|field| {
            if field.attrs.iter().any(|attr| attr.name() == "ignore") { false}
            else if is_type(& field.ty, "Module") {false}
            else {true}
    });
    let used_variants1 = variants.iter().filter(|field| {
            if field.attrs.iter().any(|attr| attr.name() == "ignore") { false}
            else if is_type(& field.ty, "Module") {false}
            else {true}
    });
    let used_variants2 = variants.iter().filter(|field| {
            if field.attrs.iter().any(|attr| attr.name() == "ignore") { false}
            else if is_type(& field.ty, "Module") {false}
            else {true}
    });
    let atmatch = used_variants0
        .filter_map(|field| {
            let field_name = field.ident.as_ref().clone();
            let is_option = is_type(&field.ty, "Option");
            match get_type(&field.ty) {
                Kind::Parameter if is_option => 
                    Some( quote! { 
                        if let Some(ref mut param) = self. #field_name {
                            self.delegate.add_param(stringify!(#field_name));
                        };
                    } ),
                Kind::Parameter => 
                    Some( quote! { self.delegate.add_param(stringify!(#field_name))  } ),
                Kind::ModIntf if is_option => 
                Some ( quote! {
                    if let Some(ref module) = self. #field_name {
                        self.delegate.add_module(stringify!(#field_name)); 
                    };
                } ),
                Kind::ModIntf => 
                    Some ( quote! { self.delegate.add_module(stringify!(#field_name) ); } ),
                _ => panic!("bad match {:?}", field.ty),
            }
        });
    let param_match = used_variants1
        .filter_map(|field| {
            let field_name = field.ident.as_ref().clone();
            let is_option = is_type(&field.ty, "Option");
            match get_type(&field.ty) {
                Kind::Parameter if is_option => 
                    Some( quote! { 
                        stringify!(#field_name) => {
                        if let Some(ref mut param) = self. #field_name {
                            Some(param.v.id)
                        } else { None }}, 
                    } ),
                Kind::Parameter => 
                    Some( quote! { stringify!(#field_name) => Some(self. #field_name . v.id), } ),
                Kind::ModIntf =>  None,
                _ => panic!("bad match {:?}", field.ty),
            }

        });
    let module_match = used_variants2
        .filter_map(|field| {
            let field_name = field.ident.as_ref().clone();
            match get_type(&field.ty) {
                Kind::Parameter => None,
                Kind::ModIntf => 
                    Some( quote! { stringify!(#field_name) => &mut self. #field_name,  } ),
                _ => panic!("bad match {:?}", field.ty),
            }
        });
    let foo = quote! {
        impl #impl_generics InitModuleStruct for #name  #ty_generics  #where_clause {
            fn init_module(mut self) -> Self {
                self.delegate()._name = stringify!(#name).into();
                #(#atmatch);* 
                ;
                self
            }
        }
        impl #impl_generics GetFieldStruct<T> for #name  #ty_generics  #where_clause {
            fn get_param(&mut self, name: &str) -> Option<i32> {
                match name {
                    #(#param_match)*
                    _ => panic!("unknown Parameter {}", name),
                }
            } 
            fn get_module(&mut self, name: &str) -> &mut ModIntf<T> {
                match name {
                    #(#module_match)*
                    _ => {panic!("not found {}", name); },
                }
            } 
        }

    };
    foo
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {}
}
