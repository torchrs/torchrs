extern crate bindgen;

use std::env;
use std::path::PathBuf;

fn main() {
    let torch_path = env::var("TORCH_PATH").expect("TORCH_PATH not defined");
    println!("cargo:rustc-link-search=native={}/torch/lib/build/TH", torch_path);
    println!("cargo:rustc-link-lib=TH");

    //let cuda_path = env::var("CUDA_PATH").expect("CUDA_PATH not defined");
    //let mut inc = String::from("-I");
    //inc.push_str(&cuda_path);
    //inc.push_str("/include");
    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()

        // Do not generate unstable Rust code that
        // requires a nightly rustc and enabling
        // unstable features.
        .no_unstable_rust()
        .clang_arg(format!("-I{}/torch/lib/include/TH", torch_path))
 	    .clang_arg(format!("-I{}/torch/lib", torch_path))
        //.clang_arg("-Ipytorch/torch/lib/include/THC")
        //.clang_arg(inc)
        // The input header we would like to generate
        // bindings for.
        .header("src/wrapper.h")
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

}