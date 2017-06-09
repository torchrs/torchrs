use std::process::Command;


fn main() {
    let output = Command::new("python")
        .arg(format!("{}/scripts/generate_wrappers.py",
                     env!("CARGO_MANIFEST_DIR")))
        .spawn()
        .expect("failed to execute process");
    println!("{:?}", output)
}
