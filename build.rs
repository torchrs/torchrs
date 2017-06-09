use std::process::Command;


fn main() {
	let output = Command::new("python")
			.arg("scripts/generate_wrapper.py")
			.output()
			.expect("failed to execute process");
	println!("{:?}", output.stdout)
} 
