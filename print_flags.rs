fn main() {
    let build = cc::Build::new();
    let compiler = build.get_compiler();
    println!("CC: {:?}", compiler.path());
    for flag in compiler.args() {
        println!("FLAG: {:?}", flag);
    }
}
