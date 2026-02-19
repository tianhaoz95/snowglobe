fn main() {
    #[cfg(target_os = "android")]
    println!("cargo:rustc-link-lib=c++_shared");

    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=framework=SystemConfiguration");
        println!("cargo:rustc-link-lib=framework=CoreFoundation");
        println!("cargo:rustc-link-lib=framework=Security");
        // This fixes the std::logic_error / C++ symbols
        println!("cargo:rustc-link-lib=c++");
    }
}