//! With a prebuilt MLX (vendor/mlx-sys), libmlx.dylib is loaded through @rpath.
fn main() {
    println!("cargo:rerun-if-env-changed=MLX_SYS_PREBUILT_DIR");
    if let Some(dir) = std::env::var_os("MLX_SYS_PREBUILT_DIR") {
        // The rpath is baked into the binary, so a relative MLX_SYS_PREBUILT_DIR would only
        // resolve from the directory cargo was run in. Make it absolute first.
        let dir = std::path::Path::new(&dir)
            .canonicalize()
            .unwrap_or_else(|e| {
                panic!(
                    "MLX_SYS_PREBUILT_DIR={:?} is not an existing directory: {e}",
                    dir
                )
            });
        let lib = dir.join("lib");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
    }
}
