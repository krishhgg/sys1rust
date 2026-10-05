//! With a prebuilt MLX (`MLX_SYS_PREBUILT_DIR`, see vendor/mlx-sys/build.rs), libmlx.dylib is
//! loaded through @rpath, so every executable that links it needs an rpath to `<dir>/lib`.
//!
//! Cargo passes `cargo:rustc-link-arg` only to the targets of the package that emits it (here
//! this crate's tests and benches), never to a dependent's binaries, so each binary crate that
//! links laya-mlx (sys1rust, sys1-bench) carries the same build script instead of relying on this
//! one. The directory is canonicalized first: a relative `MLX_SYS_PREBUILT_DIR` would otherwise
//! become a relative rpath, resolved against the working directory of whoever runs the binary.
fn main() {
    println!("cargo:rerun-if-env-changed=MLX_SYS_PREBUILT_DIR");
    if let Some(dir) = std::env::var_os("MLX_SYS_PREBUILT_DIR") {
        let dir = std::path::Path::new(&dir).canonicalize().unwrap_or_else(|e| {
            panic!("MLX_SYS_PREBUILT_DIR={} cannot be resolved to an absolute path ({e})", dir.to_string_lossy())
        });
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.join("lib").display());
    }
}
