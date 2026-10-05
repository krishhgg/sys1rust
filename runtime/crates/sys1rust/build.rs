//! With a prebuilt MLX (vendor/mlx-sys), libmlx.dylib is loaded through @rpath. Also names
//! the build and its MLX version for --version.
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
    // `--version` names the release build; packaging/build.sh sets macos14 or macos26.
    println!("cargo:rerun-if-env-changed=SYS1_BUILD_FLAVOR");
    let flavor = std::env::var("SYS1_BUILD_FLAVOR")
        .ok()
        .filter(|f| !f.is_empty())
        .unwrap_or_else(|| "source".into());
    println!("cargo:rustc-env=SYS1RUST_BUILD_FLAVOR={flavor}");
    // vendor/mlx-sys/build.rs refuses any MLX but its MLX_VERSION, so --version reads it there.
    let mlx_build =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../vendor/mlx-sys/build.rs");
    println!("cargo:rerun-if-changed={}", mlx_build.display());
    let src = std::fs::read_to_string(&mlx_build)
        .unwrap_or_else(|e| panic!("read {}: {e}", mlx_build.display()));
    let version = src
        .lines()
        .find_map(|l| l.strip_prefix("const MLX_VERSION: &str = \""))
        .and_then(|rest| rest.split('"').next())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| {
            panic!(
                "{} has no `const MLX_VERSION: &str = \"...\";` line to read the MLX version from",
                mlx_build.display()
            )
        });
    println!("cargo:rustc-env=SYS1RUST_MLX_VERSION={version}");
}
