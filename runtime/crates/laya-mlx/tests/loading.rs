//! Loading errors with the loading settings, on the real checkpoints (ignored by default;
//! needs them in the HF cache, `source bench/env.sh` first). A checkpoint whose
//! `model.safetensors` is missing, empty, or cut short (inside the header, or 1 MiB in, inside
//! the data) fails to load with an `Err` that names the file, never a panic: on the round 2
//! default, with each of `directload`, `sharehead` and `parallel_load`, and with the three
//! together. The bit equality of what these settings load is checked in `tests/settings.rs`.
//! A checkpoint that does not resolve is skipped with a note (with
//! `SYS1_TEST_ALL_CHECKPOINTS=1`, it fails the test).
//!
//! Each broken checkpoint is a directory of links to the cached one's files plus the broken
//! weights file. The same directory with the cached weights linked back loads, so the error
//! comes from the weights file alone.
//!
//! Run from the repo root with:
//! `cargo test --manifest-path runtime/Cargo.toml -p laya-mlx --release --test loading -- --ignored --nocapture`

mod common;

use common::{assert_ran, checkpoint_dir, CHECKPOINTS};
use laya_core::{Agent, BackendOptions};
use std::io::Read;
use std::path::{Path, PathBuf};

/// sys1d's round 2 default (`DEFAULT_TUNING`).
const DEFAULT: &str = "f16gelu,cache=512,wired=2048,dense_upto=1024,headprune,unpad,fuserope";
/// Each loading setting on top of [`DEFAULT`], none, and the three together.
const LOADING: [&str; 5] = ["", "directload", "sharehead", "parallel_load", "directload,sharehead,parallel_load"];

/// A checkpoint directory under the system temp dir with links to every entry of a cached one
/// but `model.safetensors`, which the test writes. Removed on drop.
struct Broken(PathBuf);

impl Broken {
    fn new(src: &Path, name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("laya-mlx-loading-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for entry in std::fs::read_dir(src).unwrap() {
            let entry = entry.unwrap();
            if entry.file_name() != "model.safetensors" {
                std::os::unix::fs::symlink(entry.path(), dir.join(entry.file_name())).unwrap();
            }
        }
        Self(dir)
    }
}

impl Drop for Broken {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Load `dir` with `DEFAULT` plus `extra`; a panic fails the test with the setting named.
fn try_load(dir: &Path, extra: &str) -> laya_core::Result<Agent> {
    let tuning = if extra.is_empty() { DEFAULT.to_string() } else { format!("{DEFAULT},{extra}") };
    let opts = BackendOptions { tuning: Some(tuning), ..Default::default() };
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Agent::load(dir, &opts, Box::new(laya_mlx::make_backend))
    }))
    .unwrap_or_else(|_| panic!("{}: loading with `{extra}` panicked", dir.display()))
}

#[test]
#[ignore]
fn missing_or_truncated_weights_are_an_error() {
    let mut ran = 0;
    for (name, repo) in CHECKPOINTS {
        let Some(src) = checkpoint_dir(name, repo) else {
            continue;
        };
        // Absolute, so the links in the temp dir resolve when HF_HUB_CACHE is relative.
        let src = src.canonicalize().unwrap_or_else(|e| panic!("{name}: {}: {e}", src.display()));
        let weights = src.join("model.safetensors");
        let size = std::fs::metadata(&weights).unwrap().len() as usize;
        let head = |n: usize| {
            let mut buf = Vec::with_capacity(n);
            std::fs::File::open(&weights).unwrap().take(n as u64).read_to_end(&mut buf).unwrap();
            buf
        };
        let header_end = 8 + u64::from_le_bytes(head(8).try_into().unwrap()) as usize;
        let mib = 1 << 20;
        assert!(header_end < mib && mib < size, "{name}: header ends at {header_end}, file is {size} bytes");
        let not_found = "weights: 'model.safetensors' not found in";
        let cut = "bytes) is truncated or not a safetensors file";
        let cases: [(&str, Option<usize>, &str); 4] = [
            ("missing", None, not_found),
            ("empty", Some(0), cut),
            ("cut_in_header", Some(header_end / 2), cut),
            ("cut_in_data", Some(mib), cut),
        ];
        for (case, len, want) in cases {
            let dir = Broken::new(&src, &format!("{name}-{case}"));
            if let Some(len) = len {
                std::fs::write(dir.0.join("model.safetensors"), head(len)).unwrap();
            }
            for extra in LOADING {
                let e = try_load(&dir.0, extra).unwrap_err().to_string();
                assert!(e.contains(want), "{name}/{case}/`{extra}`: {e}");
                if let Some(len) = len {
                    assert!(e.contains(&format!("model.safetensors ({len} {cut}")), "{name}/{case}/`{extra}`: {e}");
                }
            }
            eprintln!("{name:<16} {case:<14} every setting: {}", try_load(&dir.0, "").unwrap_err());
        }
        // The control: the same links with the cached weights load.
        let dir = Broken::new(&src, &format!("{name}-control"));
        std::os::unix::fs::symlink(&weights, dir.0.join("model.safetensors")).unwrap();
        try_load(&dir.0, LOADING[4]).unwrap_or_else(|e| panic!("{name}: the control does not load: {e}"));
        ran += 1;
    }
    assert_ran(ran);
}
