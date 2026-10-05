//! `models` and `pull` against a stdout whose reader is already gone. Both exit 0 without a
//! panic, the way a pipe into `head` ends. The cache holds a complete fake snapshot and the
//! endpoint is a closed port, so nothing touches the network.

use std::fs::{self, File};
use std::os::unix::fs::symlink;
use std::path::Path;
use std::process::{Command, Stdio};
use sys1rust::models;

/// A pinned typed-decisions snapshot that `LayaModel::status` calls complete. The status check
/// reads sizes only, so the blobs are sparse files of the manifest sizes.
fn fake_typed_decisions(cache: &Path) {
    let model = models::find("typed-decisions").unwrap();
    let snapshot = model.snapshot_dir(cache);
    for file in model.files {
        let blob = model.blob_path(cache, file);
        fs::create_dir_all(blob.parent().unwrap()).unwrap();
        File::create(&blob).unwrap().set_len(file.size).unwrap();
        // huggingface_hub links `snapshots/<rev>/<path>` to `../../blobs/<blob>`, with one more
        // `../` for each directory in `<path>`.
        let up = "../".repeat(2 + file.path.matches('/').count());
        let link = snapshot.join(file.path);
        fs::create_dir_all(link.parent().unwrap()).unwrap();
        symlink(format!("{up}blobs/{}", file.blob), &link).unwrap();
    }
    assert_eq!(model.status(cache), models::Status::Complete);
}

/// Run `sys1rust <args>` with stdout on a pipe whose read end is closed before the start.
fn run_into_closed_pipe(args: &[&str], cache: &Path) -> (Option<i32>, String) {
    let (reader, writer) = std::io::pipe().unwrap();
    drop(reader);
    let out = Command::new(env!("CARGO_BIN_EXE_sys1rust"))
        .args(args)
        .env_remove("HF_HUB_OFFLINE")
        .env_remove("HF_TOKEN")
        .env("HF_ENDPOINT", "http://127.0.0.1:9")
        .env("HF_HUB_CACHE", cache)
        .stdin(Stdio::null())
        .stdout(writer)
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stderr).into(),
    )
}

#[test]
fn models_exits_0_when_stdout_is_closed() {
    let cache = tempfile::tempdir().unwrap();
    fake_typed_decisions(cache.path());
    let (code, stderr) = run_into_closed_pipe(&["models"], cache.path());
    assert_eq!(code, Some(0), "{stderr}");
    assert!(!stderr.contains("panicked"), "{stderr}");
}

#[test]
fn pull_exits_0_when_stdout_is_closed() {
    let cache = tempfile::tempdir().unwrap();
    fake_typed_decisions(cache.path());
    let (code, stderr) = run_into_closed_pipe(&["pull", "typed-decisions"], cache.path());
    assert_eq!(code, Some(0), "{stderr}");
    assert!(!stderr.contains("panicked"), "{stderr}");
}
