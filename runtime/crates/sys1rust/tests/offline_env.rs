//! `HF_HUB_OFFLINE` turns downloads off for `serve` and `pull` only when huggingface_hub would
//! read it as true. Offline `pull` still prints the directory of a complete snapshot. One test
//! sets the variable in this process, so the binary checks give their children the value they
//! need instead of inheriting it.

use clap::Parser;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use sys1rust::cli::{offline_from_env, Cli, Command as Sub};
use sys1rust::models;

#[test]
fn hf_hub_offline_reads_like_huggingface_hub() {
    let read = |value: Option<&str>, args: &[&str]| {
        match value {
            Some(v) => std::env::set_var("HF_HUB_OFFLINE", v),
            None => std::env::remove_var("HF_HUB_OFFLINE"),
        }
        let all = ["sys1rust", "serve"]
            .into_iter()
            .chain(args.iter().copied());
        let Sub::Serve(c) = Cli::try_parse_from(all).unwrap().command else {
            panic!("not serve")
        };
        (c.offline(), offline_from_env())
    };
    assert_eq!(read(None, &[]), (false, false));
    assert_eq!(read(None, &["--offline"]), (true, false));
    for v in ["1", "on", "Yes", "TRUE"] {
        assert_eq!(read(Some(v), &[]), (true, true), "{v:?}");
    }
    for v in ["", "0", "2", " 1", "off", "abc"] {
        assert_eq!(read(Some(v), &[]), (false, false), "{v:?}");
        assert_eq!(read(Some(v), &["--offline"]), (true, false), "{v:?}");
    }
    std::env::remove_var("HF_HUB_OFFLINE");
}

/// `sys1rust <args>` with `HF_HUB_OFFLINE=<offline>`, an empty cache and an endpoint on a
/// closed port, so a download fails at once without touching the network.
fn spawn(args: &[&str], offline: &str, cache: &Path) -> Child {
    Command::new(env!("CARGO_BIN_EXE_sys1rust"))
        .args(args)
        .env("HF_HUB_OFFLINE", offline)
        .env("HF_ENDPOINT", "http://127.0.0.1:9")
        .env("HF_HUB_CACHE", cache)
        .env_remove("HF_TOKEN")
        .env_remove("SYS1_MODEL")
        .env_remove("SYS1_REVISION")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap()
}

/// Exit code, stdout and stderr of a child that stops by itself.
fn finish(child: Child) -> (Option<i32>, String, String) {
    let out = child.wait_with_output().unwrap();
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stdout).into(),
        String::from_utf8_lossy(&out.stderr).into(),
    )
}

/// A pinned typed-decisions snapshot that `LayaModel::status` calls complete, as in
/// closed_stdout.rs. The status check reads sizes only, so the blobs are sparse files of the
/// manifest sizes. Returns the snapshot directory.
fn fake_typed_decisions(cache: &Path) -> PathBuf {
    let model = models::find("typed-decisions").unwrap();
    let snapshot = model.snapshot_dir(cache);
    for file in model.files {
        let blob = model.blob_path(cache, file);
        fs::create_dir_all(blob.parent().unwrap()).unwrap();
        File::create(&blob).unwrap().set_len(file.size).unwrap();
        let up = "../".repeat(2 + file.path.matches('/').count());
        let link = snapshot.join(file.path);
        fs::create_dir_all(link.parent().unwrap()).unwrap();
        symlink(format!("{up}blobs/{}", file.blob), &link).unwrap();
    }
    assert_eq!(model.status(cache), models::Status::Complete);
    snapshot
}

/// stderr lines up to the first that contains `needle`, then kill the child. The downloader
/// backs off 7 s in all before it gives up on a file, so the test does not wait for that.
fn lines_until(mut child: Child, needle: &str) -> Vec<String> {
    let mut lines = Vec::new();
    for line in BufReader::new(child.stderr.take().unwrap()).lines() {
        let line = line.unwrap();
        let found = line.contains(needle);
        lines.push(line);
        if found {
            break;
        }
    }
    let _ = child.kill();
    child.wait().unwrap();
    lines
}

#[test]
fn a_true_value_stops_pull_and_serve_downloads() {
    let cache = tempfile::tempdir().unwrap();
    let (code, stdout, stderr) = finish(spawn(&["pull", "typed-decisions"], "1", cache.path()));
    assert_eq!(code, Some(1), "{stderr}");
    assert!(
        stderr.contains(
            "sys1rust: error: HF_HUB_OFFLINE is on, so nothing is downloaded; unset it to pull typed-decisions"
        ),
        "{stderr}"
    );
    assert_eq!(stdout, "");
    let serve = ["serve", "--model", "typed-decisions", "--port", "0"];
    let (code, _, stderr) = finish(spawn(&serve, "ON", cache.path()));
    assert_eq!(code, Some(1), "{stderr}");
    assert!(stderr.contains("downloads are off"), "{stderr}");
}

#[test]
fn offline_pull_prints_a_complete_snapshot_and_refuses_a_partial_one() {
    let cache = tempfile::tempdir().unwrap();
    let snapshot = fake_typed_decisions(cache.path());
    let (code, stdout, stderr) = finish(spawn(&["pull", "typed-decisions"], "1", cache.path()));
    assert_eq!(code, Some(0), "{stderr}");
    assert_eq!(stdout, format!("{}\n", snapshot.display()));
    assert_eq!(stderr, "");
    fs::remove_file(snapshot.join("tokenizer/tokenizer.json")).unwrap();
    let (code, stdout, stderr) = finish(spawn(&["pull", "typed-decisions"], "1", cache.path()));
    assert_eq!(code, Some(1), "{stderr}");
    assert!(stderr.contains("HF_HUB_OFFLINE is on"), "{stderr}");
    assert_eq!(stdout, "");
}

#[test]
fn a_value_huggingface_hub_reads_as_false_downloads() {
    let cache = tempfile::tempdir().unwrap();
    let lines = lines_until(
        spawn(&["pull", "typed-decisions"], "2", cache.path()),
        "retrying",
    );
    let last = lines.last().map(String::as_str).unwrap_or_default();
    assert!(
        last.contains("GET http://127.0.0.1:9/") && last.contains("retrying"),
        "{lines:#?}"
    );
    assert!(
        !lines.iter().any(|l| l.contains("HF_HUB_OFFLINE")),
        "{lines:#?}"
    );
    let serve = ["serve", "--model", "typed-decisions", "--port", "0"];
    let lines = lines_until(spawn(&serve, " 1", cache.path()), "downloading");
    let last = lines.last().map(String::as_str).unwrap_or_default();
    assert!(
        last.starts_with("sys1rust: downloading typed-decisions"),
        "{lines:#?}"
    );
}
