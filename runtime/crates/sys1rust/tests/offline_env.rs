//! `HF_HUB_OFFLINE` turns downloads off for `serve` and `pull` only when huggingface_hub would
//! read it as true. One test sets the variable in this process, so the binary checks give
//! their children the value they need instead of inheriting it.

use clap::Parser;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use sys1rust::cli::{offline_from_env, Cli, Command as Sub};

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
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap()
}

/// Exit code and stderr of a child that stops by itself.
fn finish(child: Child) -> (Option<i32>, String) {
    let out = child.wait_with_output().unwrap();
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stderr).into(),
    )
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
    let (code, stderr) = finish(spawn(&["pull", "typed-decisions"], "1", cache.path()));
    assert_eq!(code, Some(1), "{stderr}");
    assert!(
        stderr.contains("sys1rust: error: HF_HUB_OFFLINE is on"),
        "{stderr}"
    );
    let serve = ["serve", "--model", "typed-decisions", "--port", "0"];
    let (code, stderr) = finish(spawn(&serve, "ON", cache.path()));
    assert_eq!(code, Some(1), "{stderr}");
    assert!(stderr.contains("downloads are off"), "{stderr}");
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
