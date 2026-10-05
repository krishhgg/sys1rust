//! Live download test (ignored by default; needs network and the GPU). `sys1rust pull`
//! fetches typed-decisions from Hugging Face into an empty cache, `models` then lists it as
//! downloaded, and `serve --offline` loads it from that cache and answers the first request
//! of `bench/workloads/smoke.jsonl` like the fp32 reference: the same choice unless the
//! reference is a near tie, and every probability within 0.05.
//!
//! Run from the repository root with:
//! `cargo test --manifest-path runtime/Cargo.toml -p sys1rust --release --test live_pull -- --ignored --nocapture`

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

const BIN: &str = env!("CARGO_BIN_EXE_sys1rust");
/// Variables that would change what the child does.
const CLEAR: [&str; 10] = [
    "HF_HUB_OFFLINE",
    "HF_ENDPOINT",
    "HF_HOME",
    "SYS1_MODEL",
    "SYS1_REVISION",
    "SYS1_MLX_TUNING",
    "SYS1_F32",
    "LAYA_API_KEY",
    "LAYA_HOST",
    "LAYA_PORT",
];

fn sys1rust(cache: &std::path::Path) -> Command {
    let mut cmd = Command::new(BIN);
    for var in CLEAR {
        cmd.env_remove(var);
    }
    cmd.env("HF_HUB_CACHE", cache);
    cmd
}

#[test]
#[ignore = "downloads 846 MB from Hugging Face and loads the model on the GPU"]
fn pull_then_serve_from_an_empty_cache() {
    let cache = tempfile::tempdir().unwrap();
    let out = sys1rust(cache.path()).args(["pull"]).output().unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let snap = cache
        .path()
        .join("models--convaiinnovations--laya-typed-decisions/snapshots/1a793eb568e6718f15941d08f85432581df534e3");
    assert_eq!(
        String::from_utf8(out.stdout).unwrap().trim(),
        snap.to_str().unwrap()
    );

    let out = sys1rust(cache.path()).arg("models").output().unwrap();
    let table = String::from_utf8(out.stdout).unwrap();
    assert!(table.contains("downloaded, 846 MB"), "{table}");

    let mut child = sys1rust(cache.path())
        .args(["serve", "--offline", "--port", "0"])
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let stdout = child.stdout.take().unwrap();
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut lines = BufReader::new(stdout).lines();
        if let Some(Ok(first)) = lines.next() {
            let _ = tx.send(first);
        }
        for _ in lines.by_ref() {}
    });
    let ready = rx.recv_timeout(Duration::from_secs(120));
    let ready: serde_json::Value = match ready {
        Ok(line) => serde_json::from_str(&line).unwrap(),
        Err(e) => {
            let _ = child.kill();
            panic!("no ready line: {e:?}");
        }
    };
    let addr = ready["addr"].as_str().unwrap().to_string();

    let smoke = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../bench/workloads/smoke.jsonl"
    );
    let first: serde_json::Value = serde_json::from_str(
        std::fs::read_to_string(smoke)
            .unwrap()
            .lines()
            .next()
            .unwrap(),
    )
    .unwrap();
    let mut body = first["body"].clone();
    body["model"] = "typed-decisions".into();
    let body = body.to_string();
    let mut conn = TcpStream::connect(&addr).unwrap();
    write!(
        conn,
        "POST /v1/systemone HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut resp = String::new();
    conn.read_to_string(&mut resp).unwrap();

    let _ = Command::new("kill")
        .args(["-INT", &child.id().to_string()])
        .status();
    let status = child.wait().unwrap();
    assert!(resp.starts_with("HTTP/1.1 200"), "{resp}");
    assert_eq!(status.code(), Some(0));

    let (_, json) = resp.split_once("\r\n\r\n").unwrap();
    let answers = &serde_json::from_str::<serde_json::Value>(json).unwrap()["answers"];
    let reference = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../bench/reference/typed-decisions/smoke.jsonl"
    );
    let want = std::fs::read_to_string(reference)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap())
        .find(|r| r["type"] == "result" && r["id"] == first["id"])
        .expect("a reference row for the first smoke request");
    for (q, want) in want["answers"].as_object().unwrap() {
        let got = &answers[q];
        assert_eq!(got["type"], want["type"], "{q}: {got} vs {want}");
        let probs = |a: &serde_json::Value| -> Vec<(String, f64)> {
            a["probabilities"]
                .as_object()
                .map(|p| {
                    p.iter()
                        .map(|(k, v)| (k.clone(), v.as_f64().unwrap()))
                        .collect()
                })
                .unwrap_or_default()
        };
        let (got_p, mut want_p) = (probs(got), probs(want));
        for (k, w) in &want_p {
            let g = got_p
                .iter()
                .find(|(gk, _)| gk == k)
                .map_or(0.0, |(_, g)| *g);
            assert!((g - w).abs() <= 0.05, "{q} {k}: {g} vs reference {w}");
        }
        want_p.sort_by(|a, b| b.1.total_cmp(&a.1));
        let near_tie = want_p.len() > 1 && want_p[0].1 - want_p[1].1 < 0.05;
        if !want["choice"].is_null() && !near_tie {
            assert_eq!(got["choice"], want["choice"], "{q}");
        }
    }
}
