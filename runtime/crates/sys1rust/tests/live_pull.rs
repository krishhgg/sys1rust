//! Live download test (ignored by default; needs network and the GPU). `sys1rust pull`
//! fetches typed-decisions from Hugging Face into an empty cache, `models` then lists it as
//! downloaded, and `serve --offline` loads it from that cache and answers the first request
//! of `bench/workloads/smoke.jsonl` like the fp32 reference. The answer has the same choice
//! unless the reference is a near tie, the same probability labels, and every probability
//! within 0.05. A guard kills the server if the test fails or hangs at any point.
//!
//! Run from the repository root with:
//! `cargo test --manifest-path runtime/Cargo.toml -p sys1rust --release --test live_pull -- --ignored --nocapture`

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

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
/// Model load is about 3 s; a ready line later than this means a hang.
const READY_TIMEOUT: Duration = Duration::from_secs(120);
/// One request takes milliseconds; a stall past this fails the test instead of hanging it.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);

fn sys1rust(cache: &std::path::Path) -> Command {
    let mut cmd = Command::new(BIN);
    for var in CLEAR {
        cmd.env_remove(var);
    }
    cmd.env("HF_HUB_CACHE", cache);
    cmd
}

/// The `sys1rust serve` child. Dropping it kills and reaps the process if it is still
/// running, so a failed assertion or a timeout never leaves a server holding GPU memory.
struct Server(Child);

impl Server {
    /// The first stdout line, within `READY_TIMEOUT`. A reader thread keeps draining stdout
    /// afterwards so the child never blocks on a full pipe.
    fn ready_line(&mut self) -> String {
        let stdout = self.0.stdout.take().expect("piped stdout");
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let mut lines = BufReader::new(stdout).lines();
            if let Some(Ok(first)) = lines.next() {
                let _ = tx.send(first);
            }
            for _ in lines.by_ref() {}
        });
        match rx.recv_timeout(READY_TIMEOUT) {
            Ok(line) => line,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                panic!("no ready line within {READY_TIMEOUT:?}")
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                panic!(
                    "sys1rust exited before printing a ready line: {:?}",
                    self.0.wait()
                )
            }
        }
    }

    /// SIGINT, then the exit status within `SHUTDOWN_TIMEOUT`.
    fn stop(&mut self) -> ExitStatus {
        let status = Command::new("kill")
            .args(["-INT", &self.0.id().to_string()])
            .status()
            .unwrap();
        assert!(status.success(), "kill -INT");
        let deadline = Instant::now() + SHUTDOWN_TIMEOUT;
        loop {
            if let Some(st) = self.0.try_wait().unwrap() {
                return st;
            }
            assert!(
                Instant::now() < deadline,
                "sys1rust did not exit within {SHUTDOWN_TIMEOUT:?} of SIGINT"
            );
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        if let Ok(None) = self.0.try_wait() {
            eprintln!(
                "killing sys1rust (pid {}) that is still running",
                self.0.id()
            );
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
}

/// The `probabilities` of an answer by label, empty when it has none. Every value must be a
/// number, so a `null` cannot pass as 0.
fn probs(q: &str, answer: &serde_json::Value) -> BTreeMap<String, f64> {
    let Some(p) = answer.get("probabilities") else {
        return BTreeMap::new();
    };
    p.as_object()
        .unwrap_or_else(|| panic!("{q}: probabilities is not an object: {answer}"))
        .iter()
        .map(|(k, v)| {
            let x = v
                .as_f64()
                .unwrap_or_else(|| panic!("{q} {k}: {v} is not a number"));
            (k.clone(), x)
        })
        .collect()
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
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let table = String::from_utf8(out.stdout).unwrap();
    let row = table
        .lines()
        .find(|l| l.starts_with("typed-decisions "))
        .unwrap_or_else(|| panic!("no typed-decisions row: {table}"));
    // The status column follows the revision and at least 2 spaces.
    let status = row.rsplit_once("  ").map(|(_, s)| s.trim());
    assert_eq!(status, Some("downloaded, 846 MB"), "{table}");

    let mut server = Server(
        sys1rust(cache.path())
            .args(["serve", "--offline", "--port", "0"])
            .stdout(Stdio::piped())
            .spawn()
            .unwrap(),
    );
    let line = server.ready_line();
    let ready: serde_json::Value =
        serde_json::from_str(&line).unwrap_or_else(|e| panic!("ready line {line:?}: {e}"));
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
    conn.set_read_timeout(Some(REQUEST_TIMEOUT)).unwrap();
    conn.set_write_timeout(Some(REQUEST_TIMEOUT)).unwrap();
    write!(
        conn,
        "POST /v1/systemone HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut resp = String::new();
    conn.read_to_string(&mut resp).unwrap();

    let exit = server.stop();
    assert!(resp.starts_with("HTTP/1.1 200"), "{resp}");
    assert_eq!(exit.code(), Some(0), "clean shutdown exit code");

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
        let (got_p, want_p) = (probs(q, got), probs(q, want));
        assert_eq!(
            got_p.keys().collect::<Vec<_>>(),
            want_p.keys().collect::<Vec<_>>(),
            "{q}: probability labels"
        );
        for (k, w) in &want_p {
            let g = got_p[k];
            assert!((g - w).abs() <= 0.05, "{q} {k}: {g} vs reference {w}");
        }
        let mut want_p: Vec<(String, f64)> = want_p.into_iter().collect();
        want_p.sort_by(|a, b| b.1.total_cmp(&a.1));
        let near_tie = want_p.len() > 1 && want_p[0].1 - want_p[1].1 < 0.05;
        if !want["choice"].is_null() && !near_tie {
            assert_eq!(got["choice"], want["choice"], "{q}");
        }
    }
}
