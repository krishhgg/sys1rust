//! The downloader against local HTTP servers: the cache layout it writes, hash checks,
//! resuming, retries, where HF_TOKEN goes, snapshot links, and locking. No network access.

use sha1::Sha1;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use sys1rust::download::{blob_name_of, Hub, Progress};
use sys1rust::models::{incomplete_path, LayaModel, ModelFile, Status};

/// What the fake hub does for one path.
#[derive(Default, Clone)]
struct Route {
    body: Vec<u8>,
    /// 302 to this URL instead of a body.
    redirect: Option<String>,
    /// Answer 200 with the whole body even when the request has `Range`.
    ignore_range: bool,
    /// Each of the next responses (200 or 206) declares its full length but closes after
    /// sending this many body bytes.
    cuts: VecDeque<usize>,
    /// The next request gets this status and an empty body.
    fail_next: Option<u16>,
    /// Every request gets this status and an empty body.
    fail_always: Option<u16>,
    /// The next response is a 206 for the bytes from this offset, whatever the request asked
    /// for.
    stray_range: Option<usize>,
}

#[derive(Debug, Clone)]
struct Seen {
    path: String,
    host: String,
    range: Option<String>,
    auth: Option<String>,
}

/// A one-request-per-connection HTTP/1.1 server on a loopback address.
struct FakeHub {
    base: String,
    routes: Arc<Mutex<HashMap<String, Route>>>,
    seen: Arc<Mutex<Vec<Seen>>>,
}

impl FakeHub {
    /// `bind` is `127.0.0.1:0` or `[::1]:0`; two servers on the two give two hosts.
    fn start(bind: &str) -> FakeHub {
        let listener = TcpListener::bind(bind).unwrap();
        let addr = listener.local_addr().unwrap();
        let base = match addr {
            std::net::SocketAddr::V4(a) => format!("http://{a}"),
            std::net::SocketAddr::V6(a) => format!("http://[{}]:{}", a.ip(), a.port()),
        };
        let routes = Arc::new(Mutex::new(HashMap::new()));
        let seen = Arc::new(Mutex::new(Vec::new()));
        let (r, s) = (routes.clone(), seen.clone());
        std::thread::spawn(move || {
            for conn in listener.incoming().flatten() {
                let (r, s) = (r.clone(), s.clone());
                std::thread::spawn(move || handle(conn, &r, &s));
            }
        });
        FakeHub { base, routes, seen }
    }

    fn route(&self, path: &str, route: Route) {
        self.routes.lock().unwrap().insert(path.to_string(), route);
    }

    fn seen(&self) -> Vec<Seen> {
        self.seen.lock().unwrap().clone()
    }

    fn requests_for(&self, path: &str) -> Vec<Seen> {
        self.seen().into_iter().filter(|s| s.path == path).collect()
    }
}

fn handle(mut conn: TcpStream, routes: &Mutex<HashMap<String, Route>>, seen: &Mutex<Vec<Seen>>) {
    let mut reader = BufReader::new(conn.try_clone().unwrap());
    let mut line = String::new();
    if reader.read_line(&mut line).unwrap_or(0) == 0 {
        return;
    }
    let path = line.split_whitespace().nth(1).unwrap_or("").to_string();
    let mut headers = HashMap::new();
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h).unwrap_or(0) == 0 || h == "\r\n" {
            break;
        }
        if let Some((k, v)) = h.trim_end().split_once(':') {
            headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
        }
    }
    let range = headers.get("range").cloned();
    seen.lock().unwrap().push(Seen {
        path: path.clone(),
        host: headers.get("host").cloned().unwrap_or_default(),
        range: range.clone(),
        auth: headers.get("authorization").cloned(),
    });
    let mut routes = routes.lock().unwrap();
    let Some(route) = routes.get_mut(&path) else {
        return respond(&mut conn, 404, &[], b"");
    };
    if let Some(code) = route.fail_next.take().or(route.fail_always) {
        return respond(&mut conn, code, &[], b"");
    }
    if let Some(to) = &route.redirect {
        return respond(&mut conn, 302, &[("Location", to.clone())], b"");
    }
    let cut = route.cuts.pop_front();
    let body = route.body.clone();
    let start = route.stray_range.take().or_else(|| {
        range
            .as_deref()
            .and_then(|r| r.strip_prefix("bytes="))
            .and_then(|r| r.strip_suffix('-'))
            .and_then(|n| n.parse::<usize>().ok())
    });
    if let (Some(start), false) = (start, route.ignore_range) {
        let cr = format!("bytes {}-{}/{}", start, body.len() - 1, body.len());
        return respond_cut(
            &mut conn,
            206,
            &[("Content-Range", cr)],
            &body[start..],
            cut,
        );
    }
    respond_cut(&mut conn, 200, &[], &body, cut)
}

fn respond(conn: &mut TcpStream, code: u16, headers: &[(&str, String)], body: &[u8]) {
    respond_cut(conn, code, headers, body, None)
}

/// Declare `body.len()` bytes but send only the first `cut` of them when `cut` is set. The
/// connection closes when the handler returns.
fn respond_cut(
    conn: &mut TcpStream,
    code: u16,
    headers: &[(&str, String)],
    body: &[u8],
    cut: Option<usize>,
) {
    let mut head = format!(
        "HTTP/1.1 {code} X\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    for (k, v) in headers {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    head.push_str("\r\n");
    let _ = conn.write_all(head.as_bytes());
    let _ = conn.write_all(&body[..cut.unwrap_or(body.len()).min(body.len())]);
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn leak(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

/// A two-file model: weights named by sha256, as an LFS file, and a tokenizer in a subfolder
/// named by git blob sha1. Returns the model and both files' bytes.
fn tiny_model() -> (&'static LayaModel, Vec<u8>, Vec<u8>) {
    let weights: Vec<u8> = (0..300_000u32).map(|i| (i * 31 % 251) as u8).collect();
    let tokenizer = br#"{"version":"1.0"}"#.to_vec();
    let mut git = Sha1::new();
    git.update(format!("blob {}\0", tokenizer.len()));
    git.update(&tokenizer);
    let files: &'static [ModelFile] = Box::leak(Box::new([
        ModelFile {
            path: "model.safetensors",
            size: weights.len() as u64,
            blob: leak(hex(&Sha256::digest(&weights))),
        },
        ModelFile {
            path: "tokenizer/tokenizer.json",
            size: tokenizer.len() as u64,
            blob: leak(hex(&git.finalize())),
        },
    ]));
    let model = Box::leak(Box::new(LayaModel {
        name: "tiny",
        repo: "test/tiny",
        revision: "0123456789abcdef0123456789abcdef01234567",
        files,
    }));
    (model, weights, tokenizer)
}

fn url_path(m: &LayaModel, i: usize) -> String {
    format!("/{}/resolve/{}/{}", m.repo, m.revision, m.files[i].path)
}

fn serve_model(hub: &FakeHub, m: &LayaModel, weights: &[u8], tokenizer: &[u8]) {
    hub.route(
        &url_path(m, 0),
        Route {
            body: weights.to_vec(),
            ..Default::default()
        },
    );
    hub.route(
        &url_path(m, 1),
        Route {
            body: tokenizer.to_vec(),
            ..Default::default()
        },
    );
}

fn client(hub: &FakeHub) -> Hub {
    Hub::new(&hub.base, None).with_backoff(Duration::ZERO)
}

fn quiet() -> Progress<Vec<u8>> {
    Progress::new(Vec::new(), false)
}

#[test]
fn a_download_writes_the_huggingface_layout() {
    let (m, weights, tokenizer) = tiny_model();
    let hub = FakeHub::start("127.0.0.1:0");
    serve_model(&hub, m, &weights, &tokenizer);
    let cache = tempfile::tempdir().unwrap();
    let c = cache.path();
    let snap = client(&hub).ensure(c, m, &mut quiet()).unwrap();
    assert_eq!(
        snap,
        c.join("models--test--tiny/snapshots").join(m.revision)
    );
    assert_eq!(
        std::fs::read_link(snap.join("model.safetensors")).unwrap(),
        Path::new("../../blobs").join(m.files[0].blob)
    );
    assert_eq!(
        std::fs::read_link(snap.join("tokenizer/tokenizer.json")).unwrap(),
        Path::new("../../../blobs").join(m.files[1].blob)
    );
    assert_eq!(
        std::fs::read(snap.join("model.safetensors")).unwrap(),
        weights
    );
    assert_eq!(
        std::fs::read(snap.join("tokenizer/tokenizer.json")).unwrap(),
        tokenizer
    );
    let blobs: Vec<String> = std::fs::read_dir(c.join("models--test--tiny/blobs"))
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    assert_eq!(blobs.len(), 2, "no .incomplete left: {blobs:?}");
    let lock = c
        .join(".locks/models--test--tiny")
        .join(format!("{}.lock", m.files[0].blob));
    assert!(lock.exists(), "{}", lock.display());
    assert_eq!(m.status(c), Status::Complete);
}

#[test]
fn a_hash_mismatch_keeps_nothing() {
    let (m, weights, tokenizer) = tiny_model();
    let mut wrong = weights.clone();
    wrong[1000] ^= 1;
    let hub = FakeHub::start("127.0.0.1:0");
    serve_model(&hub, m, &wrong, &tokenizer);
    let cache = tempfile::tempdir().unwrap();
    let c = cache.path();
    let e = client(&hub).ensure(c, m, &mut quiet()).unwrap_err();
    let msg = format!("{e:#}");
    assert!(
        msg.contains(&format!("expected {}", m.files[0].blob)),
        "{msg}"
    );
    let blob = m.blob_path(c, &m.files[0]);
    assert!(!blob.exists() && !incomplete_path(&blob).exists());
    assert!(std::fs::symlink_metadata(m.snapshot_dir(c).join("model.safetensors")).is_err());
    // A download that started from byte 0 and hashes wrong is not tried again.
    assert_eq!(hub.requests_for(&url_path(m, 0)).len(), 1);
}

#[test]
fn a_cached_blob_with_wrong_bytes_is_downloaded_again() {
    let (m, weights, tokenizer) = tiny_model();
    let hub = FakeHub::start("127.0.0.1:0");
    serve_model(&hub, m, &weights, &tokenizer);
    let cache = tempfile::tempdir().unwrap();
    let c = cache.path();
    // A blob of the right size but the wrong bytes, and no snapshot link to it.
    let blob = m.blob_path(c, &m.files[0]);
    std::fs::create_dir_all(blob.parent().unwrap()).unwrap();
    std::fs::write(&blob, vec![7u8; weights.len()]).unwrap();
    let snap = client(&hub).ensure(c, m, &mut quiet()).unwrap();
    let seen = hub.requests_for(&url_path(m, 0));
    assert_eq!(seen.len(), 1, "{seen:?}");
    assert_eq!(seen[0].range, None);
    let link = snap.join("model.safetensors");
    assert_eq!(
        std::fs::read_link(&link).unwrap(),
        Path::new("../../blobs").join(m.files[0].blob)
    );
    assert_eq!(
        blob_name_of(&link, m.files[0].blob).unwrap(),
        m.files[0].blob
    );
    assert_eq!(std::fs::read(&link).unwrap(), weights);
}

/// Wrong bytes an earlier run left in `model.safetensors`'s `.incomplete`, shorter than the
/// file. Returns the cache's blob path.
fn seed_bad_partial(c: &Path, m: &LayaModel) -> PathBuf {
    let blob = m.blob_path(c, &m.files[0]);
    std::fs::create_dir_all(blob.parent().unwrap()).unwrap();
    std::fs::write(incomplete_path(&blob), vec![7u8; 1000]).unwrap();
    blob
}

#[test]
fn a_bad_partial_file_from_an_earlier_run_starts_over() {
    let (m, weights, tokenizer) = tiny_model();
    let hub = FakeHub::start("127.0.0.1:0");
    serve_model(&hub, m, &weights, &tokenizer);
    let cache = tempfile::tempdir().unwrap();
    let c = cache.path();
    let blob = seed_bad_partial(c, m);
    let snap = client(&hub).ensure(c, m, &mut quiet()).unwrap();
    // The GET resumes from the bad bytes, the hash fails, and one GET starts the file over.
    let seen = hub.requests_for(&url_path(m, 0));
    assert_eq!(seen.len(), 2, "{seen:?}");
    assert_eq!(seen[0].range.as_deref(), Some("bytes=1000-"));
    assert_eq!(seen[1].range, None);
    assert_eq!(
        blob_name_of(&blob, m.files[0].blob).unwrap(),
        m.files[0].blob
    );
    assert_eq!(
        std::fs::read(snap.join("model.safetensors")).unwrap(),
        weights
    );
    assert!(!incomplete_path(&blob).exists());
}

#[test]
fn a_bad_partial_file_starts_over_only_once() {
    let (m, weights, tokenizer) = tiny_model();
    let mut wrong = weights.clone();
    wrong[100_000] ^= 1;
    let hub = FakeHub::start("127.0.0.1:0");
    serve_model(&hub, m, &wrong, &tokenizer);
    let cache = tempfile::tempdir().unwrap();
    let c = cache.path();
    let blob = seed_bad_partial(c, m);
    let e = client(&hub).ensure(c, m, &mut quiet()).unwrap_err();
    let msg = format!("{e:#}");
    assert!(
        msg.contains(&format!("expected {}", m.files[0].blob)),
        "{msg}"
    );
    // The resumed GET, then 1 GET from byte 0 that also hashes wrong.
    let seen = hub.requests_for(&url_path(m, 0));
    assert_eq!(seen.len(), 2, "{seen:?}");
    assert_eq!(seen[0].range.as_deref(), Some("bytes=1000-"));
    assert_eq!(seen[1].range, None);
    assert!(!blob.exists() && !incomplete_path(&blob).exists());
}

#[test]
fn a_cut_download_resumes_with_range() {
    let (m, weights, tokenizer) = tiny_model();
    let hub = FakeHub::start("127.0.0.1:0");
    serve_model(&hub, m, &weights, &tokenizer);
    hub.route(
        &url_path(m, 0),
        Route {
            body: weights.clone(),
            cuts: VecDeque::from([100_000]),
            ..Default::default()
        },
    );
    let cache = tempfile::tempdir().unwrap();
    let snap = client(&hub).ensure(cache.path(), m, &mut quiet()).unwrap();
    let seen = hub.requests_for(&url_path(m, 0));
    assert_eq!(seen.len(), 2, "{seen:?}");
    assert_eq!(seen[0].range, None);
    let from: usize = seen[1]
        .range
        .as_deref()
        .and_then(|r| r.strip_prefix("bytes="))
        .and_then(|r| r.strip_suffix('-'))
        .and_then(|n| n.parse().ok())
        .unwrap_or_else(|| panic!("{seen:?}"));
    assert!(from > 0 && from <= 100_000, "{from}");
    assert_eq!(
        std::fs::read(snap.join("model.safetensors")).unwrap(),
        weights
    );
}

#[test]
fn a_server_that_ignores_range_restarts_the_file() {
    let (m, weights, tokenizer) = tiny_model();
    let hub = FakeHub::start("127.0.0.1:0");
    serve_model(&hub, m, &weights, &tokenizer);
    hub.route(
        &url_path(m, 0),
        Route {
            body: weights.clone(),
            ignore_range: true,
            ..Default::default()
        },
    );
    let cache = tempfile::tempdir().unwrap();
    let c = cache.path();
    let blob = m.blob_path(c, &m.files[0]);
    std::fs::create_dir_all(blob.parent().unwrap()).unwrap();
    std::fs::write(incomplete_path(&blob), vec![7u8; 1000]).unwrap();
    let snap = client(&hub).ensure(c, m, &mut quiet()).unwrap();
    let seen = hub.requests_for(&url_path(m, 0));
    assert_eq!(seen.len(), 1, "{seen:?}");
    assert_eq!(seen[0].range.as_deref(), Some("bytes=1000-"));
    assert_eq!(
        std::fs::read(snap.join("model.safetensors")).unwrap(),
        weights
    );
}

#[test]
fn a_download_that_keeps_moving_does_not_run_out_of_retries() {
    let (m, weights, tokenizer) = tiny_model();
    let hub = FakeHub::start("127.0.0.1:0");
    serve_model(&hub, m, &weights, &tokenizer);
    // Six responses in a row stop early, more than the 4 attempts allowed without progress,
    // but each one adds 40,000 bytes.
    hub.route(
        &url_path(m, 0),
        Route {
            body: weights.clone(),
            cuts: VecDeque::from([40_000; 6]),
            ..Default::default()
        },
    );
    let cache = tempfile::tempdir().unwrap();
    let snap = client(&hub).ensure(cache.path(), m, &mut quiet()).unwrap();
    assert_eq!(hub.requests_for(&url_path(m, 0)).len(), 7);
    assert_eq!(
        std::fs::read(snap.join("model.safetensors")).unwrap(),
        weights
    );
}

#[test]
fn a_server_that_ignores_range_and_stops_short_runs_out_of_retries() {
    let (m, weights, tokenizer) = tiny_model();
    let hub = FakeHub::start("127.0.0.1:0");
    serve_model(&hub, m, &weights, &tokenizer);
    // Every GET starts the file over. After the first, each one stops before byte 40,000, so
    // none gets further than the first did, even when it beats the GET before it.
    let cuts = std::iter::once(40_000)
        .chain([30_000, 35_000].repeat(10))
        .collect();
    hub.route(
        &url_path(m, 0),
        Route {
            body: weights.clone(),
            ignore_range: true,
            cuts,
            ..Default::default()
        },
    );
    let cache = tempfile::tempdir().unwrap();
    client(&hub)
        .ensure(cache.path(), m, &mut quiet())
        .unwrap_err();
    // The first GET, then 4 without progress.
    assert_eq!(hub.requests_for(&url_path(m, 0)).len(), 5);
}

#[test]
fn a_503_is_retried_and_a_404_is_not() {
    let (m, weights, _) = tiny_model();
    let hub = FakeHub::start("127.0.0.1:0");
    hub.route(
        &url_path(m, 0),
        Route {
            body: weights.clone(),
            fail_next: Some(503),
            ..Default::default()
        },
    );
    // No route for the tokenizer: 404.
    let cache = tempfile::tempdir().unwrap();
    let e = client(&hub)
        .ensure(cache.path(), m, &mut quiet())
        .unwrap_err();
    assert!(format!("{e:#}").contains("HTTP 404"), "{e:#}");
    assert_eq!(hub.requests_for(&url_path(m, 0)).len(), 2);
    assert_eq!(hub.requests_for(&url_path(m, 1)).len(), 1);
}

#[test]
fn a_partial_file_from_an_earlier_run_does_not_count_as_progress() {
    let (m, weights, tokenizer) = tiny_model();
    let hub = FakeHub::start("127.0.0.1:0");
    serve_model(&hub, m, &weights, &tokenizer);
    hub.route(
        &url_path(m, 0),
        Route {
            body: weights.clone(),
            fail_always: Some(503),
            ..Default::default()
        },
    );
    let cache = tempfile::tempdir().unwrap();
    let c = cache.path();
    let blob = m.blob_path(c, &m.files[0]);
    std::fs::create_dir_all(blob.parent().unwrap()).unwrap();
    std::fs::write(incomplete_path(&blob), &weights[..1000]).unwrap();
    let e = client(&hub).ensure(c, m, &mut quiet()).unwrap_err();
    assert!(format!("{e:#}").contains("HTTP 503"), "{e:#}");
    // The first try and 3 retries, each one resuming from byte 1000.
    let seen = hub.requests_for(&url_path(m, 0));
    assert_eq!(seen.len(), 4, "{seen:?}");
    assert!(
        seen.iter()
            .all(|s| s.range.as_deref() == Some("bytes=1000-")),
        "{seen:?}"
    );
}

#[test]
fn a_range_nobody_asked_for_restarts_a_fresh_download() {
    let (m, weights, tokenizer) = tiny_model();
    let hub = FakeHub::start("127.0.0.1:0");
    serve_model(&hub, m, &weights, &tokenizer);
    hub.route(
        &url_path(m, 0),
        Route {
            body: weights.clone(),
            stray_range: Some(1000),
            ..Default::default()
        },
    );
    let cache = tempfile::tempdir().unwrap();
    let snap = client(&hub).ensure(cache.path(), m, &mut quiet()).unwrap();
    let seen = hub.requests_for(&url_path(m, 0));
    assert_eq!(seen.len(), 2, "{seen:?}");
    assert!(seen.iter().all(|s| s.range.is_none()), "{seen:?}");
    assert_eq!(
        std::fs::read(snap.join("model.safetensors")).unwrap(),
        weights
    );
}

#[test]
fn hf_token_goes_to_the_endpoint_but_not_the_redirect_host() {
    let (m, weights, tokenizer) = tiny_model();
    let hub = FakeHub::start("127.0.0.1:0");
    let cdn = FakeHub::start("[::1]:0");
    cdn.route(
        "/cdn/weights",
        Route {
            body: weights.clone(),
            ..Default::default()
        },
    );
    hub.route(
        &url_path(m, 0),
        Route {
            redirect: Some(format!("{}/cdn/weights", cdn.base)),
            ..Default::default()
        },
    );
    hub.route(
        &url_path(m, 1),
        Route {
            body: tokenizer.clone(),
            ..Default::default()
        },
    );
    let cache = tempfile::tempdir().unwrap();
    Hub::new(&hub.base, Some("hf_test".into()))
        .with_backoff(Duration::ZERO)
        .ensure(cache.path(), m, &mut quiet())
        .unwrap();
    let first = &hub.requests_for(&url_path(m, 0))[0];
    assert_eq!(first.auth.as_deref(), Some("Bearer hf_test"));
    let redirected = &cdn.requests_for("/cdn/weights")[0];
    assert!(redirected.host.starts_with("[::1]"), "{redirected:?}");
    assert_eq!(redirected.auth, None);
}

#[test]
fn hf_token_does_not_follow_a_redirect_to_another_port_on_the_same_host() {
    let (m, weights, tokenizer) = tiny_model();
    let hub = FakeHub::start("127.0.0.1:0");
    let other = FakeHub::start("127.0.0.1:0");
    assert_ne!(hub.base, other.base);
    serve_model(&hub, m, &weights, &tokenizer);
    other.route(
        "/other/weights",
        Route {
            body: weights.clone(),
            ..Default::default()
        },
    );
    hub.route(
        &url_path(m, 0),
        Route {
            redirect: Some(format!("{}/other/weights", other.base)),
            ..Default::default()
        },
    );
    let cache = tempfile::tempdir().unwrap();
    Hub::new(&hub.base, Some("hf_test".into()))
        .with_backoff(Duration::ZERO)
        .ensure(cache.path(), m, &mut quiet())
        .unwrap();
    let first = &hub.requests_for(&url_path(m, 0))[0];
    assert_eq!(first.auth.as_deref(), Some("Bearer hf_test"));
    let redirected = other.seen();
    assert_eq!(redirected.len(), 1, "{redirected:?}");
    assert!(
        redirected[0].host.starts_with("127.0.0.1:"),
        "{redirected:?}"
    );
    assert_eq!(redirected[0].auth, None);
}

#[test]
fn a_complete_snapshot_costs_no_request_and_a_lost_link_comes_back_from_the_blob() {
    let (m, weights, tokenizer) = tiny_model();
    let hub = FakeHub::start("127.0.0.1:0");
    serve_model(&hub, m, &weights, &tokenizer);
    let cache = tempfile::tempdir().unwrap();
    let c = cache.path();
    client(&hub).ensure(c, m, &mut quiet()).unwrap();
    let n = hub.seen().len();
    client(&hub).ensure(c, m, &mut quiet()).unwrap();
    assert_eq!(hub.seen().len(), n);
    let link = m.snapshot_dir(c).join("tokenizer/tokenizer.json");
    std::fs::remove_file(&link).unwrap();
    client(&hub).ensure(c, m, &mut quiet()).unwrap();
    assert_eq!(hub.seen().len(), n);
    assert_eq!(std::fs::read(&link).unwrap(), tokenizer);
}

#[test]
fn a_link_to_the_right_blob_stays_as_it_is() {
    let (m, weights, tokenizer) = tiny_model();
    let hub = FakeHub::start("127.0.0.1:0");
    serve_model(&hub, m, &weights, &tokenizer);
    let cache = tempfile::tempdir().unwrap();
    let c = cache.path();
    // Another process made the right link, but the blob is not there yet, so this download
    // fetches the blob and then finds the link in place.
    let link = m.snapshot_dir(c).join("model.safetensors");
    std::fs::create_dir_all(link.parent().unwrap()).unwrap();
    std::os::unix::fs::symlink(Path::new("../../blobs").join(m.files[0].blob), &link).unwrap();
    let inode = std::fs::symlink_metadata(&link).unwrap().ino();
    client(&hub).ensure(c, m, &mut quiet()).unwrap();
    assert_eq!(hub.requests_for(&url_path(m, 0)).len(), 1);
    assert_eq!(std::fs::symlink_metadata(&link).unwrap().ino(), inode);
    assert_eq!(std::fs::read(&link).unwrap(), weights);
}

#[test]
fn a_stale_link_is_replaced() {
    let (m, weights, tokenizer) = tiny_model();
    let hub = FakeHub::start("127.0.0.1:0");
    serve_model(&hub, m, &weights, &tokenizer);
    let cache = tempfile::tempdir().unwrap();
    let c = cache.path();
    let snap = m.snapshot_dir(c);
    // A link whose blob a user deleted.
    let link = snap.join("tokenizer/tokenizer.json");
    std::fs::create_dir_all(link.parent().unwrap()).unwrap();
    std::os::unix::fs::symlink("../../../blobs/gone", &link).unwrap();
    client(&hub).ensure(c, m, &mut quiet()).unwrap();
    assert_eq!(
        std::fs::read_link(&link).unwrap(),
        Path::new("../../../blobs").join(m.files[1].blob)
    );
    assert_eq!(std::fs::read(&link).unwrap(), tokenizer);
    // No temporary link stays behind.
    let names = |dir: &Path| {
        let mut v: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        v.sort();
        v
    };
    assert_eq!(names(&snap), ["model.safetensors", "tokenizer"]);
    assert_eq!(names(&snap.join("tokenizer")), ["tokenizer.json"]);
}

#[test]
fn two_downloads_at_once_fetch_each_file_once() {
    let (m, weights, tokenizer) = tiny_model();
    let hub = FakeHub::start("127.0.0.1:0");
    serve_model(&hub, m, &weights, &tokenizer);
    let cache = tempfile::tempdir().unwrap();
    let c = cache.path();
    std::thread::scope(|s| {
        for _ in 0..2 {
            s.spawn(|| client(&hub).ensure(c, m, &mut quiet()).unwrap());
        }
    });
    assert_eq!(hub.requests_for(&url_path(m, 0)).len(), 1);
    assert_eq!(hub.requests_for(&url_path(m, 1)).len(), 1);
    assert_eq!(m.status(c), Status::Complete);
}
