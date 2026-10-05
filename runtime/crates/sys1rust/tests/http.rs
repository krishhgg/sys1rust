//! HTTP-layer tests over fake predictors (no MLX, no checkpoint): every status code, every
//! limit, auth, admission, the streamed body cap, the body read deadline, the model-field
//! rules, headers, `/health`, a panicking predictor and a client that disconnects while queued.

mod common;

use common::*;
use serde_json::{json, Map, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use sys1rust::config::MAX_CONCURRENT_CAP;
use sys1rust::validate::{
    nesting_depth, validate_body, MAX_BODY_BYTES, MAX_JSON_DEPTH, MAX_STATE_CHARS,
};
use sys1rust::worker::{Predictor, Worker};
use tokio::io::AsyncWriteExt;

/// Answers every question with choice `a`, except a score question, which gets laya-core's
/// `legend` (a copy of its criteria, one key per level); `usage.input_tokens` is the state's
/// char count.
struct Echo;

impl Predictor for Echo {
    fn predict(&self, state: &Value, questions: &Value) -> laya_core::Result<Value> {
        let answers: Map<String, Value> = questions
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, q)| {
                let answer = match (&q["type"], &q["criteria"]) {
                    (Value::String(t), Value::Array(levels)) if t == "score" => {
                        let legend: Map<String, Value> = levels
                            .iter()
                            .enumerate()
                            .map(|(i, c)| (i.to_string(), c.clone()))
                            .collect();
                        json!({"type": "score", "score": 1.0, "legend": legend})
                    }
                    _ => json!({"type": "choice", "choice": "a"}),
                };
                (k.clone(), answer)
            })
            .collect();
        let n = match state {
            Value::String(s) => s.chars().count(),
            other => other.to_string().chars().count(),
        };
        Ok(
            json!({"model": "laya-rl-agent", "answers": answers, "usage": {"input_tokens": n, "output_tokens": 0}}),
        )
    }
    fn engine(&self) -> String {
        "fake".into()
    }
}

/// Fails every prediction the given way.
enum Failing {
    Question,
    Backend,
    Panic,
}

impl Predictor for Failing {
    fn predict(&self, _: &Value, _: &Value) -> laya_core::Result<Value> {
        match self {
            Failing::Question => Err(laya_core::Error::Question(
                "question 'q': unknown type 'nope'".into(),
            )),
            Failing::Backend => Err(laya_core::Error::Backend(
                "metal: out of memory at /secret/path".into(),
            )),
            Failing::Panic => panic!("boom inside predict"),
        }
    }
    fn engine(&self) -> String {
        "fake".into()
    }
}

/// Blocks inside `predict` until the test sends one `()` per call; counts calls.
struct Gated {
    calls: AtomicUsize,
    started: Sender<()>,
    release: Mutex<Receiver<()>>,
}

impl Gated {
    fn new() -> (Arc<Gated>, Receiver<()>, Sender<()>) {
        let (started_tx, started_rx) = channel();
        let (release_tx, release_rx) = channel();
        let g = Arc::new(Gated {
            calls: AtomicUsize::new(0),
            started: started_tx,
            release: Mutex::new(release_rx),
        });
        (g, started_rx, release_tx)
    }
}

impl Predictor for Gated {
    fn predict(&self, state: &Value, questions: &Value) -> laya_core::Result<Value> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.started.send(()).unwrap();
        self.release.lock().unwrap().recv().unwrap();
        Echo.predict(state, questions)
    }
    fn engine(&self) -> String {
        "fake".into()
    }
}

async fn echo_server() -> TestServer {
    TestServer::start(Arc::new(Echo), 16, None).await
}

fn choice(n: usize) -> Value {
    let crit: Map<String, Value> = (0..n).map(|i| (format!("o{i}"), json!("option"))).collect();
    json!({"type": "choice", "instructions": "pick", "criteria": crit})
}

fn score(n: usize) -> Value {
    let levels: Vec<Value> = (0..n).map(|i| json!(format!("level {i}"))).collect();
    json!({"type": "score", "instructions": "rate", "criteria": levels})
}

// --- success path -------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn success_shape_headers_and_routing() {
    let srv = echo_server().await;
    let r = srv.post_json(&valid_body()).await;
    assert_eq!(r.status, 200, "{:?}", String::from_utf8_lossy(&r.body));
    assert_eq!(r.header("content-type"), Some("application/json"));
    let v = r.json();
    let keys: Vec<&String> = v.as_object().unwrap().keys().collect();
    assert_eq!(
        keys,
        ["model", "answers", "usage", "routing"],
        "Agent key order kept, routing appended"
    );
    assert_eq!(v["answers"]["queue"]["choice"], "a");
    assert_eq!(
        v["routing"],
        json!({"model": SERVED_NAME, "repo": SERVED_REPO, "reason": "only checkpoint served"})
    );
    let st = r.header("server-timing").expect("Server-Timing");
    let dur = st
        .strip_prefix("inference;dur=")
        .expect("inference;dur= prefix");
    assert_eq!(
        dur.split('.').nth(1).map(str::len),
        Some(2),
        "two decimals: {st}"
    );
    let ms = r
        .header("x-inference-time-ms")
        .expect("X-Inference-Time-Ms");
    assert_eq!(ms, dur);
    assert!(ms.parse::<f64>().unwrap() >= 0.0);
    srv.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn keep_alive_serves_two_requests_on_one_connection() {
    let srv = echo_server().await;
    let mut s = srv.connect().await;
    let body = valid_body().to_string();
    let req = build_request(
        "POST",
        "/v1/systemone",
        &[("Content-Type", "application/json")],
        Some(body.as_bytes()),
    );
    send(&mut s, &req).await;
    let r1 = read_response(&mut s).await;
    send(&mut s, &req).await;
    let r2 = read_response(&mut s).await;
    assert_eq!((r1.status, r2.status), (200, 200));
    assert_ne!(r1.header("connection"), Some("close"));
    let h = build_request("GET", "/health", &[], None);
    send(&mut s, &h).await;
    assert_eq!(read_response(&mut s).await.status, 200);
    srv.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn empty_questions_object_is_accepted() {
    let srv = echo_server().await;
    let r = srv.post_json(&json!({"state": "x", "questions": {}})).await;
    assert_eq!(r.status, 200);
    assert_eq!(r.json()["answers"], json!({}));
    srv.stop().await;
}

// --- routes ---------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn health_reports_served_model() {
    let srv = echo_server().await;
    let r = srv.request("GET", "/health", &[], None).await;
    assert_eq!(r.status, 200);
    let v = r.json();
    assert_eq!(v["status"], "ok");
    assert_eq!(v["loaded"], json!([SERVED_NAME]));
    assert_eq!(v["revisions"], json!({SERVED_NAME: SERVED_SHA}));
    assert_eq!(v["device"], "gpu");
    assert_eq!(v["engine"], "fake");
    assert_eq!(v["tuning"], "f16gelu");
    srv.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unknown_route_is_404_and_wrong_method_is_405() {
    let srv = echo_server().await;
    let r = srv.request("GET", "/nope", &[], None).await;
    assert_eq!(r.status, 404);
    assert_eq!(r.detail(), "Not Found");
    let r = srv.request("GET", "/v1/systemone", &[], None).await;
    assert_eq!(r.status, 405);
    assert_eq!(r.detail(), "Method Not Allowed");
    let r = srv
        .request(
            "POST",
            "/health",
            &[("Content-Type", "application/json")],
            Some(b"{}"),
        )
        .await;
    assert_eq!(r.status, 405);
    assert_eq!(r.detail(), "Method Not Allowed");
    srv.stop().await;
}

// --- auth -----------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bearer_auth() {
    let srv = TestServer::start(Arc::new(Echo), 16, Some("s3cret")).await;
    let body = valid_body().to_string();
    let b = body.as_bytes();
    let r = srv.post_bytes(b, &[]).await;
    assert_eq!(
        (r.status, r.detail()),
        (401, "invalid or missing bearer token".to_string())
    );
    let r = srv
        .post_bytes(b, &[("Authorization", "Bearer wrong")])
        .await;
    assert_eq!(r.status, 401);
    let r = srv
        .post_bytes(b, &[("Authorization", "Basic s3cret")])
        .await;
    assert_eq!(r.status, 401);
    let r = srv
        .post_bytes(b, &[("Authorization", "Bearer s3cret")])
        .await;
    assert_eq!(r.status, 200);
    // /health stays open.
    let r = srv.request("GET", "/health", &[], None).await;
    assert_eq!(r.status, 200);
    // A non-ASCII header value (latin-1 byte 0xE9) is a 401, never a 500.
    let mut s = srv.connect().await;
    let mut raw = b"POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer s\xe9cret\r\nContent-Type: application/json\r\n".to_vec();
    raw.extend_from_slice(format!("Content-Length: {}\r\n\r\n", b.len()).as_bytes());
    raw.extend_from_slice(b);
    send(&mut s, &raw).await;
    let r = read_response(&mut s).await;
    assert_eq!(
        (r.status, r.detail()),
        (401, "invalid or missing bearer token".to_string())
    );
    srv.stop().await;
}

// --- 400s -----------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bad_request_bodies_are_400_with_upstream_details() {
    let srv = echo_server().await;
    let cases: Vec<(Vec<u8>, &str)> = vec![
        (b"{not json".to_vec(), "request body must be valid JSON"),
        (b"".to_vec(), "request body must be valid JSON"),
        (b"\xff\xfe".to_vec(), "request body must be valid JSON"),
        (
            b"[1,2]".to_vec(),
            "request body must be an object with a 'questions' field",
        ),
        (
            b"\"str\"".to_vec(),
            "request body must be an object with a 'questions' field",
        ),
        (
            br#"{"state": "x"}"#.to_vec(),
            "request body must be an object with a 'questions' field",
        ),
        (br#"{"questions": {}}"#.to_vec(), "'state' is required"),
        (
            br#"{"state": null, "questions": {}}"#.to_vec(),
            "'state' is required",
        ),
        (
            br#"{"state": "x", "questions": null}"#.to_vec(),
            "'questions' must be an object",
        ),
        (
            br#"{"state": "x", "questions": "q"}"#.to_vec(),
            "'questions' must be an object",
        ),
        (
            br#"{"state": "x", "questions": [1]}"#.to_vec(),
            "'questions' must be an object",
        ),
    ];
    for (body, detail) in cases {
        let r = srv.post_bytes(&body, &[]).await;
        assert_eq!(r.status, 400, "{:?}", String::from_utf8_lossy(&body));
        assert_eq!(r.detail(), detail, "{:?}", String::from_utf8_lossy(&body));
    }
    // A non-string state is accepted (serialized for the model).
    let r = srv
        .post_json(&json!({"state": {"a": 1}, "questions": {"q": choice(2)}}))
        .await;
    assert_eq!(r.status, 200);
    let r = srv
        .post_json(&json!({"state": 0, "questions": {"q": choice(2)}}))
        .await;
    assert_eq!(r.status, 200);
    let r = srv
        .post_json(&json!({"state": false, "questions": {"q": choice(2)}}))
        .await;
    assert_eq!(r.status, 200);
    srv.stop().await;
}

// --- 413 limits -----------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn request_limits_are_413() {
    let srv = echo_server().await;

    let qs: Map<String, Value> = (0..65).map(|i| (format!("q{i}"), choice(2))).collect();
    let r = srv.post_json(&json!({"state": "x", "questions": qs})).await;
    assert_eq!(
        (r.status, r.detail()),
        (413, "too many questions (65 > 64)".to_string())
    );
    let qs: Map<String, Value> = (0..64).map(|i| (format!("q{i}"), choice(2))).collect();
    let r = srv.post_json(&json!({"state": "x", "questions": qs})).await;
    assert_eq!(r.status, 200, "64 questions are allowed");

    let r = srv
        .post_json(&json!({"state": "x", "questions": {"pick": choice(101)}}))
        .await;
    assert_eq!(
        (r.status, r.detail()),
        (
            413,
            "too many choice options for 'pick' (101 > 100)".to_string()
        )
    );
    let r = srv
        .post_json(&json!({"state": "x", "questions": {"pick": choice(100)}}))
        .await;
    assert_eq!(r.status, 200);
    // Criteria given as a list count too; the qid is shown as a Python repr.
    let list: Vec<Value> = (0..101).map(|i| json!(format!("o{i}"))).collect();
    let r = srv
        .post_json(&json!({"state": "x", "questions": {"it's": {"type": "choice", "instructions": "i", "criteria": list}}}))
        .await;
    assert_eq!(
        (r.status, r.detail()),
        (
            413,
            "too many choice options for \"it's\" (101 > 100)".to_string()
        )
    );

    let r = srv
        .post_json(&json!({"state": "x", "questions": {"rate": score(33)}}))
        .await;
    assert_eq!(
        (r.status, r.detail()),
        (
            413,
            "too many score levels for 'rate' (33 > 32)".to_string()
        )
    );
    let r = srv
        .post_json(&json!({"state": "x", "questions": {"rate": score(32)}}))
        .await;
    assert_eq!(r.status, 200);

    let qs: Map<String, Value> = (0..6).map(|i| (format!("q{i}"), choice(100))).collect();
    let r = srv.post_json(&json!({"state": "x", "questions": qs})).await;
    assert_eq!(
        (r.status, r.detail()),
        (
            413,
            "too many answer options across questions (600 > 512)".to_string()
        )
    );
    let mut qs: Map<String, Value> = (0..5).map(|i| (format!("q{i}"), choice(100))).collect();
    qs.insert("s".into(), score(12));
    let r = srv.post_json(&json!({"state": "x", "questions": qs})).await;
    assert_eq!(r.status, 200, "exactly 512 options are allowed");
    // Questions that are not objects, and noul questions, count nothing here (422 later).
    let mut qs: Map<String, Value> = (0..5).map(|i| (format!("q{i}"), choice(100))).collect();
    qs.insert(
        "n".into(),
        json!({"type": "noul", "instructions": "yes?", "criteria": "anything"}),
    );
    qs.insert("bad".into(), json!("not an object"));
    let r = srv.post_json(&json!({"state": "x", "questions": qs})).await;
    assert_eq!(r.status, 200);

    // State: Unicode scalar values, not bytes. 50,000 two-byte chars pass; 50,001 do not.
    let ok: String = "é".repeat(MAX_STATE_CHARS);
    let r = srv
        .post_json(&json!({"state": ok, "questions": {"q": choice(2)}}))
        .await;
    assert_eq!(r.status, 200);
    let big: String = "é".repeat(MAX_STATE_CHARS + 1);
    let r = srv
        .post_json(&json!({"state": big, "questions": {"q": choice(2)}}))
        .await;
    assert_eq!(
        (r.status, r.detail()),
        (413, "state too large (50001 > 50000 chars)".to_string())
    );
    // A non-string state is measured as upstream's `len(str(state))`, the Python repr with
    // its ", " separators: `[1, 1, ..., 1]` over 30,000 items is 90,000 chars.
    let arr: Vec<Value> = (0..30000).map(|_| json!(1)).collect();
    let r = srv
        .post_json(&json!({"state": arr, "questions": {"q": choice(2)}}))
        .await;
    assert_eq!(
        (r.status, r.detail()),
        (413, "state too large (90000 > 50000 chars)".to_string())
    );
    srv.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn declared_body_over_cap_is_413_before_reading() {
    let srv = echo_server().await;
    let mut s = srv.connect().await;
    let head = format!(
        "POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        MAX_BODY_BYTES + 1
    );
    send(&mut s, head.as_bytes()).await;
    // No body bytes are sent; the response must come from the declared length alone.
    let r = read_response(&mut s).await;
    assert_eq!(
        (r.status, r.detail()),
        (413, "request body too large".to_string())
    );
    srv.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chunked_body_over_cap_is_413_while_streaming() {
    let srv = echo_server().await;
    let mut s = srv.connect().await;
    let head = b"POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n";
    send(&mut s, head).await;
    let (mut rd, mut wr) = s.into_split();
    // Stream 3 MiB in 64 KiB chunks; the server may close mid-stream, so write errors are fine.
    let writer = tokio::spawn(async move {
        let chunk = vec![b'{'; 64 * 1024];
        let header = format!("{:x}\r\n", chunk.len());
        for _ in 0..48 {
            if wr.write_all(header.as_bytes()).await.is_err()
                || wr.write_all(&chunk).await.is_err()
                || wr.write_all(b"\r\n").await.is_err()
            {
                return;
            }
        }
        let _ = wr.write_all(b"0\r\n\r\n").await;
    });
    let mut buf = Vec::new();
    let head_end = loop {
        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break p;
        }
        let n = tokio::io::AsyncReadExt::read_buf(&mut rd, &mut buf)
            .await
            .unwrap();
        assert!(n > 0, "closed without a response");
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
    assert!(head.starts_with("HTTP/1.1 413"), "{head}");
    let _ = writer.await;
    srv.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stalled_body_is_408_and_frees_its_admission_slot() {
    let timeout = std::time::Duration::from_millis(300);
    let srv = TestServer::start_with_body_timeout(Arc::new(Echo), 1, timeout).await;
    let body = valid_body().to_string();
    // Declare the full length, send half, then stop. The only admission slot is now held.
    let mut s = srv.connect().await;
    let head = format!(
        "POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        body.len()
    );
    send(&mut s, head.as_bytes()).await;
    send(&mut s, &body.as_bytes()[..body.len() / 2]).await;
    let t = std::time::Instant::now();
    let r = read_response(&mut s).await;
    assert_eq!(
        (r.status, r.detail()),
        (408, "request body read timed out".to_string())
    );
    assert!(t.elapsed() >= timeout, "answered before the deadline");
    assert!(
        t.elapsed() < timeout * 10,
        "took {:?} for a {timeout:?} deadline",
        t.elapsed()
    );
    // The slot is free again: a complete request on a fresh connection is served.
    let r = srv.post_json(&valid_body()).await;
    assert_eq!(r.status, 200, "{:?}", String::from_utf8_lossy(&r.body));
    // A body that arrives in pieces within the deadline is fine.
    let mut s = srv.connect().await;
    send(&mut s, head.as_bytes()).await;
    send(&mut s, &body.as_bytes()[..body.len() / 2]).await;
    tokio::time::sleep(timeout / 3).await;
    send(&mut s, &body.as_bytes()[body.len() / 2..]).await;
    assert_eq!(read_response(&mut s).await.status, 200);
    srv.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chunked_body_under_cap_is_served() {
    let srv = echo_server().await;
    let mut s = srv.connect().await;
    let body = valid_body().to_string();
    let (a, b) = body.split_at(body.len() / 2);
    let raw = format!(
        "POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n{a}\r\n{:x}\r\n{b}\r\n0\r\n\r\n",
        a.len(),
        b.len()
    );
    send(&mut s, raw.as_bytes()).await;
    let r = read_response(&mut s).await;
    assert_eq!(r.status, 200);
    srv.stop().await;
}

// --- 422 overrides, model field -------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn budget_overrides_are_422() {
    let srv = echo_server().await;
    let mut b = valid_body();
    b["max_len"] = json!(512);
    let r = srv.post_json(&b).await;
    assert_eq!(
        (r.status, r.detail()),
        (
            422,
            "max_len overrides are not supported by sys1rust".to_string()
        )
    );
    let mut b = valid_body();
    b["head_max_len"] = json!("192");
    let r = srv.post_json(&b).await;
    assert_eq!(
        (r.status, r.detail()),
        (
            422,
            "head_max_len overrides are not supported by sys1rust".to_string()
        )
    );
    let mut b = valid_body();
    b["max_len"] = Value::Null;
    b["head_max_len"] = Value::Null;
    let r = srv.post_json(&b).await;
    assert_eq!(r.status, 200, "null overrides mean 'not set'");
    srv.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn model_field_rules() {
    let srv = echo_server().await;
    let accepted = [
        Value::Null,
        json!("jev-1"),
        json!("convaiinnovations/laya"),
        json!(123),
        json!(["typed-decisions"]),
        json!("typed-decisions"),
        json!("  Typed-Decisions "),
        json!(" CONVAIINNOVATIONS/LAYA-TYPED-DECISIONS "),
    ];
    for m in accepted {
        let mut b = valid_body();
        b["model"] = m.clone();
        let r = srv.post_json(&b).await;
        assert_eq!(r.status, 200, "model={m}");
        assert_eq!(r.json()["routing"]["model"], SERVED_NAME, "model={m}");
    }
    for m in [
        "english",
        "multilingual",
        "convaiinnovations/laya-multilingual",
        "MULTILINGUAL",
    ] {
        let mut b = valid_body();
        b["model"] = json!(m);
        let r = srv.post_json(&b).await;
        assert_eq!(r.status, 400, "model={m}");
        let d = r.detail();
        assert!(
            d.contains(m.trim()) && d.contains(SERVED_NAME),
            "names both: {d}"
        );
    }
    srv.stop().await;
}

// --- inference errors -----------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn question_error_is_422_with_message() {
    let srv = TestServer::start(Arc::new(Failing::Question), 16, None).await;
    let r = srv.post_json(&valid_body()).await;
    assert_eq!(
        (r.status, r.detail()),
        (422, "question 'q': unknown type 'nope'".to_string())
    );
    srv.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn backend_error_is_500_without_the_cause() {
    let srv = TestServer::start(Arc::new(Failing::Backend), 16, None).await;
    let r = srv.post_json(&valid_body()).await;
    assert_eq!(
        (r.status, r.detail()),
        (500, "inference failed".to_string())
    );
    assert!(!String::from_utf8_lossy(&r.body).contains("/secret/path"));
    srv.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn panic_is_500_and_the_server_survives() {
    let srv = TestServer::start(Arc::new(Failing::Panic), 16, None).await;
    let r = srv.post_json(&valid_body()).await;
    assert_eq!(
        (r.status, r.detail()),
        (500, "inference failed".to_string())
    );
    // Worker still alive: next request is handled (and panics again), health is still ok.
    let r = srv.post_json(&valid_body()).await;
    assert_eq!(r.status, 500);
    let h = srv.request("GET", "/health", &[], None).await;
    assert_eq!(h.status, 200);
    assert_eq!(h.json()["status"], "ok");
    srv.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dead_worker_is_reported_by_health_and_503() {
    let srv = TestServer::start_dead().await;
    let h = srv.request("GET", "/health", &[], None).await;
    assert_eq!(h.status, 503);
    assert_ne!(h.json()["status"], "ok");
    let r = srv.post_json(&valid_body()).await;
    assert_eq!(
        (r.status, r.detail()),
        (503, "inference worker is not running".to_string())
    );
    srv.stop().await;
}

// --- admission ------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn admission_full_is_503_with_retry_after() {
    let (gate, started, release) = Gated::new();
    let srv = TestServer::start(gate.clone(), 1, Some("k")).await;
    let body = valid_body().to_string();
    let auth = ("Authorization", "Bearer k");
    // Request A holds the only slot while the fake blocks.
    let mut a = srv.connect().await;
    send(
        &mut a,
        &build_request(
            "POST",
            "/v1/systemone",
            &[("Content-Type", "application/json"), auth],
            Some(body.as_bytes()),
        ),
    )
    .await;
    started
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("A reached the predictor");
    // B is refused before its body is read (an invalid body still gets the 503).
    let r = srv.post_bytes(b"{not json", &[auth]).await;
    assert_eq!(
        (r.status, r.detail()),
        (503, "server busy, try again later".to_string())
    );
    assert_eq!(r.header("retry-after"), Some("1"));
    // Auth is checked before admission: a bad token while busy is still 401.
    let r = srv
        .post_bytes(body.as_bytes(), &[("Authorization", "Bearer wrong")])
        .await;
    assert_eq!(r.status, 401);
    // /health is not gated.
    assert_eq!(srv.request("GET", "/health", &[], None).await.status, 200);
    release.send(()).unwrap();
    let ra = read_response(&mut a).await;
    assert_eq!(ra.status, 200);
    // The slot is free again.
    let mut c = srv.connect().await;
    send(
        &mut c,
        &build_request(
            "POST",
            "/v1/systemone",
            &[("Content-Type", "application/json"), auth],
            Some(body.as_bytes()),
        ),
    )
    .await;
    started
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("C reached the predictor");
    release.send(()).unwrap();
    assert_eq!(read_response(&mut c).await.status, 200);
    assert_eq!(gate.calls.load(Ordering::SeqCst), 2);
    srv.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn client_that_disconnects_while_queued_is_skipped() {
    let (gate, started, release) = Gated::new();
    let srv = TestServer::start(gate.clone(), 8, None).await;
    let body = valid_body().to_string();
    let req = build_request(
        "POST",
        "/v1/systemone",
        &[("Content-Type", "application/json")],
        Some(body.as_bytes()),
    );
    // A occupies the inference thread.
    let mut a = srv.connect().await;
    send(&mut a, &req).await;
    started
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap();
    // B is admitted, validated and queued behind A, then hangs up.
    let mut b = srv.connect().await;
    send(&mut b, &req).await;
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    drop(b);
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    // Let A finish; B must be skipped without touching the predictor.
    release.send(()).unwrap();
    assert_eq!(read_response(&mut a).await.status, 200);
    // C runs next and sees exactly the second call.
    let mut c = srv.connect().await;
    send(&mut c, &req).await;
    started
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap();
    release.send(()).unwrap();
    assert_eq!(read_response(&mut c).await.status, 200);
    assert_eq!(
        gate.calls.load(Ordering::SeqCst),
        2,
        "the disconnected client's job was skipped"
    );
    srv.stop().await;
}

// --- nesting depth --------------------------------------------------------------------------

/// `{"state": [[...]], "questions": {...}}` with the state array `depth` levels deep.
fn deep_body(depth: usize, questions: &Value) -> Vec<u8> {
    format!(
        r#"{{"state": {}{}, "questions": {questions}}}"#,
        "[".repeat(depth),
        "]".repeat(depth)
    )
    .into_bytes()
}

/// A document nested exactly `MAX_JSON_DEPTH` deep is served end to end (the fake predictor
/// serializes the state on the inference thread, the way laya-core does), one level deeper
/// is upstream's 400, and the server is still fine afterwards. Debug builds need about
/// 3 KiB of stack per level for this, so the test also proves the threads are sized for it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn nesting_at_the_limit_is_served_and_one_past_is_400() {
    let srv = echo_server().await;
    let questions = json!({"q": choice(2)});
    let r = srv
        .post_bytes(&deep_body(MAX_JSON_DEPTH - 1, &questions), &[])
        .await;
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    // Echo's usage is the state's serialized length: 2 chars per level.
    assert_eq!(
        r.json()["usage"]["input_tokens"],
        json!(2 * (MAX_JSON_DEPTH - 1))
    );
    let r = srv
        .post_bytes(&deep_body(MAX_JSON_DEPTH, &questions), &[])
        .await;
    assert_eq!(
        (r.status, r.detail()),
        (400, "request body must be valid JSON".to_string())
    );
    let r = srv.post_json(&valid_body()).await;
    assert_eq!(r.status, 200);
    srv.stop().await;
}

/// A request at the limit whose inference thread is already gone dies on the tokio side with
/// the 503, not a stack overflow.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deep_request_to_a_dead_worker_is_503() {
    let (worker, ready) = Worker::spawn(Box::new(|| Err(anyhow::anyhow!("no checkpoint"))), 4);
    assert!(ready.recv().unwrap().is_err());
    let handle = worker.handle.clone();
    assert!(worker.join(std::time::Duration::from_secs(5)));
    let req = validate_body(&deep_body(MAX_JSON_DEPTH - 1, &json!({})), SERVED_NAME).unwrap();
    assert!(matches!(
        handle.predict(req, json!({})).await,
        Err(sys1rust::worker::PredictError::Dead)
    ));
}

/// A score request whose one criterion is nested as deep as the document limit allows:
/// root, `questions`, the question and `criteria` are four levels, the criterion the rest.
fn deep_score_body() -> (Vec<u8>, String) {
    let criterion = format!(
        "{}{}",
        "[".repeat(MAX_JSON_DEPTH - 4),
        "]".repeat(MAX_JSON_DEPTH - 4)
    );
    let body = format!(
        r#"{{"state": "s", "questions": {{"q": {{"type": "score", "instructions": "rate", "criteria": [{criterion}]}}}}}}"#
    );
    assert_eq!(nesting_depth(body.as_bytes()), MAX_JSON_DEPTH);
    (body.into_bytes(), criterion)
}

/// The answer to a deep score request copies the criterion into `legend`, so the response is
/// as deep as the request. It is serialized on the inference thread and the handler only
/// forwards bytes, so a 2 MiB tokio thread never walks it: the response carries the criterion
/// byte for byte and the server is still up afterwards. In a debug build serializing this on
/// the handler's thread would need 16 MiB of stack and abort the process, so the test fails
/// loudly in debug and release alike.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deep_score_legend_is_served_and_the_server_survives() {
    let srv = echo_server().await;
    let (body, criterion) = deep_score_body();
    let r = srv.post_bytes(&body, &[]).await;
    assert_eq!(
        r.status,
        200,
        "{}",
        String::from_utf8_lossy(&r.body[..200.min(r.body.len())])
    );
    assert_eq!(nesting_depth(&r.body), MAX_JSON_DEPTH);
    let legend = format!(r#""legend":{{"0":{criterion}}}"#);
    assert!(
        r.body.windows(legend.len()).any(|w| w == legend.as_bytes()),
        "legend carries the criterion byte for byte"
    );
    assert!(r.body.ends_with(
        format!(
            r#","routing":{{"model":"{SERVED_NAME}","repo":"{SERVED_REPO}","reason":"only checkpoint served"}}}}"#
        )
        .as_bytes()
    ));
    let h = srv.request("GET", "/health", &[], None).await;
    assert_eq!(h.status, 200);
    let r = srv.post_json(&valid_body()).await;
    assert_eq!(r.status, 200);
    srv.stop().await;
}

/// The client of a deep score request leaves while its forward runs; the deep answer has
/// nowhere to go and dies on the inference thread. The server goes on serving.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deep_answer_for_a_gone_client_dies_on_the_inference_thread() {
    let (gate, started, release) = Gated::new();
    let srv = TestServer::start(gate.clone(), 8, None).await;
    let (body, _) = deep_score_body();
    let mut a = srv.connect().await;
    send(
        &mut a,
        &build_request(
            "POST",
            "/v1/systemone",
            &[("Content-Type", "application/json")],
            Some(&body),
        ),
    )
    .await;
    started
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap();
    drop(a);
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    release.send(()).unwrap();
    // The next request gets the second forward.
    let shallow = valid_body().to_string();
    let mut b = srv.connect().await;
    send(
        &mut b,
        &build_request(
            "POST",
            "/v1/systemone",
            &[("Content-Type", "application/json")],
            Some(shallow.as_bytes()),
        ),
    )
    .await;
    started
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap();
    release.send(()).unwrap();
    assert_eq!(read_response(&mut b).await.status, 200);
    assert_eq!(gate.calls.load(Ordering::SeqCst), 2);
    srv.stop().await;
}

/// Deep bodies are parsed off the runtime: on this single-threaded runtime four of them are
/// in flight while `/health` and a shallow request are answered. The parses are each near the
/// body cap so they take real time, and every request still completes.
#[tokio::test]
async fn health_answers_while_deep_bodies_parse() {
    let srv = echo_server().await;
    // A deep state padded with a long string: about 1.9 MiB, under the body cap, over the
    // state cap (a 413 once parsed and measured).
    let padding = "x".repeat(MAX_BODY_BYTES - 2 * MAX_JSON_DEPTH - 4096);
    let body = format!(
        r#"{{"state": {}"{padding}"{}, "questions": {{}}}}"#,
        "[".repeat(MAX_JSON_DEPTH - 1),
        "]".repeat(MAX_JSON_DEPTH - 1)
    )
    .into_bytes();
    assert!(body.len() <= MAX_BODY_BYTES);
    let req = build_request(
        "POST",
        "/v1/systemone",
        &[("Content-Type", "application/json")],
        Some(&body),
    );
    let mut conns = Vec::new();
    for _ in 0..4 {
        let mut c = srv.connect().await;
        send(&mut c, &req).await;
        conns.push(c);
    }
    let h = srv.request("GET", "/health", &[], None).await;
    assert_eq!(h.status, 200);
    let r = srv.post_json(&valid_body()).await;
    assert_eq!(r.status, 200);
    for mut c in conns {
        let r = read_response(&mut c).await;
        assert_eq!(r.status, 413, "{}", r.detail());
        assert!(
            r.detail().starts_with("state too large ("),
            "{}",
            r.detail()
        );
    }
    srv.stop().await;
}

// --- capacity -------------------------------------------------------------------------------

/// The largest `LAYA_MAX_CONCURRENT` the configuration produces builds the channel and the
/// semaphore without panicking, and the server admits requests.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn max_concurrent_at_the_cap_starts_and_serves() {
    let srv = TestServer::start(Arc::new(Echo), MAX_CONCURRENT_CAP, None).await;
    let r = srv.post_json(&valid_body()).await;
    assert_eq!(r.status, 200);
    srv.stop().await;
}
