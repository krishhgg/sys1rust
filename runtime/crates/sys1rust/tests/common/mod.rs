//! Test helpers: a server fixture over a fake predictor and a raw HTTP/1.1 client, so the
//! tests control framing (chunked bodies, declared lengths, disconnects) byte for byte.

#![allow(dead_code)]

use serde_json::Value;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use sys1rust::config::ServedModel;
use sys1rust::http::{router, serve, AppState};
use sys1rust::worker::{Predictor, Worker};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::oneshot;

pub const SERVED_NAME: &str = "typed-decisions";
pub const SERVED_REPO: &str = "convaiinnovations/laya-typed-decisions";
pub const SERVED_SHA: &str = "1a793eb568e6718f15941d08f85432581df534e3";

pub fn served() -> ServedModel {
    ServedModel {
        name: SERVED_NAME.into(),
        repo: SERVED_REPO.into(),
        revision: Some(SERVED_SHA.into()),
        dir: PathBuf::from("/nonexistent/checkpoint"),
    }
}

pub struct TestServer {
    pub addr: SocketAddr,
    shutdown: Option<oneshot::Sender<()>>,
    task: Option<tokio::task::JoinHandle<std::io::Result<()>>>,
    worker: Option<Worker>,
}

impl TestServer {
    /// Start a server on a free port over `predictor`; `None` for the api key leaves auth off.
    pub async fn start(
        predictor: Arc<dyn Predictor + Send + Sync>,
        max_concurrent: usize,
        api_key: Option<&str>,
    ) -> TestServer {
        let factory = Box::new(move || Ok(Box::new(predictor) as Box<dyn Predictor>));
        let (worker, ready) = Worker::spawn(factory, max_concurrent);
        let ready = ready.recv().unwrap().expect("fake predictor loads");
        Self::start_with_worker(worker, ready.engine, max_concurrent, api_key, None).await
    }

    /// Like [`start`], with a shorter body read deadline than the default 10 s.
    pub async fn start_with_body_timeout(
        predictor: Arc<dyn Predictor + Send + Sync>,
        max_concurrent: usize,
        body_timeout: std::time::Duration,
    ) -> TestServer {
        let factory = Box::new(move || Ok(Box::new(predictor) as Box<dyn Predictor>));
        let (worker, ready) = Worker::spawn(factory, max_concurrent);
        let ready = ready.recv().unwrap().expect("fake predictor loads");
        Self::start_with_worker(
            worker,
            ready.engine,
            max_concurrent,
            None,
            Some(body_timeout),
        )
        .await
    }

    /// Start over a worker whose thread has already ended (the factory failed).
    pub async fn start_dead() -> TestServer {
        let factory = Box::new(|| Err(anyhow::anyhow!("no checkpoint")));
        let (worker, ready) = Worker::spawn(factory, 4);
        assert!(ready.recv().unwrap().is_err());
        Self::start_with_worker(worker, "fake".into(), 4, None, None).await
    }

    async fn start_with_worker(
        worker: Worker,
        engine: String,
        max_concurrent: usize,
        api_key: Option<&str>,
        body_timeout: Option<std::time::Duration>,
    ) -> TestServer {
        let mut state = AppState::new(
            worker.handle.clone(),
            served(),
            engine,
            "f16gelu".into(),
            api_key.map(str::to_string),
            max_concurrent,
        );
        if let Some(d) = body_timeout {
            Arc::get_mut(&mut state)
                .expect("state not shared yet")
                .body_timeout = d;
        }
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = oneshot::channel();
        let task = tokio::spawn(serve(listener, router(state), async move {
            let _ = rx.await;
        }));
        TestServer {
            addr,
            shutdown: Some(tx),
            task: Some(task),
            worker: Some(worker),
        }
    }

    /// Stop accepting, wait for in-flight requests, then join the inference thread.
    pub async fn stop(mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        if let Some(task) = self.task.take() {
            task.await.unwrap().unwrap();
        }
        if let Some(worker) = self.worker.take() {
            assert!(
                worker.join(std::time::Duration::from_secs(5)),
                "inference thread joined"
            );
        }
    }

    pub async fn connect(&self) -> TcpStream {
        let s = TcpStream::connect(self.addr).await.unwrap();
        s.set_nodelay(true).unwrap();
        s
    }

    /// One request on a fresh connection.
    pub async fn request(
        &self,
        method: &str,
        path: &str,
        headers: &[(&str, &str)],
        body: Option<&[u8]>,
    ) -> Resp {
        let mut s = self.connect().await;
        send(&mut s, &build_request(method, path, headers, body)).await;
        read_response(&mut s).await
    }

    pub async fn post_json(&self, body: &Value) -> Resp {
        self.post_bytes(body.to_string().as_bytes(), &[]).await
    }

    pub async fn post_bytes(&self, body: &[u8], extra: &[(&str, &str)]) -> Resp {
        let mut headers = vec![("Content-Type", "application/json")];
        headers.extend_from_slice(extra);
        self.request("POST", "/v1/systemone", &headers, Some(body))
            .await
    }
}

#[derive(Debug)]
pub struct Resp {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Resp {
    pub fn header(&self, name: &str) -> Option<&str> {
        let name = name.to_ascii_lowercase();
        self.headers
            .iter()
            .find(|(k, _)| *k == name)
            .map(|(_, v)| v.as_str())
    }
    pub fn json(&self) -> Value {
        serde_json::from_slice(&self.body)
            .unwrap_or_else(|e| panic!("{e}: {:?}", String::from_utf8_lossy(&self.body)))
    }
    pub fn detail(&self) -> String {
        self.json()["detail"]
            .as_str()
            .expect("detail string")
            .to_string()
    }
}

pub fn build_request(
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: Option<&[u8]>,
) -> Vec<u8> {
    let mut out = format!("{method} {path} HTTP/1.1\r\nHost: localhost\r\n");
    for (k, v) in headers {
        out.push_str(&format!("{k}: {v}\r\n"));
    }
    if let Some(b) = body {
        out.push_str(&format!("Content-Length: {}\r\n", b.len()));
    }
    out.push_str("\r\n");
    let mut bytes = out.into_bytes();
    if let Some(b) = body {
        bytes.extend_from_slice(b);
    }
    bytes
}

pub async fn send(s: &mut TcpStream, bytes: &[u8]) {
    s.write_all(bytes).await.unwrap();
    s.flush().await.unwrap();
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

async fn fill_until(s: &mut TcpStream, buf: &mut Vec<u8>, want: usize) {
    while buf.len() < want {
        let n = s.read_buf(buf).await.unwrap();
        assert!(
            n > 0,
            "connection closed with {} of {} body bytes",
            buf.len(),
            want
        );
    }
}

/// Read one HTTP/1.1 response: status line, headers, then a Content-Length or chunked body.
pub async fn read_response(s: &mut TcpStream) -> Resp {
    let mut buf = Vec::new();
    let head_end = loop {
        if let Some(p) = find(&buf, b"\r\n\r\n") {
            break p;
        }
        let n = s.read_buf(&mut buf).await.unwrap();
        assert!(
            n > 0,
            "connection closed before response headers: {:?}",
            String::from_utf8_lossy(&buf)
        );
    };
    let head = String::from_utf8(buf[..head_end].to_vec()).unwrap();
    let mut lines = head.split("\r\n");
    let status_line = lines.next().unwrap();
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .unwrap()
        .parse()
        .unwrap();
    let headers: Vec<(String, String)> = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
        .collect();
    let mut rest = buf[head_end + 4..].to_vec();
    let content_length = headers
        .iter()
        .find(|(k, _)| k == "content-length")
        .map(|(_, v)| v.parse::<usize>().unwrap());
    let chunked = headers
        .iter()
        .any(|(k, v)| k == "transfer-encoding" && v.to_ascii_lowercase().contains("chunked"));
    let body = if let Some(n) = content_length {
        fill_until(s, &mut rest, n).await;
        rest.truncate(n);
        rest
    } else if chunked {
        let mut body = Vec::new();
        loop {
            let line_end = loop {
                if let Some(p) = find(&rest, b"\r\n") {
                    break p;
                }
                let n = s.read_buf(&mut rest).await.unwrap();
                assert!(n > 0, "connection closed inside chunked body");
            };
            let size =
                usize::from_str_radix(std::str::from_utf8(&rest[..line_end]).unwrap().trim(), 16)
                    .unwrap();
            rest.drain(..line_end + 2);
            if size == 0 {
                break;
            }
            fill_until(s, &mut rest, size + 2).await;
            body.extend_from_slice(&rest[..size]);
            rest.drain(..size + 2);
        }
        body
    } else {
        loop {
            let n = s.read_buf(&mut rest).await.unwrap();
            if n == 0 {
                break;
            }
        }
        rest
    };
    Resp {
        status,
        headers,
        body,
    }
}

/// A minimal valid request body: one choice question over a short state.
pub fn valid_body() -> Value {
    serde_json::json!({
        "state": "I was charged twice this month, I want my money back",
        "questions": {
            "queue": {"type": "choice", "instructions": "Which team?",
                      "criteria": {"billing": "billing and refunds", "tech": "login and app issues"}}
        }
    })
}
