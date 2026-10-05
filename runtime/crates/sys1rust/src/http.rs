//! The HTTP layer: `POST /v1/systemone` and `GET /health`, with upstream's status codes and
//! `{"detail": ...}` error bodies. Handlers do no model work; a validated request becomes a
//! job for the inference thread ([`crate::worker`]).
//!
//! Order inside `/v1/systemone`, as upstream: bearer auth, non-blocking admission, declared
//! body length, streamed body cap, JSON and limit checks, then the forward.
//!
//! One intentional difference: the body must arrive within [`BODY_READ_TIMEOUT`] or the
//! request gets 408. Upstream (uvicorn) waits for a stalled body forever, and here that would
//! hold an admission slot and block graceful shutdown.

use crate::config::{ServedModel, MAX_CONCURRENT_CAP};
use crate::validate::{validate_body_async, Rejection, MAX_BODY_BYTES};
use crate::worker::{PredictError, WorkerHandle};
use crate::{log, sanitize};
use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::header::{AUTHORIZATION, CONTENT_LENGTH, CONTENT_TYPE, RETRY_AFTER};
use axum::http::{HeaderValue, Response, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::serve::ListenerExt;
use axum::Router;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;
use subtle::ConstantTimeEq;
use tokio::net::TcpListener;
use tokio::sync::Semaphore;

/// How long a request body may take to arrive once admitted. 2 MiB (the cap) in 10 s is
/// 200 KiB/s, far below any loopback or LAN client; a peer slower than that is stalled.
pub const BODY_READ_TIMEOUT: Duration = Duration::from_secs(10);

/// Everything the handlers share.
pub struct AppState {
    worker: WorkerHandle,
    served: ServedModel,
    /// Upstream's `routing` block, appended to every result by the inference thread.
    routing: Value,
    engine: String,
    tuning: String,
    /// `Bearer <key>` as bytes, compared in constant time against the raw header value.
    expected_auth: Option<Vec<u8>>,
    admission: Arc<Semaphore>,
    /// [`BODY_READ_TIMEOUT`]; public so tests can shorten it.
    pub body_timeout: Duration,
}

impl AppState {
    pub fn new(
        worker: WorkerHandle,
        served: ServedModel,
        engine: String,
        tuning: String,
        api_key: Option<String>,
        max_concurrent: usize,
    ) -> Arc<Self> {
        let routing = json!({
            "model": served.name,
            "repo": served.repo,
            "reason": "only checkpoint served",
        });
        Arc::new(AppState {
            worker,
            served,
            routing,
            engine,
            tuning,
            expected_auth: api_key
                .filter(|k| !k.is_empty())
                .map(|k| format!("Bearer {k}").into_bytes()),
            admission: Arc::new(Semaphore::new(max_concurrent.clamp(1, MAX_CONCURRENT_CAP))),
            body_timeout: BODY_READ_TIMEOUT,
        })
    }
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/systemone", post(systemone))
        .route("/health", get(health))
        .fallback(|| async { Rejection::new(StatusCode::NOT_FOUND, "Not Found") })
        .method_not_allowed_fallback(|| async {
            Rejection::new(StatusCode::METHOD_NOT_ALLOWED, "Method Not Allowed")
        })
        .with_state(state)
}

/// Serve `app` on `listener` with HTTP/1.1 keep-alive, `TCP_NODELAY` on every accepted
/// socket (Nagle delayed upstream's small responses on macOS) and graceful shutdown: once
/// `shutdown` completes, stop accepting and finish the in-flight requests.
///
/// There is no header read timeout and no connection cap, like uvicorn's defaults upstream.
/// `axum::serve` builds the hyper connection without a timer, so neither is reachable from
/// here. A connection that never completes its headers costs a socket and a task but no
/// admission slot. The default bind is loopback; a deployment that listens wider should put
/// a reverse proxy in front for header timeouts and connection limits.
pub async fn serve<F>(listener: TcpListener, app: Router, shutdown: F) -> std::io::Result<()>
where
    F: Future<Output = ()> + Send + 'static,
{
    let listener = listener.tap_io(|tcp| {
        if let Err(e) = tcp.set_nodelay(true) {
            log(format!("TCP_NODELAY: {e}"));
        }
    });
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await
}

impl IntoResponse for Rejection {
    fn into_response(self) -> axum::response::Response {
        let mut r = json_response(self.status, &json!({"detail": self.detail}));
        if let Some(secs) = self.retry_after {
            r.headers_mut().insert(RETRY_AFTER, HeaderValue::from(secs));
        }
        r
    }
}

fn json_response(status: StatusCode, body: &Value) -> Response<Body> {
    serialized_response(
        status,
        serde_json::to_vec(body).expect("json value serializes"),
    )
}

/// A response whose body is already JSON bytes.
fn serialized_response(status: StatusCode, body: Vec<u8>) -> Response<Body> {
    Response::builder()
        .status(status)
        .header(CONTENT_TYPE, HeaderValue::from_static("application/json"))
        .body(Body::from(body))
        .expect("valid response")
}

async fn health(State(st): State<Arc<AppState>>) -> Response<Body> {
    let alive = st.worker.is_alive();
    let name = st.served.name.as_str();
    let body = json!({
        "status": if alive { "ok" } else { "error" },
        "loaded": [name],
        "revisions": { name: st.served.revision },
        "device": if st.engine.contains("cpu") { "cpu" } else { "gpu" },
        "engine": st.engine,
        "tuning": st.tuning,
        "worker": if alive { "running" } else { "not running" },
    });
    json_response(
        if alive {
            StatusCode::OK
        } else {
            StatusCode::SERVICE_UNAVAILABLE
        },
        &body,
    )
}

async fn systemone(
    State(st): State<Arc<AppState>>,
    req: Request,
) -> Result<Response<Body>, Rejection> {
    check_auth(&st, req.headers().get(AUTHORIZATION))?;
    // Non-blocking: over-cap load is refused, not queued, so the bodies buffered at once stay
    // bounded. The permit lives until this function returns, i.e. through inference, except
    // that a deep body's parse thread carries it (see `validate_body_async` below), so it
    // covers the parse even if the client leaves and this future is dropped mid-parse.
    let permit = st.admission.clone().try_acquire_owned().map_err(|_| {
        // Admission turns over at inference speed, so one second is the honest hint.
        Rejection::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "server busy, try again later",
        )
        .with_retry_after(1)
    })?;
    if !st.worker.is_alive() {
        return Err(Rejection::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "inference worker is not running",
        ));
    }
    // A declared length over the cap is refused before any byte is read; the streaming cap
    // below is what holds for chunked or understated bodies.
    if let Some(len) = req
        .headers()
        .get(CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().parse::<usize>().ok())
    {
        if len > MAX_BODY_BYTES {
            return Err(body_too_large());
        }
    }
    let raw = tokio::time::timeout(st.body_timeout, read_body_capped(req.into_body()))
        .await
        .map_err(|_| {
            Rejection::new(StatusCode::REQUEST_TIMEOUT, "request body read timed out")
        })??;
    // From here on this task holds no request-derived `Value`: a deep body is parsed on its
    // own thread, what comes back is a `Validated` (flat drop) that goes straight to the
    // inference thread, and the answer is the serialized body. See `validate::DEEP_STACK`.
    // The permit goes along: a deep parse holds it on its thread and hands it back with the
    // result, so a disconnect mid-parse frees the slot only once the 64 MiB thread is done
    // and the parse threads alive at once never exceed the admission cap. The inference
    // path needs no such care, its channel is bounded and the worker skips a job whose
    // reply is closed.
    let (v, _permit) = validate_body_async(raw, &st.served.name, permit).await?;
    let pred = match st.worker.predict(v, st.routing.clone()).await {
        Ok(p) => p,
        Err(PredictError::Question(m)) => {
            return Err(Rejection::new(StatusCode::UNPROCESSABLE_ENTITY, m))
        }
        Err(PredictError::Dead) => {
            return Err(Rejection::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "inference worker is not running",
            ))
        }
        Err(PredictError::Failed(cause)) | Err(PredictError::Panicked(cause)) => {
            // The client learns nothing about paths, weights or memory; the operator does.
            log(format!(
                "inference failed for model={}: {}",
                st.served.name,
                sanitize(&cause)
            ));
            return Err(Rejection::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "inference failed",
            ));
        }
    };
    let ms = format!("{:.2}", pred.infer_ms);
    let mut resp = serialized_response(StatusCode::OK, pred.body);
    let headers = resp.headers_mut();
    headers.insert(
        "Server-Timing",
        HeaderValue::from_str(&format!("inference;dur={ms}")).expect("ascii"),
    );
    headers.insert(
        "X-Inference-Time-Ms",
        HeaderValue::from_str(&ms).expect("ascii"),
    );
    Ok(resp)
}

/// Constant-time over bytes: a header value may carry non-ASCII (latin-1) bytes on the wire,
/// and every such value must answer 401, never 500.
fn check_auth(st: &AppState, header: Option<&HeaderValue>) -> Result<(), Rejection> {
    let Some(expected) = &st.expected_auth else {
        return Ok(());
    };
    let supplied: &[u8] = header.map(HeaderValue::as_bytes).unwrap_or(b"");
    if bool::from(supplied.ct_eq(expected)) {
        Ok(())
    } else {
        Err(Rejection::new(
            StatusCode::UNAUTHORIZED,
            "invalid or missing bearer token",
        ))
    }
}

fn body_too_large() -> Rejection {
    Rejection::new(StatusCode::PAYLOAD_TOO_LARGE, "request body too large")
}

/// Read the body frame by frame and stop as soon as it passes the cap: the peer is already
/// over the limit and nothing further can make the request acceptable.
async fn read_body_capped(mut body: Body) -> Result<Vec<u8>, Rejection> {
    let mut buf = Vec::new();
    while let Some(frame) = body.frame().await {
        let frame = frame.map_err(|_| {
            Rejection::new(StatusCode::BAD_REQUEST, "request body could not be read")
        })?;
        if let Ok(data) = frame.into_data() {
            if buf.len() + data.len() > MAX_BODY_BYTES {
                return Err(body_too_large());
            }
            buf.extend_from_slice(&data);
        }
    }
    Ok(buf)
}
