//! The inference thread. MLX objects must stay on one OS thread, so the predictor is built
//! and used only here; HTTP handlers send validated jobs over a bounded channel and wait on
//! a oneshot for the answer. One forward at a time, in arrival order, no cross-request
//! batching (measured to gain nothing on this laptop).
//!
//! The answer that crosses back is the serialized response body, not a `Value`: a score
//! answer's `legend` copies the request's criteria, which may be nested [`MAX_JSON_DEPTH`]
//! deep, and walking that (serializing or dropping it) takes far more stack than a tokio
//! worker has. Everything that touches the result happens here, on [`DEEP_STACK`].
//!
//! [`MAX_JSON_DEPTH`]: crate::validate::MAX_JSON_DEPTH

use crate::config::MAX_CONCURRENT_CAP;
use crate::validate::{Validated, DEEP_STACK};
use serde_json::Value;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::mpsc as std_mpsc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};

/// The inference call behind the HTTP layer; the real one wraps `laya_core::Agent`.
pub trait Predictor {
    /// `Agent::predict`: one state, a `{qid: definition}` object.
    fn predict(&self, state: &Value, questions: &Value) -> laya_core::Result<Value>;
    /// Runs once before the server binds, so the first client request is not the slow first
    /// forward.
    fn warmup(&self) -> laya_core::Result<()> {
        Ok(())
    }
    /// Backend description for `/health` and the ready line, e.g. `mlx(gpu,f16)`.
    fn engine(&self) -> String;
}

impl<P: Predictor + ?Sized> Predictor for std::sync::Arc<P> {
    fn predict(&self, state: &Value, questions: &Value) -> laya_core::Result<Value> {
        (**self).predict(state, questions)
    }
    fn warmup(&self) -> laya_core::Result<()> {
        (**self).warmup()
    }
    fn engine(&self) -> String {
        (**self).engine()
    }
}

/// Builds the predictor on the inference thread (the model load happens there).
pub type Factory = Box<dyn FnOnce() -> anyhow::Result<Box<dyn Predictor>> + Send + 'static>;

/// What the thread reports once the model is loaded and warm.
#[derive(Debug, Clone)]
pub struct Ready {
    pub engine: String,
    pub load_ms: f64,
    pub warmup_ms: f64,
}

/// One answered prediction: the complete response body (the predictor's result with the
/// `routing` block appended, serialized on the inference thread) and the time inside
/// `predict`, not the queue wait.
#[derive(Debug, Clone)]
pub struct Prediction {
    pub body: Vec<u8>,
    pub infer_ms: f64,
}

#[derive(Debug)]
pub enum PredictError {
    /// `laya_core::Error::Question`: the client's question is invalid (422).
    Question(String),
    /// Any other `laya_core::Error` (500; the message is for the log only).
    Failed(String),
    /// `predict` panicked (500); the worker keeps running.
    Panicked(String),
    /// The inference thread is gone (503).
    Dead,
}

/// A queued request. `req` keeps its non-recursive drop, so a job that dies on a tokio thread
/// (the inference thread is gone, or the client left while the send was pending) is safe at
/// any nesting depth. `routing` is the small constant block the HTTP layer adds to every
/// result (upstream's `Router.predict` sets it); it is appended here, before serialization.
struct Job {
    req: Validated,
    routing: Value,
    reply: oneshot::Sender<Result<Prediction, PredictError>>,
}

/// The HTTP side of the channel. Cloned into the app state.
#[derive(Clone)]
pub struct WorkerHandle {
    tx: mpsc::Sender<Job>,
}

impl WorkerHandle {
    /// False once the inference thread has exited (all it holds is the receiver).
    pub fn is_alive(&self) -> bool {
        !self.tx.is_closed()
    }

    /// Queue one prediction and wait for its answer: the response body with `routing`
    /// appended to the result.
    pub async fn predict(
        &self,
        req: Validated,
        routing: Value,
    ) -> Result<Prediction, PredictError> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Job {
                req,
                routing,
                reply,
            })
            .await
            .map_err(|_| PredictError::Dead)?;
        // The worker drops a job without answering only when it skips it because this
        // receiver is already gone, so a closed channel here means the thread died.
        rx.await.unwrap_or(Err(PredictError::Dead))
    }
}

/// The inference thread plus a handle to it.
pub struct Worker {
    pub handle: WorkerHandle,
    thread: Option<JoinHandle<()>>,
}

impl Worker {
    /// Start the thread: it runs `factory`, warms up, reports on the returned receiver, then
    /// serves jobs until every [`WorkerHandle`] is dropped. `capacity` bounds the queue; the
    /// admission semaphore already caps in-flight requests, so it is sized to match, within
    /// what a tokio channel can hold.
    pub fn spawn(
        factory: Factory,
        capacity: usize,
    ) -> (Worker, std_mpsc::Receiver<anyhow::Result<Ready>>) {
        let (tx, rx) = mpsc::channel::<Job>(capacity.clamp(1, MAX_CONCURRENT_CAP));
        let (ready_tx, ready_rx) = std_mpsc::channel();
        // The thread renders the state and the criteria as Python text (laya-core's dumps
        // and repr recurse once per level), copies criteria into the result and serializes
        // it, so its stack is sized for MAX_JSON_DEPTH.
        let thread = std::thread::Builder::new()
            .name("sys1rust-infer".into())
            .stack_size(DEEP_STACK)
            .spawn(move || run(factory, rx, ready_tx))
            .expect("spawn inference thread");
        (
            Worker {
                handle: WorkerHandle { tx },
                thread: Some(thread),
            },
            ready_rx,
        )
    }

    /// Drop this handle and wait for the thread to exit (it does once the last handle is
    /// gone). Returns false if it is still running after `timeout`.
    pub fn join(mut self, timeout: Duration) -> bool {
        let Some(thread) = self.thread.take() else {
            return true;
        };
        drop(self.handle);
        let deadline = Instant::now() + timeout;
        while !thread.is_finished() {
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        thread.join().is_ok()
    }
}

fn run(
    factory: Factory,
    mut rx: mpsc::Receiver<Job>,
    ready_tx: std_mpsc::Sender<anyhow::Result<Ready>>,
) {
    let t0 = Instant::now();
    let predictor = match factory() {
        Ok(p) => p,
        Err(e) => {
            let _ = ready_tx.send(Err(e));
            return;
        }
    };
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let t1 = Instant::now();
    if let Err(e) = predictor.warmup() {
        let _ = ready_tx.send(Err(anyhow::anyhow!("warm-up failed: {e}")));
        return;
    }
    let warmup_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let _ = ready_tx.send(Ok(Ready {
        engine: predictor.engine(),
        load_ms,
        warmup_ms,
    }));

    while let Some(job) = rx.blocking_recv() {
        if job.reply.is_closed() {
            // The client hung up while queued; its forward would be wasted.
            continue;
        }
        let t = Instant::now();
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            predictor.predict(&job.req.state, &job.req.questions)
        }));
        let infer_ms = t.elapsed().as_secs_f64() * 1000.0;
        let out = match outcome {
            Ok(Ok(result)) => {
                serialize(result, job.routing).map(|body| Prediction { body, infer_ms })
            }
            Ok(Err(laya_core::Error::Question(m))) => Err(PredictError::Question(m)),
            Ok(Err(e)) => Err(PredictError::Failed(e.to_string())),
            Err(payload) => Err(PredictError::Panicked(panic_message(payload.as_ref()))),
        };
        // A closed reply means the client left; `out` dies here, on this stack.
        let _ = job.reply.send(out);
        // `job.req` and the result are dropped here too, never on a tokio thread.
    }
}

/// Append `routing` to the predictor's result and serialize it, all on this thread. The
/// result is deep only through a score `legend`, which copies a criterion of the request:
/// the response is never nested deeper than the request that produced it, and upstream's
/// `json.dumps` (JSONResponse) serializes everything its `json.loads` accepted, so a deep
/// legend is a 200 there and here. The result is an object in `Agent::predict`'s contract;
/// anything else is a predictor bug reported as the opaque 500.
fn serialize(result: Value, routing: Value) -> Result<Vec<u8>, PredictError> {
    let Value::Object(mut map) = result else {
        return Err(PredictError::Failed(
            "predictor returned a result that is not an object".into(),
        ));
    };
    map.insert("routing".into(), routing);
    serde_json::to_vec(&map)
        .map_err(|e| PredictError::Failed(format!("serializing the result: {e}")))
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "non-string panic payload".into()
    }
}
