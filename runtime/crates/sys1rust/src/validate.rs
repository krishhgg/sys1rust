//! Upstream's request checks, in upstream's order and with its `detail` strings
//! (`laya_serve.py`: `_systemone_inner`, `_check_request_limits`, `_resolve_model`), plus the
//! two rules this spec adds: budget overrides are refused, and a `model` naming another
//! checkpoint is an error rather than a routing hint.

use crate::config::checkpoint_name;
use axum::http::StatusCode;
use laya_core::pyjson::{py_str, repr_str};
use serde::Deserialize;
use serde_json::Value;

pub const MAX_QUESTIONS: usize = 64;
pub const MAX_STATE_CHARS: usize = 50_000;
pub const MAX_BODY_BYTES: usize = 2 * 1024 * 1024;
pub const MAX_CHOICE_OPTIONS: usize = 100;
pub const MAX_SCORE_LEVELS: usize = 32;
pub const MAX_TOTAL_OPTIONS: usize = 512;

/// Deepest container nesting a body may have (`[[]]` is 2). Upstream has no explicit limit:
/// `json.loads` raises RecursionError at CPython's C recursion limit, which
/// `_systemone_inner` (laya_serve.py, the `except (ValueError, RecursionError)` around
/// `json.loads`) answers with 400 "request body must be valid JSON". That limit is 10,000 in
/// CPython 3.12 less the C frames already in use, so on the bench's 3.12.13 a document 9,997
/// deep is parsed and 9,998 is refused; `str()` and `json.dumps` in the rest of the handler
/// have the same limit, so everything `json.loads` accepts is served end to end. The limit
/// here is a fixed 10,000, so the two or three levels below it that a given CPython build
/// refuses are served instead. Past it the answer is upstream's 400.
pub const MAX_JSON_DEPTH: usize = 10_000;
/// Bodies nested deeper than this are parsed and checked on a thread with [`DEEP_STACK`]
/// instead of the handler's 2 MiB tokio thread. Measured in a debug build, serde_json needs
/// about 3.2 KiB of stack per level (192 KiB at 64 levels) and the Python-text walks up to
/// 2.4 KiB, so 32 levels leave the handler thread most of its stack.
pub const INLINE_DEPTH: usize = 32;
/// Stack for the threads that walk a value nested up to [`MAX_JSON_DEPTH`]: the parse and
/// check thread here and the inference thread, where laya-core renders the state and the
/// criteria as Python text, copies a score criterion into the answer's `legend`, and the
/// response is serialized to bytes. Those are the only threads a request-derived value is
/// ever parsed, cloned, serialized or recursively dropped on; a tokio thread sees at most a
/// [`Validated`] (flat drop) and response bytes. Measured at 10,000 levels in a debug build:
/// parse 32 MiB, clone 24 MiB, repr, dumps and drop 16 MiB; a release build needs a quarter
/// of that. This is a reservation of address space, only the pages touched are committed.
pub const DEEP_STACK: usize = 64 * 1024 * 1024;

/// A refused request: the status and the `detail` string of the error body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rejection {
    pub status: StatusCode,
    pub detail: String,
    /// `Retry-After` seconds, set only on the admission 503.
    pub retry_after: Option<u32>,
}

impl Rejection {
    pub fn new(status: StatusCode, detail: impl Into<String>) -> Self {
        Rejection {
            status,
            detail: detail.into(),
            retry_after: None,
        }
    }

    pub fn with_retry_after(mut self, seconds: u32) -> Self {
        self.retry_after = Some(seconds);
        self
    }
}

fn bad_request(detail: &str) -> Rejection {
    Rejection::new(StatusCode::BAD_REQUEST, detail)
}

fn too_large(detail: String) -> Rejection {
    Rejection::new(StatusCode::PAYLOAD_TOO_LARGE, detail)
}

/// The parts of a valid request handed to the inference thread. Dropping one never recurses
/// into the values: `Value`'s own drop takes 6 MiB of stack at [`MAX_JSON_DEPTH`] levels in
/// a release build, and a request can die on a 2 MiB tokio thread (the inference thread has
/// gone, or the client left while the job was queued).
#[derive(Debug)]
pub struct Validated {
    pub state: Value,
    pub questions: Value,
}

impl Drop for Validated {
    fn drop(&mut self) {
        drop_flat(std::mem::take(&mut self.state));
        drop_flat(std::mem::take(&mut self.questions));
    }
}

/// Drop `v` with a work list instead of recursion.
fn drop_flat(v: Value) {
    let mut pending = vec![v];
    while let Some(v) = pending.pop() {
        match v {
            Value::Array(items) => pending.extend(items),
            Value::Object(map) => pending.extend(map.into_iter().map(|(_, v)| v)),
            _ => {}
        }
    }
}

/// Deepest container nesting in `raw`, counted the way the parser recurses: every `[` or `{`
/// outside a string opens a level. Strings follow JSON's lexical rules (`\` escapes the next
/// byte), so a bracket inside one does not count. For text that is not JSON the count is
/// still an upper bound on the parser's recursion: the parser stops at its first error, and
/// up to that point it agrees with this scan on where strings begin and end.
pub fn nesting_depth(raw: &[u8]) -> usize {
    let (mut depth, mut deepest) = (0usize, 0usize);
    let mut in_string = false;
    let mut escaped = false;
    for &b in raw {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'[' | b'{' => {
                depth += 1;
                deepest = deepest.max(depth);
            }
            b']' | b'}' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    deepest
}

/// `serde_json::from_slice` without its 128-level recursion limit. Callers check
/// [`nesting_depth`] first and run this on a stack sized for the result.
///
/// This parses standard JSON (RFC 8259) in UTF-8, and that is an intentional difference
/// from upstream. Upstream's `json.loads(raw)` also accepts the tokens `NaN`, `Infinity` and
/// `-Infinity`, `\u` escapes of unpaired surrogates, and bodies in UTF-16 or UTF-32 or with a
/// byte order mark. A `serde_json::Value` cannot hold a NaN or an unpaired surrogate, so
/// matching those would take a different value type through laya-core. sys1rust answers all of
/// them with the 400 it gives malformed JSON, where upstream may serve them. Standard
/// encoders do not write the three tokens: `JSON.stringify` writes `null` for NaN and the
/// infinities, and `requests` and `httpx` refuse to send them. An unpaired surrogate is
/// different: `JSON.stringify` escapes one as `\ud800`, so a JavaScript client whose string
/// holds one gets this 400 anywhere in the body, even in a field upstream ignores.
fn parse_json(raw: &[u8]) -> serde_json::Result<Value> {
    let mut de = serde_json::Deserializer::from_slice(raw);
    de.disable_recursion_limit();
    let v = Value::deserialize(&mut de)?;
    de.end()?;
    Ok(v)
}

/// The depth gate both entry points share: past [`MAX_JSON_DEPTH`] is upstream's
/// RecursionError branch (the same 400 as malformed JSON); otherwise, whether the body is deep
/// enough to need a thread with [`DEEP_STACK`].
fn needs_deep_stack(raw: &[u8]) -> Result<bool, Rejection> {
    let depth = nesting_depth(raw);
    if depth > MAX_JSON_DEPTH {
        return Err(bad_request("request body must be valid JSON"));
    }
    Ok(depth > INLINE_DEPTH)
}

/// The thread a deep body is parsed and checked on: a stack for [`MAX_JSON_DEPTH`] levels.
/// Every value the checks build dies on that thread too.
fn deep_thread() -> std::thread::Builder {
    std::thread::Builder::new()
        .name("sys1rust-deep-body".into())
        .stack_size(DEEP_STACK)
}

/// The OS refused the thread or it panicked before answering: the operator gets the cause,
/// the client upstream's opaque 500.
fn deep_thread_failed(cause: impl std::fmt::Display) -> Rejection {
    crate::log(format!("deep body thread: {cause}"));
    Rejection::new(StatusCode::INTERNAL_SERVER_ERROR, "inference failed")
}

/// Parse and check a complete request body against `served_name`, blocking the caller while
/// a deep body is parsed on its own thread. For the HTTP handler use [`validate_body_async`].
pub fn validate_body(raw: &[u8], served_name: &str) -> Result<Validated, Rejection> {
    if !needs_deep_stack(raw)? {
        return check_parsed(raw, served_name);
    }
    std::thread::scope(|s| {
        let thread = deep_thread()
            .spawn_scoped(s, || check_parsed(raw, served_name))
            .map_err(deep_thread_failed)?;
        thread
            .join()
            .unwrap_or_else(|_| Err(deep_thread_failed("panicked")))
    })
}

/// [`validate_body`] for an async caller: a deep body is parsed on its own thread while the
/// caller's task yields, so a tokio worker is never held for the parse and `/health` and the
/// other connections keep moving. `hold` is whatever guard covers the request (the handler's
/// admission permit; `()` when there is none). A shallow body is checked inline and the
/// guard comes straight back. For a deep body the guard moves onto the parse thread with
/// `raw` and comes back with the result; if the caller is gone by the time the thread
/// finishes (the client disconnected), the result dies on its stack and the guard is
/// released there, after the parse, so the permit counts the parse thread for as long as
/// it runs. When the result does cross to the caller it is a [`Validated`], whose drop is
/// flat, so the tokio side may drop it but must not walk it.
pub async fn validate_body_async<H: Send + 'static>(
    raw: Vec<u8>,
    served_name: &str,
    hold: H,
) -> Result<(Validated, H), Rejection> {
    if !needs_deep_stack(&raw)? {
        return check_parsed(&raw, served_name).map(|v| (v, hold));
    }
    let served_name = served_name.to_string();
    let (checked, hold) = on_deep_stack(move || check_parsed(&raw, &served_name), hold).await?;
    checked.map(|v| (v, hold))
}

/// Run `work` on a [`deep_thread`] and await its result without blocking the runtime. The
/// thread owns `hold` while `work` runs and sends it back with the result, so a caller that
/// drops this future early does not release the guard: it drops on the thread once `work`
/// returns. If the thread cannot be spawned the guard drops here with the closure, which is
/// fine because the caller answers the error at once.
async fn on_deep_stack<T: Send + 'static, H: Send + 'static>(
    work: impl FnOnce() -> T + Send + 'static,
    hold: H,
) -> Result<(T, H), Rejection> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    deep_thread()
        .spawn(move || {
            let result = work();
            // A closed receiver means the caller is gone; the result and the guard are
            // dropped right here, after the work.
            let _ = tx.send((result, hold));
        })
        .map_err(deep_thread_failed)?;
    // The sender is dropped without a value only when `work` panicked.
    rx.await.map_err(|_| deep_thread_failed("panicked"))
}

/// [`validate_body`] after the depth check: upstream's checks in upstream's order.
fn check_parsed(raw: &[u8], served_name: &str) -> Result<Validated, Rejection> {
    let body = parse_json(raw).map_err(|_| bad_request("request body must be valid JSON"))?;
    let Value::Object(mut obj) = body else {
        return Err(bad_request(
            "request body must be an object with a 'questions' field",
        ));
    };
    if !obj.contains_key("questions") {
        return Err(bad_request(
            "request body must be an object with a 'questions' field",
        ));
    }
    let state = match obj.remove("state") {
        None | Some(Value::Null) => return Err(bad_request("'state' is required")),
        Some(s) => s,
    };
    let questions = obj.remove("questions").expect("checked above");
    check_limits(&state, &questions)?;
    for key in ["max_len", "head_max_len"] {
        if matches!(obj.get(key), Some(v) if !v.is_null()) {
            return Err(Rejection::new(
                StatusCode::UNPROCESSABLE_ENTITY,
                format!("{key} overrides are not supported by sys1rust"),
            ));
        }
    }
    check_model(obj.get("model"), served_name)?;
    Ok(Validated { state, questions })
}

/// `_check_request_limits`: reject absent or oversized requests before tokenization.
pub fn check_limits(state: &Value, questions: &Value) -> Result<(), Rejection> {
    let Value::Object(qs) = questions else {
        return Err(bad_request("'questions' must be an object"));
    };
    if qs.len() > MAX_QUESTIONS {
        return Err(too_large(format!(
            "too many questions ({} > {MAX_QUESTIONS})",
            qs.len()
        )));
    }
    let mut total = 0usize;
    for (qid, q) in qs {
        let Value::Object(q) = q else { continue };
        let crit = q.get("criteria");
        match q.get("type").and_then(Value::as_str) {
            Some("choice") => {
                let count = match crit {
                    Some(Value::Object(m)) => m.len(),
                    Some(Value::Array(a)) => a.len(),
                    _ => continue,
                };
                total += count;
                if count > MAX_CHOICE_OPTIONS {
                    return Err(too_large(format!(
                        "too many choice options for {} ({count} > {MAX_CHOICE_OPTIONS})",
                        repr_str(qid)
                    )));
                }
            }
            Some("score") => {
                let Some(Value::Array(a)) = crit else {
                    continue;
                };
                total += a.len();
                if a.len() > MAX_SCORE_LEVELS {
                    return Err(too_large(format!(
                        "too many score levels for {} ({} > {MAX_SCORE_LEVELS})",
                        repr_str(qid),
                        a.len()
                    )));
                }
            }
            _ => {}
        }
    }
    if total > MAX_TOTAL_OPTIONS {
        return Err(too_large(format!(
            "too many answer options across questions ({total} > {MAX_TOTAL_OPTIONS})"
        )));
    }
    let n = state_chars(state);
    if n > MAX_STATE_CHARS {
        return Err(too_large(format!(
            "state too large ({n} > {MAX_STATE_CHARS} chars)"
        )));
    }
    Ok(())
}

/// Characters in the state as upstream counts them (`laya_serve.py`, `_check_request_limits`):
/// `len(state) if isinstance(state, str) else len(str(state))`, so a string by its Unicode
/// scalar values and anything else by the Python repr of the decoded object (`{'a': 1}` is 8,
/// `True` and `None` are 4). Upstream wraps that in `try/except Exception` with a fallback of
/// `MAX_STATE_CHARS + 1`; the only way `str()` fails on a decoded JSON value is a
/// RecursionError, and `json.loads` has the same recursion limit as `str()`, so a state it
/// decoded is one `str()` renders. The walk here runs on a stack sized for [`MAX_JSON_DEPTH`]
/// and cannot fail either, so there is no such branch.
pub fn state_chars(state: &Value) -> usize {
    match state {
        Value::String(s) => s.chars().count(),
        other => py_str(other).chars().count(),
    }
}

/// The `model` field: absent, null or anything that is not a known checkpoint name or
/// published id means "the served checkpoint"; a known name for another checkpoint is 400.
pub fn check_model(model: Option<&Value>, served_name: &str) -> Result<(), Rejection> {
    let Some(Value::String(s)) = model else {
        return Ok(());
    };
    match checkpoint_name(s) {
        Some(name) if name != served_name => Err(bad_request(&format!(
            "model {} names checkpoint '{name}', but this server serves only '{served_name}'",
            repr_str(s.trim())
        ))),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The question id in a 413 detail is Python's `repr`, escapes included.
    #[test]
    fn option_limit_detail_uses_python_repr() {
        let opts: serde_json::Map<String, Value> =
            (0..101).map(|i| (format!("o{i}"), json!("x"))).collect();
        let body = json!({"state": "s", "questions": {"zero\u{200b}width": {"type": "choice", "criteria": opts}}});
        let e = validate_body(body.to_string().as_bytes(), "typed-decisions").unwrap_err();
        assert_eq!(e.status, StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(
            e.detail,
            "too many choice options for 'zero\\u200bwidth' (101 > 100)"
        );
    }

    /// A string is measured as is; anything else as Python's `str()` of it (`{'a': 1}`,
    /// `True`, `None`, `[1, 2.0, 'x']`), spaces and literal spellings included.
    #[test]
    fn state_chars_is_len_of_python_str() {
        assert_eq!(state_chars(&json!("héllo")), 5);
        assert_eq!(state_chars(&json!("it's")), 4);
        assert_eq!(state_chars(&json!({"a": 1})), 8);
        assert_eq!(state_chars(&json!(true)), 4);
        assert_eq!(state_chars(&Value::Null), 4);
        assert_eq!(state_chars(&json!([1, 2.0, "x"])), 13);
        assert_eq!(state_chars(&json!({"k": "héllo"})), 14);
        // The limit is on the Python text: `['aa', 'aa', ...]` is 6 chars per item, so 8334
        // items are 50004 (a 413 with that count in the detail) and 8333 are 49998.
        let items: Vec<Value> = (0..8334).map(|_| json!("aa")).collect();
        assert_eq!(state_chars(&Value::Array(items.clone())), 50_004);
        let e = check_limits(&Value::Array(items), &json!({})).unwrap_err();
        assert_eq!(e.status, StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(e.detail, "state too large (50004 > 50000 chars)");
        let items: Vec<Value> = (0..8333).map(|_| json!("aa")).collect();
        assert!(check_limits(&Value::Array(items), &json!({})).is_ok());
    }

    #[test]
    fn model_rule() {
        assert!(check_model(None, "typed-decisions").is_ok());
        assert!(check_model(Some(&Value::Null), "typed-decisions").is_ok());
        assert!(check_model(Some(&json!("jev-1")), "typed-decisions").is_ok());
        assert!(check_model(Some(&json!(7)), "typed-decisions").is_ok());
        assert!(check_model(Some(&json!(" Typed-Decisions ")), "typed-decisions").is_ok());
        let e = check_model(Some(&json!("english")), "typed-decisions").unwrap_err();
        assert_eq!(e.status, StatusCode::BAD_REQUEST);
        assert!(
            e.detail.contains("'english'") && e.detail.contains("'typed-decisions'"),
            "{}",
            e.detail
        );
    }

    #[test]
    fn validate_body_order() {
        let served = "typed-decisions";
        let ok = validate_body(
            br#"{"state":"s","questions":{"q":{"type":"noul","instructions":"?"}}}"#,
            served,
        )
        .unwrap();
        assert_eq!(ok.state, json!("s"));
        assert!(ok.questions.is_object());
        // Overrides are checked after the limits and before the model field.
        let e = validate_body(
            br#"{"state":"s","questions":{},"max_len":1,"model":"english"}"#,
            served,
        )
        .unwrap_err();
        assert_eq!(e.status, StatusCode::UNPROCESSABLE_ENTITY);
        let e = validate_body(
            br#"{"state":"s","questions":{},"max_len":null,"model":"english"}"#,
            served,
        )
        .unwrap_err();
        assert_eq!(e.status, StatusCode::BAD_REQUEST);
        let e = validate_body(br#"{"state":"s","questions":[],"max_len":1}"#, served).unwrap_err();
        assert_eq!(e.detail, "'questions' must be an object");
    }

    /// What `json.loads` accepts beyond standard JSON gets the malformed-JSON 400 here, the
    /// intentional difference [`parse_json`] documents. Standard JSON around it still parses.
    #[test]
    fn python_only_json_is_refused_like_malformed_json() {
        let served = "typed-decisions";
        let utf16: Vec<u8> =
            r#"{"state":"s","questions":{}}"#.encode_utf16().flat_map(u16::to_le_bytes).collect();
        let mut bom = b"\xef\xbb\xbf".to_vec();
        bom.extend_from_slice(br#"{"state":"s","questions":{}}"#);
        let bodies: [&[u8]; 6] = [
            br#"{"state":"s","questions":{},"x":NaN}"#,
            br#"{"state":"s","questions":{},"x":Infinity}"#,
            br#"{"state":"s","questions":{},"x":-Infinity}"#,
            br#"{"state":"\ud800","questions":{}}"#,
            &bom,
            &utf16,
        ];
        for raw in bodies {
            let e = validate_body(raw, served).unwrap_err();
            assert_eq!(e.status, StatusCode::BAD_REQUEST, "{raw:?}");
            assert_eq!(e.detail, "request body must be valid JSON", "{raw:?}");
        }
        let ok = validate_body(br#"{"state":"\ud83d\ude00 NaN","questions":{}}"#, served).unwrap();
        assert_eq!(ok.state, json!("\u{1F600} NaN"));
    }

    fn nested_array(depth: usize) -> String {
        format!("{}{}", "[".repeat(depth), "]".repeat(depth))
    }

    /// `{"state": <array>, "questions": {}}`: the state sits one level inside the document.
    fn body_with_state_depth(depth: usize) -> String {
        format!(r#"{{"state": {}, "questions": {{}}}}"#, nested_array(depth))
    }

    #[test]
    fn nesting_depth_counts_containers_outside_strings() {
        assert_eq!(nesting_depth(b""), 0);
        assert_eq!(nesting_depth(b"1"), 0);
        assert_eq!(nesting_depth(b"[]"), 1);
        assert_eq!(nesting_depth(br#"{"a": [1, {"b": []}], "c": 2}"#), 4);
        // Brackets, an escaped quote and an escaped backslash inside strings do not count.
        assert_eq!(nesting_depth(br#"["[[[[", "{\"}", "\\"]"#), 1);
        // Text that is not JSON still gets an upper bound on the parser's recursion.
        assert_eq!(nesting_depth(b"]]]]"), 0);
        assert_eq!(nesting_depth(b"[[[["), 4);
        assert_eq!(nesting_depth(b"\"[[["), 0);
        assert_eq!(nesting_depth(body_with_state_depth(9).as_bytes()), 10);
    }

    /// The document limit is [`MAX_JSON_DEPTH`]: a state array nested one less fits and is
    /// served (upstream serves it too: `str()` of `[[...]]` is 2 chars per level, under the
    /// 50,000 cap), one more is upstream's RecursionError 400. Runs in debug and release.
    #[test]
    fn nesting_at_the_limit_is_accepted_and_one_past_is_400() {
        let served = "typed-decisions";
        let ok =
            validate_body(body_with_state_depth(MAX_JSON_DEPTH - 1).as_bytes(), served).unwrap();
        // Measured with a loop: this thread has 2 MiB and must not walk the value recursively.
        let (mut depth, mut cur) = (0, &ok.state);
        while let Value::Array(items) = cur {
            depth += 1;
            match items.first() {
                Some(inner) => cur = inner,
                None => break,
            }
        }
        assert_eq!(depth, MAX_JSON_DEPTH - 1);
        assert_eq!(ok.questions, json!({}));
        let e =
            validate_body(body_with_state_depth(MAX_JSON_DEPTH).as_bytes(), served).unwrap_err();
        assert_eq!(e.status, StatusCode::BAD_REQUEST);
        assert_eq!(e.detail, "request body must be valid JSON");
        // The bound holds for text the parser would reject anyway, like 1 MiB of `[`.
        let e = validate_body(&vec![b'['; 1 << 20], served).unwrap_err();
        assert_eq!(e.detail, "request body must be valid JSON");
    }

    /// Nesting inside `questions` is walked by the same code (a criterion value goes through
    /// laya-core's `dumps` on the inference thread), so it is accepted up to the same limit.
    #[test]
    fn deep_nesting_inside_questions_is_accepted() {
        // Root, questions, the question and criteria are four levels.
        let body = format!(
            r#"{{"state": "s", "questions": {{"q": {{"type": "choice", "instructions": "?", "criteria": {{"a": {}}}}}}}}}"#,
            nested_array(MAX_JSON_DEPTH - 4)
        );
        assert_eq!(nesting_depth(body.as_bytes()), MAX_JSON_DEPTH);
        let ok = validate_body(body.as_bytes(), "typed-decisions").unwrap();
        assert_eq!(ok.state, json!("s"));
        let body = format!(
            r#"{{"state": "s", "questions": {{"q": {{"type": "choice", "instructions": "?", "criteria": {{"a": {}}}}}}}}}"#,
            nested_array(MAX_JSON_DEPTH - 3)
        );
        let e = validate_body(body.as_bytes(), "typed-decisions").unwrap_err();
        assert_eq!(e.detail, "request body must be valid JSON");
    }

    /// A deep object state is measured like upstream's `str()`: `{'a': ` per level, the `1`
    /// and a `}` per level, 7 chars a level plus one. Upstream reports the same count for
    /// this state (Python 3.12.13: `state too large (69973 > 50000 chars)` at 9,996 levels).
    #[test]
    fn deep_object_state_is_measured_like_python() {
        let depth = MAX_JSON_DEPTH - 1;
        let body = format!(
            r#"{{"state": {}1{}, "questions": {{}}}}"#,
            r#"{"a": "#.repeat(depth),
            "}".repeat(depth)
        );
        let e = validate_body(body.as_bytes(), "typed-decisions").unwrap_err();
        assert_eq!(e.status, StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(
            e.detail,
            format!("state too large ({} > 50000 chars)", 7 * depth + 1)
        );
    }

    /// The async entry point gives the same answers as the sync one at the limit, one past
    /// it and for a shallow body, and runs on a plain tokio runtime. The guard comes back
    /// with the result on both the inline and the deep path.
    #[tokio::test]
    async fn validate_body_async_matches_the_sync_path() {
        let served = "typed-decisions";
        let (ok, hold) = validate_body_async(
            body_with_state_depth(MAX_JSON_DEPTH - 1).into_bytes(),
            served,
            "deep",
        )
        .await
        .unwrap();
        assert_eq!(hold, "deep");
        assert_eq!(ok.questions, json!({}));
        assert!(matches!(ok.state, Value::Array(_)));
        let e = validate_body_async(
            body_with_state_depth(MAX_JSON_DEPTH).into_bytes(),
            served,
            (),
        )
        .await
        .unwrap_err();
        assert_eq!(e.detail, "request body must be valid JSON");
        let (ok, hold) = validate_body_async(
            br#"{"state":"s","questions":{}}"#.to_vec(),
            served,
            "shallow",
        )
        .await
        .unwrap();
        assert_eq!(hold, "shallow");
        assert_eq!(ok.state, json!("s"));
        let e = validate_body_async(b"nope".to_vec(), served, ())
            .await
            .unwrap_err();
        assert_eq!(e.detail, "request body must be valid JSON");
    }

    /// While the deep thread works, the runtime keeps running other tasks. On this
    /// single-threaded runtime the thread is released by a task that can only run if the
    /// await yielded; a blocking join would deadlock, which the timeout turns into a failure.
    #[tokio::test]
    async fn deep_thread_does_not_block_the_runtime() {
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let work = on_deep_stack(
            move || {
                release_rx.recv().unwrap();
                7
            },
            (),
        );
        let releaser = async {
            tokio::task::yield_now().await;
            release_tx.send(()).unwrap();
        };
        let (got, ()) = tokio::time::timeout(std::time::Duration::from_secs(10), async {
            tokio::join!(work, releaser)
        })
        .await
        .expect("the runtime was blocked while the deep thread waited");
        assert_eq!(got.unwrap(), (7, ()));
    }

    /// A caller that is gone before the thread finishes (the client disconnected) leaves the
    /// result to die on the deep thread; a panic on that thread is upstream's opaque 500.
    #[tokio::test]
    async fn deep_thread_caller_gone_or_panicked() {
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
        let work = on_deep_stack(
            move || {
                release_rx.recv().unwrap();
                let _ = done_tx.send(());
                validate_body(
                    body_with_state_depth(MAX_JSON_DEPTH - 1).as_bytes(),
                    "typed-decisions",
                )
            },
            (),
        );
        // Poll once so the thread starts, then drop the future.
        let mut work = Box::pin(work);
        assert!(futures_poll_once(&mut work).is_none());
        drop(work);
        release_tx.send(()).unwrap();
        done_rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("the thread ran to completion on its own");
        let e = on_deep_stack(|| -> usize { panic!("boom") }, ())
            .await
            .unwrap_err();
        assert_eq!(e.status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(e.detail, "inference failed");
    }

    /// The guard outlives a dropped caller: with the handler's future gone (the client
    /// disconnected) the admission permit stays taken while the thread still works, and
    /// comes back only once the work ends. Every step is gated on a channel, so the
    /// assertions never race the thread.
    #[tokio::test]
    async fn deep_thread_holds_the_guard_until_the_work_ends() {
        let admission = std::sync::Arc::new(tokio::sync::Semaphore::new(1));
        let permit = admission.clone().acquire_owned().await.unwrap();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let work = on_deep_stack(
            move || {
                release_rx.recv().unwrap();
            },
            permit,
        );
        // Poll once so the thread starts with the permit, then drop the future.
        let mut work = Box::pin(work);
        assert!(futures_poll_once(&mut work).is_none());
        drop(work);
        // The thread is blocked in `work`, so the permit cannot have been released yet.
        assert_eq!(admission.available_permits(), 0);
        release_tx.send(()).unwrap();
        // The thread returns from `work`, finds the receiver gone and drops the permit.
        let reacquired =
            tokio::time::timeout(std::time::Duration::from_secs(10), admission.acquire())
                .await
                .expect("the permit was not released after the work ended")
                .unwrap();
        assert_eq!(admission.available_permits(), 0);
        drop(reacquired);
        assert_eq!(admission.available_permits(), 1);
    }

    /// Poll `fut` once with a no-op waker.
    fn futures_poll_once<T>(
        fut: &mut std::pin::Pin<Box<impl std::future::Future<Output = T>>>,
    ) -> Option<T> {
        let waker = std::task::Waker::noop();
        let mut cx = std::task::Context::from_waker(waker);
        match fut.as_mut().poll(&mut cx) {
            std::task::Poll::Ready(v) => Some(v),
            std::task::Poll::Pending => None,
        }
    }

    /// A request at the limit can be dropped on a thread far too small to unwind it
    /// recursively.
    #[test]
    fn deep_request_drops_without_recursion() {
        let v = validate_body(
            body_with_state_depth(MAX_JSON_DEPTH - 1).as_bytes(),
            "typed-decisions",
        )
        .unwrap();
        std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(move || drop(v))
            .unwrap()
            .join()
            .unwrap();
    }
}
