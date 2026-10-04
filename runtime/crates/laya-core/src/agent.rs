//! The high-level runtime: `Agent.predict` / `Agent.predict_batch` over any [`Backend`].
//!
//! Changed in sys1rust from laya-r-mlx 914c9a7: `Timing` records the batch shape; `collate`
//! pads to the backend's `padded_len`; `load` can load the tokenizer on a second thread
//! (`parallel_load`).

use crate::backend::{Backend, BackendOptions, Batch};
use crate::config::ModelConfig;
use crate::decode::{decode_answers, ActHead, Temperatures};
use crate::question::{parse_questions, Question};
use crate::sequence::{collate_to, encode_state, EncodedItem};
use crate::tokenizer::LayaTokenizer;
use crate::weights::Weights;
use crate::{Error, Result, MODEL_NAME};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

/// Rows (state and question pairs) per forward pass when `predict_batch` gets no batch size.
/// 128 rows at `max_len` 512 keep the fp16 attention scores of a 12-head encoder under 1 GiB.
/// sys1d never reaches this: it passes one state per request, bounded by its request limits.
pub const DEFAULT_MAX_ROWS: usize = 128;

/// States per forward pass for `n_questions` questions each, staying under [`DEFAULT_MAX_ROWS`].
pub fn default_chunk(n_questions: usize) -> usize {
    (DEFAULT_MAX_ROWS / n_questions.max(1)).max(1)
}

/// Builds a backend for a checkpoint. Implemented by each backend crate.
pub type BackendFactory<'a> =
    dyn FnOnce(&Weights, &ModelConfig, &BackendOptions) -> Result<Box<dyn Backend>> + 'a;

pub struct Agent {
    pub model_dir: PathBuf,
    pub cfg: ModelConfig,
    pub tokenizer: LayaTokenizer,
    pub temperatures: Temperatures,
    act_head: ActHead,
    backend: Box<dyn Backend>,
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent")
            .field("model_dir", &self.model_dir)
            .field("backend", &self.backend.name())
            .finish()
    }
}

/// Timing of the phases of one `predict_batch` call, for benchmarking.
#[derive(Debug, Clone, Default)]
pub struct Timing {
    pub encode_us: u128,
    pub forward_us: u128,
    pub decode_us: u128,
    /// Padded sequence length and row count of the last forward.
    pub batch_len: usize,
    pub batch_rows: usize,
}

impl Agent {
    /// Load a checkpoint directory (see [`crate::resolve::resolve_model_dir`]) with a backend.
    ///
    /// With the `parallel_load` setting ([`BackendOptions::parallel_load`]) the tokenizer
    /// loads on a second thread while this one opens the weights and builds the backend. The
    /// agent is the same, and so is the error when loading fails: the tokenizer's first, as
    /// when the parts load one after the other. A tokenizer file that is not there, or a
    /// tokenizer that has failed by the time the weights are open, stops the load before the
    /// backend is built, also as in order.
    pub fn load(
        model_dir: &Path,
        opts: &BackendOptions,
        make_backend: Box<BackendFactory<'_>>,
    ) -> Result<Self> {
        let cfg = ModelConfig::load(model_dir)?;
        // Without the tokenizer file the load runs in order, so the tokenizer's error comes
        // before any backend work.
        let parallel = opts.parallel_load()? && LayaTokenizer::file(model_dir).is_file();
        // The weights and the backend load on the calling thread: MLX keeps its default device
        // and stream per thread, so the backend is built where it was before.
        let open = || -> Result<(Weights, ActHead)> {
            let weights = Weights::open(model_dir)?;
            weights.verify()?;
            let act_head = ActHead::load(&weights)?;
            Ok((weights, act_head))
        };
        let (tokenizer, act_head, backend) = if parallel {
            std::thread::scope(|s| -> Result<_> {
                let worker = std::thread::Builder::new()
                    .name("laya-tokenizer".into())
                    .spawn_scoped(s, || LayaTokenizer::load(model_dir));
                let Ok(worker) = worker else {
                    // No second thread: load in order on this one.
                    let tokenizer = LayaTokenizer::load(model_dir)?;
                    let (weights, act_head) = open()?;
                    return Ok((tokenizer, act_head, make_backend(&weights, &cfg, opts)?));
                };
                // Every path joins the worker before it returns, so the scope never re-raises
                // the worker's panic.
                let join = |worker: std::thread::ScopedJoinHandle<'_, Result<LayaTokenizer>>| {
                    worker.join().unwrap_or_else(|_| {
                        Err(Error::Tokenizer(
                            "the tokenizer loading thread panicked".into(),
                        ))
                    })
                };
                let opened = open();
                // A tokenizer that has already failed stops the load here, before the backend
                // build (MLX's start and the weight upload). This looks once and never waits:
                // a tokenizer that fails later, while the backend builds, is reported after it.
                let done = if worker.is_finished() {
                    Ok(join(worker)?)
                } else {
                    Err(worker)
                };
                let built = opened.and_then(|(weights, act_head)| {
                    Ok((act_head, make_backend(&weights, &cfg, opts)?))
                });
                let tokenizer = match done {
                    Ok(tokenizer) => tokenizer,
                    Err(worker) => join(worker)?,
                };
                let (act_head, backend) = built?;
                Ok((tokenizer, act_head, backend))
            })?
        } else {
            let tokenizer = LayaTokenizer::load(model_dir)?;
            let (weights, act_head) = open()?;
            let backend = make_backend(&weights, &cfg, opts)?;
            (tokenizer, act_head, backend)
        };
        let mut agent = Self::from_parts(cfg, tokenizer, act_head, backend)?;
        agent.model_dir = model_dir.to_path_buf();
        Ok(agent)
    }

    /// Assemble an agent from already loaded parts. The action head must read the encoder
    /// hidden size plus its 4 scalar features, or the rows of the backend's pooled output
    /// would be sliced at the wrong width, and it must have `len(act_costs) + 1` output
    /// classes, the shape upstream builds it with and loads strictly.
    pub fn from_parts(
        cfg: ModelConfig,
        tokenizer: LayaTokenizer,
        act_head: ActHead,
        backend: Box<dyn Backend>,
    ) -> Result<Self> {
        act_head.check_pooled_width(cfg.hidden_size())?;
        act_head.check_n_act(cfg.n_act())?;
        let temperatures = Temperatures::from_config(&cfg.agent);
        Ok(Self {
            model_dir: PathBuf::new(),
            cfg,
            tokenizer,
            temperatures,
            act_head,
            backend,
        })
    }

    pub fn backend_name(&self) -> String {
        self.backend.name()
    }

    pub fn backend(&self) -> &dyn Backend {
        self.backend.as_ref()
    }

    /// Tokenize one state against validated questions (exposed for parity tests).
    pub fn encode(&self, state: &Value, questions: &[Question]) -> Result<Vec<EncodedItem>> {
        encode_state(
            &self.tokenizer,
            state,
            questions,
            self.cfg.agent.max_len,
            self.cfg.agent.head_max_len,
        )
    }

    pub fn collate(&self, items: &[EncodedItem]) -> Batch {
        let len = items.iter().map(|it| it.ids.len()).max().unwrap_or(0);
        let len = self.backend.padded_len(len, items.len());
        collate_to(items, self.tokenizer.pad_id, len)
    }

    /// `Agent.predict` / `system_one`: one state, a `{qid: definition}` object.
    pub fn predict(&self, state: &Value, questions: &Value) -> Result<Value> {
        Ok(self
            .predict_batch(std::slice::from_ref(state), questions, None)?
            .remove(0))
    }

    /// `Agent.predict_batch`: the same questions over many states, packed into shared forward
    /// passes of `batch_size` states each, or [`default_chunk`] states when it is `None` or 0.
    pub fn predict_batch(
        &self,
        states: &[Value],
        questions: &Value,
        batch_size: Option<usize>,
    ) -> Result<Vec<Value>> {
        Ok(self.predict_batch_timed(states, questions, batch_size)?.0)
    }

    pub fn predict_batch_timed(
        &self,
        states: &[Value],
        questions: &Value,
        batch_size: Option<usize>,
    ) -> Result<(Vec<Value>, Timing)> {
        let mut timing = Timing::default();
        if states.is_empty() {
            return Ok((Vec::new(), timing));
        }
        let qs = parse_questions(questions)?;
        if qs.is_empty() {
            let empty = json!({ "model": MODEL_NAME, "answers": {}, "usage": { "input_tokens": 0, "output_tokens": 0 } });
            return Ok((vec![empty; states.len()], timing));
        }
        // Without an explicit batch size, cap the rows per forward pass so a large call does not
        // allocate ids, masks and activations for every state at once. One state is never split.
        let chunk = match batch_size {
            Some(b) if b > 0 => b,
            _ => default_chunk(qs.len()),
        };
        let mut results = Vec::with_capacity(states.len());
        for part in states.chunks(chunk) {
            let t0 = std::time::Instant::now();
            let mut items = Vec::with_capacity(part.len() * qs.len());
            for st in part {
                items.extend(self.encode(st, &qs)?);
            }
            let batch = self.collate(&items);
            timing.encode_us += t0.elapsed().as_micros();
            timing.batch_len = batch.len;
            timing.batch_rows = batch.n;

            let t1 = std::time::Instant::now();
            let out = self.backend.forward(&batch)?;
            timing.forward_us += t1.elapsed().as_micros();

            let t2 = std::time::Instant::now();
            let nq = qs.len();
            for (si, _) in part.iter().enumerate() {
                let offset = si * nq;
                let n_tokens: usize = batch.seq_lens[offset..offset + nq].iter().sum();
                let answers = decode_answers(
                    &out,
                    &batch,
                    &self.act_head,
                    &self.temperatures,
                    &qs,
                    offset,
                )?;
                results.push(json!({
                    "model": MODEL_NAME,
                    "answers": answers,
                    "usage": { "input_tokens": n_tokens, "output_tokens": 0 },
                }));
            }
            timing.decode_us += t2.elapsed().as_micros();
        }
        Ok((results, timing))
    }

    /// JSON-string convenience for FFI: `state` may be any JSON value (a bare string is text).
    pub fn predict_json(&self, state_json: &str, questions_json: &str) -> Result<String> {
        let state: Value = serde_json::from_str(state_json)?;
        let questions: Value = serde_json::from_str(questions_json)?;
        Ok(serde_json::to_string(&self.predict(&state, &questions)?)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AgentConfig, EncoderConfig};
    use crate::testing::SyntheticBackend;
    use serde_json::json;

    const HIDDEN: usize = 8;

    /// The `tokenizer.json` of [`test_tokenizer`].
    fn tokenizer_spec() -> Value {
        let words = [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "question", ":", "choice", "score",
            "noul", "yes", "no", "level", "0", "1", "2", "3", "refund", "billing", "the", "a",
            "Which", "team", "?", "How", "urgent", "Is", "money", "involved",
        ];
        let vocab: serde_json::Map<String, Value> = words
            .iter()
            .enumerate()
            .map(|(i, w)| (w.to_string(), Value::from(i as u32)))
            .collect();
        json!({
            "version": "1.0",
            "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"},
            "pre_tokenizer": {"type": "Whitespace"},
        })
    }

    /// A whitespace word-level tokenizer over a few words; anything else is `[UNK]`.
    fn test_tokenizer() -> LayaTokenizer {
        let spec = serde_json::to_vec(&tokenizer_spec()).unwrap();
        let tok = tokenizers::Tokenizer::from_bytes(spec).unwrap();
        LayaTokenizer::from_tokenizer(tok, &Value::Null).unwrap()
    }

    /// An action head with small, varied weights so `act_probability` depends on the row.
    fn test_act_head(d_in: usize) -> ActHead {
        test_act_head_with_classes(d_in, 2)
    }

    fn test_act_head_with_classes(d_in: usize, n_act: usize) -> ActHead {
        let [(_, s0, w0), (_, _, b0), (_, s2, w2), (_, _, b2)] = act_head_tensors(d_in, n_act);
        ActHead::from_tensors(s0, w0, b0, s2, w2, b2).unwrap()
    }

    /// The `act_head.*` tensors of [`test_act_head_with_classes`]: name, shape, values.
    fn act_head_tensors(d_in: usize, n_act: usize) -> [(&'static str, Vec<usize>, Vec<f32>); 4] {
        let h = 6;
        let w0 = (0..h * d_in)
            .map(|i| ((i * 7 % 11) as f32 - 5.0) / 10.0)
            .collect();
        let b0 = (0..h).map(|i| i as f32 / 10.0).collect();
        let w2 = (0..n_act * h)
            .map(|i| ((i * 5 % 7) as f32 - 3.0) / 10.0)
            .collect();
        let b2 = (0..n_act).map(|i| 0.1 - 0.2 * i as f32).collect();
        [
            ("act_head.0.weight", vec![h, d_in], w0),
            ("act_head.0.bias", vec![h], b0),
            ("act_head.2.weight", vec![n_act, h], w2),
            ("act_head.2.bias", vec![n_act], b2),
        ]
    }

    /// `encoder/config.json` and `rl_agent_config.json` of [`test_config`].
    fn config_files() -> (Value, Value) {
        let encoder = json!({
            "model_type": "modernbert", "vocab_size": 32, "hidden_size": HIDDEN,
            "num_hidden_layers": 2, "num_attention_heads": 2, "intermediate_size": 16,
        });
        let agent = json!({
            "encoder": "test", "max_len": 64, "head_max_len": 32,
            "act_costs": {"escalate": 1.0}, "temperature": [1.2, 0.8, 1.0],
        });
        (encoder, agent)
    }

    fn test_config() -> ModelConfig {
        let (encoder, agent) = config_files();
        let encoder = EncoderConfig::from_value(&encoder).unwrap();
        let agent: AgentConfig = serde_json::from_value(agent).unwrap();
        ModelConfig { agent, encoder }
    }

    /// An agent over [`SyntheticBackend`] with batches padded to a multiple of `pad_to`.
    fn test_agent(pad_to: usize) -> Agent {
        let backend = SyntheticBackend {
            hidden: HIDDEN,
            pad_to,
        };
        Agent::from_parts(
            test_config(),
            test_tokenizer(),
            test_act_head(HIDDEN + 4),
            Box::new(backend),
        )
        .unwrap()
    }

    fn questions() -> Value {
        json!({
            "route": {"type": "choice", "instructions": "Which team ?",
                      "criteria": {"billing": "refund the a", "tech": null, "other": ""}},
            "urgency": {"type": "score", "instructions": "How urgent ?",
                        "criteria": ["none", "low", "high", "now"]},
            "money": {"type": "noul", "instructions": "Is money involved ?"},
        })
    }

    /// States of different token lengths, including a dict and a list (truncated from the
    /// left), so a batch pads rows differently from a single-state call.
    fn states() -> Vec<Value> {
        vec![
            json!("refund the billing"),
            json!("a a a the the no yes level 0 1 2 3 refund billing question"),
            json!({"from": "a", "text": "the refund"}),
            json!([
                "yes",
                "no",
                "the a refund billing yes no yes no the a the a the a"
            ]),
            json!(""),
        ]
    }

    /// Collating several states into one forward, with the backend's padding, gives every
    /// state the answers it gets on its own: the same decisions, probabilities, confidences,
    /// `act_probability` and token counts.
    #[test]
    fn a_batch_answers_like_one_state_at_a_time() {
        let (questions, states) = (questions(), states());
        for pad_to in [1, 16] {
            let agent = test_agent(pad_to);
            let single: Vec<Value> = states
                .iter()
                .map(|s| agent.predict(s, &questions).unwrap())
                .collect();
            // Every state in one forward, then chunks of two and of one.
            for batch_size in [None, Some(2), Some(1)] {
                let (batched, timing) = agent
                    .predict_batch_timed(&states, &questions, batch_size)
                    .unwrap();
                assert_eq!(
                    batched, single,
                    "pad_to {pad_to}, batch_size {batch_size:?}"
                );
                assert_eq!(
                    timing.batch_len % pad_to,
                    0,
                    "padded to the backend's bucket"
                );
            }
            let (_, timing) = agent
                .predict_batch_timed(&states, &questions, None)
                .unwrap();
            assert_eq!(
                timing.batch_rows,
                states.len() * 3,
                "one forward for all rows"
            );
            // The order of states in the batch does not leak between rows.
            let reversed: Vec<Value> = states.iter().rev().cloned().collect();
            let batched = agent.predict_batch(&reversed, &questions, None).unwrap();
            assert_eq!(batched, single.iter().rev().cloned().collect::<Vec<_>>());
            // The backend is not constant: different states get different answers.
            assert_ne!(single[0]["answers"], single[1]["answers"]);
            assert_ne!(single[0]["usage"], single[1]["usage"]);
            let route = &single[0]["answers"]["route"];
            assert!(["billing", "tech", "other"].contains(&route["choice"].as_str().unwrap()));
        }
    }

    #[test]
    fn from_parts_rejects_an_action_head_of_another_width() {
        let e = Agent::from_parts(
            test_config(),
            test_tokenizer(),
            test_act_head(HIDDEN + 5),
            Box::new(SyntheticBackend {
                hidden: HIDDEN,
                pad_to: 1,
            }),
        )
        .err()
        .map(|e| e.to_string())
        .unwrap();
        assert!(
            e.contains("input width 13") && e.contains("hidden size 8"),
            "{e}"
        );
    }

    /// `test_config` lists one act cost, so the head must have two classes: a three-class
    /// head is refused at assembly, before any `act_probability` is read from it.
    #[test]
    fn from_parts_rejects_an_action_head_of_another_class_count() {
        let e = Agent::from_parts(
            test_config(),
            test_tokenizer(),
            test_act_head_with_classes(HIDDEN + 4, 3),
            Box::new(SyntheticBackend {
                hidden: HIDDEN,
                pad_to: 1,
            }),
        )
        .err()
        .map(|e| e.to_string())
        .unwrap();
        assert!(
            e.contains("3 output classes") && e.contains("lists 1 act_costs"),
            "{e}"
        );
        assert_eq!(test_config().n_act(), 2);
    }

    /// A checkpoint directory of the test model under the system temp dir, removed on drop:
    /// the [`config_files`], the [`tokenizer_spec`], and a `model.safetensors` with the
    /// [`act_head_tensors`] and one tensor of each family [`Weights::verify`] asks for.
    struct ModelDir(PathBuf);

    impl ModelDir {
        fn new() -> Self {
            use safetensors::tensor::TensorView;
            use std::sync::atomic::{AtomicUsize, Ordering};
            static N: AtomicUsize = AtomicUsize::new(0);
            let dir = std::env::temp_dir().join(format!(
                "laya-core-agent-{}-{}",
                std::process::id(),
                N.fetch_add(1, Ordering::Relaxed)
            ));
            let (encoder, agent) = config_files();
            for (path, value) in [
                ("encoder/config.json", encoder),
                ("rl_agent_config.json", agent),
                ("tokenizer/tokenizer.json", tokenizer_spec()),
            ] {
                let path = dir.join(path);
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                std::fs::write(path, serde_json::to_vec(&value).unwrap()).unwrap();
            }
            let mut tensors: Vec<(&str, Vec<usize>, Vec<u8>)> = act_head_tensors(HIDDEN + 4, 2)
                .into_iter()
                .map(|(name, shape, v)| {
                    (
                        name,
                        shape,
                        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
                    )
                })
                .collect();
            for name in ["encoder.w", "type_emb.weight", "scorer.w"] {
                tensors.push((name, vec![2], vec![0; 8]));
            }
            let views = tensors.iter().map(|(name, shape, bytes)| {
                let view = TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes);
                (*name, view.unwrap())
            });
            let bytes = safetensors::serialize(views, None).unwrap();
            std::fs::write(dir.join("model.safetensors"), bytes).unwrap();
            Self(dir)
        }
    }

    impl Drop for ModelDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// [`Agent::load`] with `tuning` and a factory that builds a [`SyntheticBackend`] padding
    /// to 16, or fails with `backend_error`. The factory checks it runs on the calling thread.
    fn load_with(dir: &Path, tuning: &str, backend_error: Option<&str>) -> Result<Agent> {
        load_seen(dir, tuning, backend_error).0
    }

    /// [`load_with`], and whether the load called the factory.
    fn load_seen(dir: &Path, tuning: &str, backend_error: Option<&str>) -> (Result<Agent>, bool) {
        let caller = std::thread::current().id();
        let opts = BackendOptions {
            tuning: Some(tuning.into()),
            ..Default::default()
        };
        let backend_error = backend_error.map(str::to_string);
        let called = std::cell::Cell::new(false);
        let agent = Agent::load(
            dir,
            &opts,
            Box::new(|_: &Weights, cfg: &ModelConfig, _: &BackendOptions| {
                assert_eq!(
                    std::thread::current().id(),
                    caller,
                    "backend built off the calling thread"
                );
                called.set(true);
                match backend_error {
                    Some(e) => Err(Error::Backend(e)),
                    None => Ok(Box::new(SyntheticBackend {
                        hidden: cfg.hidden_size(),
                        pad_to: 16,
                    }) as Box<dyn Backend>),
                }
            }),
        );
        (agent, called.get())
    }

    /// `parallel_load` loads the same agent as loading in order, and the same as the one
    /// assembled from the same parts in memory.
    #[test]
    fn parallel_load_loads_the_same_agent() {
        let dir = ModelDir::new();
        let in_order = load_with(&dir.0, "", None).unwrap();
        let parallel = load_with(&dir.0, "f16gelu,parallel_load", None).unwrap();
        let parts = test_agent(16);
        assert_eq!(parallel.model_dir, dir.0);
        let questions = questions();
        for state in states() {
            let want = parts.predict(&state, &questions).unwrap();
            assert_eq!(in_order.predict(&state, &questions).unwrap(), want);
            assert_eq!(parallel.predict(&state, &questions).unwrap(), want);
        }
    }

    /// With `parallel_load` every failed load is the same error as loading in order, never a
    /// panic: a missing or broken tokenizer, missing or cut weights, a backend that fails, and
    /// both the tokenizer and the weights missing (the tokenizer's error comes first). A bad
    /// value of the setting fails before anything loads.
    #[test]
    fn parallel_load_fails_like_loading_in_order() {
        fn remove(dir: &Path, file: &str) {
            std::fs::remove_file(dir.join(file)).unwrap();
        }
        fn cut(dir: &Path) {
            let path = dir.join("model.safetensors");
            let bytes = std::fs::read(&path).unwrap();
            std::fs::write(&path, &bytes[..bytes.len() / 2]).unwrap();
        }
        type Break = fn(&Path);
        let cases: [(&str, Break, Option<&str>, &str); 6] = [
            (
                "no tokenizer",
                |d| remove(d, "tokenizer/tokenizer.json"),
                None,
                "tokenizer: ",
            ),
            (
                "broken tokenizer",
                |d| std::fs::write(d.join("tokenizer/tokenizer.json"), b"{").unwrap(),
                None,
                "tokenizer: ",
            ),
            (
                "no weights",
                |d| remove(d, "model.safetensors"),
                None,
                "weights: 'model.safetensors' not found",
            ),
            (
                "cut weights",
                cut,
                None,
                "is truncated or not a safetensors file",
            ),
            (
                "backend fails",
                |_| {},
                Some("no device"),
                "backend: no device",
            ),
            (
                "no tokenizer, no weights",
                |d| {
                    remove(d, "tokenizer/tokenizer.json");
                    remove(d, "model.safetensors");
                },
                None,
                "tokenizer: ",
            ),
        ];
        for (name, break_it, backend_error, want) in cases {
            let dir = ModelDir::new();
            break_it(&dir.0);
            let in_order = load_with(&dir.0, "", backend_error)
                .unwrap_err()
                .to_string();
            let parallel = load_with(&dir.0, "parallel_load", backend_error)
                .unwrap_err()
                .to_string();
            assert!(in_order.contains(want), "{name}: {in_order}");
            assert_eq!(parallel, in_order, "{name}");
        }
        let dir = ModelDir::new();
        let e = load_with(&dir.0, "parallel_load=2", Some("built anyway"))
            .unwrap_err()
            .to_string();
        assert!(e.contains("`parallel_load=2`"), "{e}");
    }

    /// A missing tokenizer file fails the load before the backend is built, with
    /// `parallel_load` as in order: the backend build is where MLX starts and uploads the
    /// weights.
    #[test]
    fn a_missing_tokenizer_fails_before_the_backend() {
        let dir = ModelDir::new();
        std::fs::remove_file(dir.0.join("tokenizer/tokenizer.json")).unwrap();
        for tuning in ["", "parallel_load"] {
            let (agent, built) = load_seen(&dir.0, tuning, None);
            let e = agent.unwrap_err().to_string();
            assert!(e.starts_with("tokenizer: "), "{tuning:?}: {e}");
            assert!(!built, "{tuning:?}: the backend was built");
        }
    }

    #[test]
    fn default_chunk_bounds_rows_and_keeps_whole_states() {
        assert_eq!(default_chunk(1), DEFAULT_MAX_ROWS);
        assert_eq!(default_chunk(3), DEFAULT_MAX_ROWS / 3);
        assert!(default_chunk(3) * 3 <= DEFAULT_MAX_ROWS);
        // More questions than the row cap still runs one state per pass.
        assert_eq!(default_chunk(DEFAULT_MAX_ROWS + 1), 1);
        assert_eq!(default_chunk(0), DEFAULT_MAX_ROWS);
    }
}
