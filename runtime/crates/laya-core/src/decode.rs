//! Temperature scaling, calibrated confidence, the action head and typed answer decoding
//! (`Agent._decode_answers`, `DecisionModel.forward`'s act-head tail, `confidence_from_probs`).
//!
//! Changed in sys1rust from laya-r-mlx 914c9a7: every answer carries `answer_confidence`
//! (laya 0.3.21), the probability mass on the reported answer, between `confidence` and
//! `action`.

use crate::backend::{BackendOutput, Batch};
use crate::config::AgentConfig;
use crate::pyjson::{self, dumps_key, round_dp};
use crate::question::{QType, Question};
use crate::weights::Weights;
use crate::{Error, Result};
use serde_json::{json, Map, Value};

pub const TEMP_MIN: f64 = 0.5;
pub const TEMP_MAX: f64 = 5.0;

/// `clamp_temperature`: a usable temperature confined to `[0.5, 5.0]`, `1.0` when not a number.
pub fn clamp_temperature(v: &Value) -> f64 {
    let t = match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    };
    match t {
        Some(t) if t.is_finite() => t.clamp(TEMP_MIN, TEMP_MAX),
        _ => 1.0,
    }
}

/// Calibration temperatures as the runtime applies them (already clamped).
#[derive(Debug, Clone)]
pub struct Temperatures {
    pub by_type: [f64; 3],
    pub by_options: Map<String, Value>,
    /// Entries the checkpoint shipped that were rejected/clamped (`name=raw -> applied`).
    pub rejected: Vec<String>,
}

impl Temperatures {
    pub fn from_config(cfg: &AgentConfig) -> Self {
        let mut by_type = [1.0; 3];
        let mut rejected = Vec::new();
        for (i, raw) in cfg.temperature.iter().enumerate().take(3) {
            by_type[i] = clamp_temperature(raw);
            if raw.as_f64() != Some(by_type[i]) {
                rejected.push(format!("temperature[{i}]={raw} -> {}", by_type[i]));
            }
        }
        let mut by_options = Map::new();
        for (k, raw) in &cfg.temperature_by_options {
            let t = clamp_temperature(raw);
            if raw.as_f64() != Some(t) {
                rejected.push(format!("{k}={raw} -> {t}"));
            }
            by_options.insert(k.clone(), Value::from(t));
        }
        Self {
            by_type,
            by_options,
            rejected,
        }
    }

    pub fn temp_bucket(qtype: QType, k: usize) -> String {
        let size = if k <= 2 {
            "2"
        } else if k <= 5 {
            "3-5"
        } else if k <= 10 {
            "6-10"
        } else {
            "11+"
        };
        format!("{}:{}", qtype.name(), size)
    }

    pub fn scale(&self, qtype: QType, k: usize) -> f64 {
        self.by_options
            .get(&Self::temp_bucket(qtype, k))
            .and_then(Value::as_f64)
            .unwrap_or(self.by_type[qtype.index()])
    }
}

/// `confidence_from_probs`: normalized Shannon entropy confidence `1 - H(p) / log(k)`.
pub fn confidence_from_probs(p: &[f64], k: usize) -> f64 {
    if k < 2 {
        return 1.0;
    }
    let ent: f64 = p[..k].iter().map(|&x| -x * x.clamp(1e-12, 1.0).ln()).sum();
    (1.0 - ent / (k as f64).ln()).clamp(0.0, 1.0)
}

fn softmax_f64(z: &[f64]) -> Vec<f64> {
    let m = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = z.iter().map(|&x| (x - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|x| x / s).collect()
}

fn softmax_f32(z: &[f32]) -> Vec<f32> {
    let m = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = z.iter().map(|&x| (x - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.into_iter().map(|x| x / s).collect()
}

/// Exact (erf) GELU, as `torch.nn.GELU()` / `F.gelu` default.
pub fn gelu(x: f32) -> f32 {
    let xf = x as f64;
    (0.5 * xf * (1.0 + libm::erf(xf / std::f64::consts::SQRT_2))) as f32
}

/// `act_head = Linear(d + 4, 256) -> GELU -> Linear(256, n_act)`, evaluated on the CPU in f32.
#[derive(Debug, Clone)]
pub struct ActHead {
    pub d_in: usize,
    pub hidden: usize,
    pub n_act: usize,
    w0: Vec<f32>,
    b0: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

impl ActHead {
    pub fn load(w: &Weights) -> Result<Self> {
        let (s0, w0) = w.tensor_f32("act_head.0.weight")?;
        let (_, b0) = w.tensor_f32("act_head.0.bias")?;
        let (s2, w2) = w.tensor_f32("act_head.2.weight")?;
        let (_, b2) = w.tensor_f32("act_head.2.bias")?;
        Self::from_tensors(s0, w0, b0, s2, w2, b2)
    }

    /// Build the head from the two layers' `(shape, row-major data)`, with the same checks as
    /// [`ActHead::load`]. For tests and embedders that hold the tensors already.
    pub fn from_tensors(
        s0: Vec<usize>,
        w0: Vec<f32>,
        b0: Vec<f32>,
        s2: Vec<usize>,
        w2: Vec<f32>,
        b2: Vec<f32>,
    ) -> Result<Self> {
        if s0.len() != 2 || s2.len() != 2 || s2[1] != s0[0] {
            return Err(Error::Weights(format!(
                "act_head shapes {s0:?} / {s2:?} are not a 2-layer MLP"
            )));
        }
        let (hidden, d_in, n_act) = (s0[0], s0[1], s2[0]);
        // The forward reads `d_in - 4` pooled values and then the 4 scalar features, so a
        // width of 4 or less has nothing to read the hidden state into.
        if d_in <= 4 {
            return Err(Error::Weights(format!(
                "act_head.0.weight has input width {d_in}, but the action head reads the pooled hidden state plus 4 scalar features, so it needs at least 5"
            )));
        }
        if hidden == 0 || n_act == 0 {
            return Err(Error::Weights(format!(
                "act_head shapes {s0:?} / {s2:?} have an empty layer"
            )));
        }
        if b0.len() != hidden || b2.len() != n_act {
            return Err(Error::Weights(format!(
                "act_head biases have {} and {} values; the layers have {hidden} and {n_act} outputs",
                b0.len(),
                b2.len()
            )));
        }
        if w0.len() != hidden * d_in || w2.len() != n_act * hidden {
            return Err(Error::Weights(format!(
                "act_head weights have {} and {} values; shapes {s0:?} / {s2:?} need {} and {}",
                w0.len(),
                w2.len(),
                hidden * d_in,
                n_act * hidden
            )));
        }
        Ok(Self {
            d_in,
            hidden,
            n_act,
            w0,
            b0,
            w2,
            b2,
        })
    }

    /// Width of the pooled hidden state the head reads: its input minus the 4 scalar features.
    pub fn pooled_width(&self) -> usize {
        self.d_in - 4
    }

    /// The pooled width must be the encoder hidden size the backend returns per row, or the
    /// row slicing in [`decode_answers`] reads the wrong values.
    pub fn check_pooled_width(&self, hidden_size: usize) -> Result<()> {
        if self.pooled_width() != hidden_size {
            return Err(Error::Weights(format!(
                "act_head.0.weight has input width {}, which is not the encoder hidden size {hidden_size} plus 4 scalar features",
                self.d_in
            )));
        }
        Ok(())
    }

    /// The head's output classes must be `len(act_costs) + 1`: upstream builds
    /// `nn.Linear(256, n_act)` from the config and `load_state_dict(strict=True)` refuses a
    /// checkpoint of another width, so `act_probability` never comes from a head trained for
    /// a different action set.
    pub fn check_n_act(&self, n_act: usize) -> Result<()> {
        if self.n_act != n_act {
            return Err(Error::Weights(format!(
                "act_head.2.weight has {} output classes, but rl_agent_config.json lists {} act_costs, so the action head must have {n_act}",
                self.n_act,
                n_act.saturating_sub(1)
            )));
        }
        Ok(())
    }

    /// Action-class probabilities for one row.
    pub fn probs(&self, pooled: &[f32], feats: [f32; 4]) -> Vec<f32> {
        debug_assert_eq!(pooled.len() + 4, self.d_in);
        let mut h = vec![0f32; self.hidden];
        for (j, hj) in h.iter_mut().enumerate() {
            let row = &self.w0[j * self.d_in..(j + 1) * self.d_in];
            let mut acc = self.b0[j];
            for (a, b) in row[..pooled.len()].iter().zip(pooled) {
                acc += a * b;
            }
            for (a, b) in row[pooled.len()..].iter().zip(feats.iter()) {
                acc += a * b;
            }
            *hj = gelu(acc);
        }
        let mut out = vec![0f32; self.n_act];
        for (j, oj) in out.iter_mut().enumerate() {
            let row = &self.w2[j * self.hidden..(j + 1) * self.hidden];
            *oj = self.b2[j] + row.iter().zip(&h).map(|(a, b)| a * b).sum::<f32>();
        }
        softmax_f32(&out)
    }
}

/// The four scalar features the action head sees, from one row of masked logits.
pub fn act_features(logits_row: &[f32], marker_count: usize) -> [f32; 4] {
    let p = softmax_f32(logits_row);
    let k = marker_count.max(2) as f32;
    let ent: f32 = -p.iter().map(|&x| x * x.max(1e-9).ln()).sum::<f32>() / k.ln();
    let mut sorted = p.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let top1 = sorted.first().copied().unwrap_or(0.0);
    let top2 = sorted.get(1).copied().unwrap_or(0.0);
    [top1, top1 - top2, ent, k / 255.0]
}

/// Upstream's error for a float that is not finite in the result. `laya serve` builds
/// `JSONResponse(content=result)` inside its `try` block, Starlette renders it with
/// `json.dumps(allow_nan=False)`, and the `ValueError` that raises (for a value or a dict
/// key alike) is what the handler turns into a 422 with `detail = str(e)`. sys1rust maps
/// `Error::Question` to the same status and detail, so the client sees upstream's text.
fn out_of_range(x: f64) -> Error {
    Error::Question(format!(
        "Out of range float values are not JSON compliant: {}",
        pyjson::float_repr(x)
    ))
}

/// A request value as Python's `json.dumps` prints it, for the result fields that copy a
/// value from the request: a choice label and a score legend entry. With
/// `arbitrary_precision` a Number keeps its literal text and every serde serializer writes it
/// back verbatim, so a value typed `1E5`, `1.10` or `-0` would go out as typed where upstream
/// prints `100000.0`, `1.1` and `0`. Re-parsing [`pyjson::dumps`] of the number puts Python's
/// text in the Number instead; a list or dict is walked for the numbers inside it. A
/// non-finite float (`1e400`) has no JSON text: upstream fails the whole response when it
/// serializes it, so this is [`out_of_range`] with the same message.
fn python_value(v: &Value) -> Result<Value> {
    match v {
        Value::Number(n) => {
            if let Some(x) = pyjson::non_finite(n) {
                return Err(out_of_range(x));
            }
            Ok(serde_json::from_str(&pyjson::dumps(v))?)
        }
        Value::Array(a) => Ok(Value::Array(
            a.iter().map(python_value).collect::<Result<Vec<_>>>()?,
        )),
        Value::Object(o) => Ok(Value::Object(
            o.iter()
                .map(|(k, x)| Ok((k.clone(), python_value(x)?)))
                .collect::<Result<Map<_, _>>>()?,
        )),
        other => Ok(other.clone()),
    }
}

/// The `probabilities` key for a choice label, or [`out_of_range`] for a float label with no
/// JSON text: Python raises for a non-finite dict key as it does for a value.
fn label_key(label: &Value) -> Result<String> {
    python_value(label)?;
    Ok(dumps_key(label))
}

/// Decode one state's rows (`rows[0]..rows[n]` of `out`) into the Python result object.
pub fn decode_answers(
    out: &BackendOutput,
    batch: &Batch,
    act_head: &ActHead,
    temps: &Temperatures,
    questions: &[Question],
    offset: usize,
) -> Result<Value> {
    let d = act_head.pooled_width();
    let mut answers = Map::with_capacity(questions.len());
    for (j, q) in questions.iter().enumerate() {
        let r = offset + j;
        let k = batch.marker_count[r];
        let row = &out.logits[r * batch.kmax..(r + 1) * batch.kmax];
        let pooled = &out.pooled[r * d..(r + 1) * d];
        let act = act_head.probs(pooled, act_features(row, k));
        let act_probability = round_dp(act[0] as f64, 4);

        let t_scale = temps.scale(q.qtype, k);
        let z: Vec<f64> = row[..k].iter().map(|&x| x as f64 / t_scale).collect();
        let p = softmax_f64(&z);
        let conf = round_dp(confidence_from_probs(&p, k), 4);
        let ext = json!({ "act_probability": act_probability });

        let ans = match q.qtype {
            QType::Choice => {
                let argmax = p
                    .iter()
                    .enumerate()
                    .fold(0, |b, (i, &x)| if x > p[b] { i } else { b });
                // `keys = list(q["crit"].keys())`: the raw labels. The answer is the label
                // itself; the probabilities map is keyed by what `json.dumps` writes for it.
                // (`"1"` and `1` are distinct Python keys with one JSON spelling; upstream
                // then emits a duplicate key, which a Map cannot hold, so the last one wins
                // here as it does in any client that parses upstream's text.)
                // `choice` is serialized before `probabilities`, so a non-finite label fails
                // on the winner first, then on the first such label in option order.
                let choice = python_value(&q.choice_labels[argmax])?;
                let mut probs = Map::new();
                for (label, &v) in q.choice_labels.iter().zip(&p) {
                    probs.insert(label_key(label)?, Value::from(round_dp(v, 4)));
                }
                json!({
                    "type": "choice",
                    "choice": choice,
                    "probabilities": probs,
                    "confidence": conf,
                    "answer_confidence": round_dp(p[argmax], 4),
                    "action": ext,
                })
            }
            QType::Score => {
                let exp_score: f64 = p.iter().enumerate().map(|(i, &x)| i as f64 * x).sum();
                let mut legend = Map::new();
                let mut probs = Map::new();
                // `{str(i): c for i, c in enumerate(q["crit"])}`: the raw descriptions, with
                // every number inside printed as Python would.
                for (i, c) in q.score_legend.iter().enumerate() {
                    legend.insert(i.to_string(), python_value(c)?);
                }
                for (i, &v) in p.iter().enumerate() {
                    probs.insert(i.to_string(), Value::from(round_dp(v, 4)));
                }
                let top = p.iter().cloned().fold(0.0, f64::max);
                json!({
                    "type": "score",
                    "score": round_dp(exp_score, 4),
                    "legend": legend,
                    "probabilities": probs,
                    "confidence": conf,
                    "answer_confidence": round_dp(top, 4),
                    "action": ext,
                })
            }
            QType::Noul => {
                let p1 = p.get(1).copied().unwrap_or(0.0);
                // `max(p_yes, p_no)`: for noul the two confidences are the same number.
                let top = round_dp(p1.max(1.0 - p1), 4);
                json!({
                    "type": "noul",
                    "noul": round_dp(p1, 4),
                    "confidence": top,
                    "answer_confidence": top,
                    "action": ext,
                })
            }
        };
        answers.insert(q.id.clone(), ans);
    }
    Ok(Value::Object(answers))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn temperature_clamping() {
        assert_eq!(clamp_temperature(&json!(0.1006)), 0.5);
        assert_eq!(clamp_temperature(&json!(7.0)), 5.0);
        assert_eq!(clamp_temperature(&json!("1.5")), 1.5);
        assert_eq!(clamp_temperature(&json!("abc")), 1.0);
        assert_eq!(clamp_temperature(&Value::Null), 1.0);
        assert_eq!(Temperatures::temp_bucket(QType::Choice, 20), "choice:11+");
        assert_eq!(Temperatures::temp_bucket(QType::Noul, 2), "noul:2");
        assert_eq!(Temperatures::temp_bucket(QType::Score, 5), "score:3-5");
        assert_eq!(Temperatures::temp_bucket(QType::Choice, 6), "choice:6-10");
    }

    /// A `model.safetensors` holding only an action head of the given shapes, in a fresh
    /// directory under the system temp dir.
    fn act_head_checkpoint(d_in: usize, hidden: usize, n_act: usize, b0_len: usize) -> PathBuf {
        use safetensors::tensor::TensorView;
        use safetensors::Dtype;
        static N: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "laya-core-act-head-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let zeros = |n: usize| vec![0u8; 4 * n];
        let (w0, b0, w2, b2) = (
            zeros(hidden * d_in),
            zeros(b0_len),
            zeros(n_act * hidden),
            zeros(n_act),
        );
        let tensors = [
            ("act_head.0.weight", vec![hidden, d_in], &w0),
            ("act_head.0.bias", vec![b0_len], &b0),
            ("act_head.2.weight", vec![n_act, hidden], &w2),
            ("act_head.2.bias", vec![n_act], &b2),
        ]
        .into_iter()
        .map(|(name, shape, data)| (name, TensorView::new(Dtype::F32, shape, data).unwrap()));
        std::fs::write(
            dir.join("model.safetensors"),
            safetensors::serialize(tensors, None).unwrap(),
        )
        .unwrap();
        dir
    }

    /// An input width that leaves no room for the pooled hidden state is a weights error at
    /// load time, not an underflow in the first prediction.
    #[test]
    fn act_head_rejects_an_input_width_without_a_hidden_state() {
        for d_in in [0, 3, 4] {
            let dir = act_head_checkpoint(d_in, 8, 2, 8);
            let e = ActHead::load(&Weights::open(&dir).unwrap())
                .unwrap_err()
                .to_string();
            assert!(
                e.contains(&format!("input width {d_in}")) && e.contains("at least 5"),
                "d_in {d_in}: {e}"
            );
            std::fs::remove_dir_all(&dir).unwrap();
        }
        let dir = act_head_checkpoint(6, 8, 2, 7);
        let e = ActHead::load(&Weights::open(&dir).unwrap())
            .unwrap_err()
            .to_string();
        assert!(e.contains("biases have 7 and 2 values"), "{e}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn act_head_width_must_match_the_encoder_hidden_size() {
        let dir = act_head_checkpoint(6, 8, 2, 8);
        let head = ActHead::load(&Weights::open(&dir).unwrap()).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!((head.d_in, head.hidden, head.n_act), (6, 8, 2));
        assert_eq!(head.pooled_width(), 2);
        head.check_pooled_width(2).unwrap();
        let e = head.check_pooled_width(768).unwrap_err().to_string();
        assert!(
            e.contains("input width 6") && e.contains("hidden size 768"),
            "{e}"
        );
        // Zero weights: every action class is equally likely.
        assert_eq!(head.probs(&[1.0, 2.0], [0.5, 0.1, 0.2, 0.01]), [0.5, 0.5]);
    }

    /// A one-class head with a config listing one act cost (two classes) would report
    /// `act_probability` 1.0 for every answer; the mismatch is a weights error instead.
    #[test]
    fn act_head_output_classes_must_match_act_costs() {
        let dir = act_head_checkpoint(6, 8, 1, 8);
        let head = ActHead::load(&Weights::open(&dir).unwrap()).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(head.n_act, 1);
        head.check_n_act(1).unwrap();
        let e = head.check_n_act(2).unwrap_err().to_string();
        assert!(
            e.contains("1 output classes")
                && e.contains("lists 1 act_costs")
                && e.contains("must have 2"),
            "{e}"
        );
        let dir = act_head_checkpoint(6, 8, 3, 8);
        let head = ActHead::load(&Weights::open(&dir).unwrap()).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        head.check_n_act(3).unwrap();
        let e = head.check_n_act(2).unwrap_err().to_string();
        assert!(e.contains("3 output classes"), "{e}");
    }

    #[test]
    fn confidence() {
        assert_eq!(confidence_from_probs(&[1.0], 1), 1.0);
        assert!((confidence_from_probs(&[0.5, 0.5], 2)).abs() < 1e-12);
        assert!((confidence_from_probs(&[1.0, 0.0], 2) - 1.0).abs() < 1e-9);
    }

    /// An act head with zero weights: `act_probability` is a uniform softmax, so the test is
    /// only about the answer fields.
    fn zero_act_head(d: usize) -> ActHead {
        ActHead {
            d_in: d + 4,
            hidden: 1,
            n_act: 2,
            w0: vec![0.0; d + 4],
            b0: vec![0.0],
            w2: vec![0.0; 2],
            b2: vec![0.0; 2],
        }
    }

    fn question(id: &str, qtype: QType, k: usize) -> Question {
        let names: Vec<String> = (0..k).map(|i| format!("o{i}")).collect();
        Question {
            id: id.into(),
            qtype,
            instructions: String::new(),
            options: names.clone(),
            choice_labels: if qtype == QType::Choice {
                names.iter().map(|n| json!(n)).collect()
            } else {
                vec![]
            },
            score_legend: if qtype == QType::Score {
                (0..k).map(|i| json!(format!("level {i}"))).collect()
            } else {
                vec![]
            },
        }
    }

    #[test]
    fn answer_confidence_per_type() {
        // One row per type: choice over 3 options, score over 3 levels, noul (2 options,
        // padded to kmax 3). Temperatures are 1, so the probabilities are plain softmaxes.
        let qs = [
            question("pick", QType::Choice, 3),
            question("rate", QType::Score, 3),
            question("yes", QType::Noul, 2),
        ];
        let batch = Batch {
            n: 3,
            len: 1,
            kmax: 3,
            input_ids: vec![0; 3],
            attention_mask: vec![1; 3],
            seq_lens: vec![1; 3],
            marker_pos: vec![0; 9],
            marker_count: vec![3, 3, 2],
            qtype: vec![0, 1, 2],
        };
        let out = BackendOutput {
            logits: vec![2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, -1e4],
            pooled: vec![0.0; 3 * 2],
        };
        let temps = Temperatures {
            by_type: [1.0; 3],
            by_options: Map::new(),
            rejected: vec![],
        };
        let answers = decode_answers(&out, &batch, &zero_act_head(2), &temps, &qs, 0).unwrap();
        let keys = |a: &Value| -> Vec<String> { a.as_object().unwrap().keys().cloned().collect() };

        let pick = &answers["pick"];
        assert_eq!(
            keys(pick),
            [
                "type",
                "choice",
                "probabilities",
                "confidence",
                "answer_confidence",
                "action"
            ]
        );
        assert_eq!(pick["choice"], "o0");
        // e^2 / (e^2 + 2) = 0.78699, rounded to 4 dp like the probabilities.
        assert_eq!(pick["answer_confidence"], 0.787);
        assert_eq!(pick["answer_confidence"], pick["probabilities"]["o0"]);

        let rate = &answers["rate"];
        assert_eq!(
            keys(rate),
            [
                "type",
                "score",
                "legend",
                "probabilities",
                "confidence",
                "answer_confidence",
                "action"
            ]
        );
        // e / (e + 2) = 0.57612, the largest level probability.
        assert_eq!(rate["answer_confidence"], 0.5761);
        assert_eq!(rate["answer_confidence"], rate["probabilities"]["1"]);

        let yes = &answers["yes"];
        assert_eq!(
            keys(yes),
            ["type", "noul", "confidence", "answer_confidence", "action"]
        );
        // p_yes = 1 / (1 + e) = 0.26894; both confidences are max(p_yes, p_no).
        assert_eq!(yes["noul"], 0.2689);
        assert_eq!(yes["confidence"], 0.7311);
        assert_eq!(yes["answer_confidence"], yes["confidence"]);
    }

    /// List-form labels: `choice` is the raw label with its JSON type, as upstream's
    /// `keys[int(p.argmax())]` returns it, and the `probabilities` keys are the strings
    /// `json.dumps` writes for those labels as dict keys.
    #[test]
    fn choice_answer_keeps_the_raw_label() {
        let labels = vec![json!(2), json!("b"), json!(true), json!(1.5)];
        let q = |id: &str| Question {
            id: id.into(),
            qtype: QType::Choice,
            instructions: String::new(),
            options: ["2", "b", "True", "1.5"].map(String::from).to_vec(),
            choice_labels: labels.clone(),
            score_legend: vec![],
        };
        // One row per winning label: an int, a string, a bool and a float.
        let qs = [q("int"), q("str"), q("bool"), q("float")];
        let batch = Batch {
            n: 4,
            len: 1,
            kmax: 4,
            input_ids: vec![0; 4],
            attention_mask: vec![1; 4],
            seq_lens: vec![1; 4],
            marker_pos: vec![0; 16],
            marker_count: vec![4; 4],
            qtype: vec![0; 4],
        };
        let mut logits = vec![0.0; 16];
        for r in 0..4 {
            logits[r * 4 + r] = 3.0;
        }
        let out = BackendOutput {
            logits,
            pooled: vec![0.0; 4 * 2],
        };
        let temps = Temperatures {
            by_type: [1.0; 3],
            by_options: Map::new(),
            rejected: vec![],
        };
        let answers = decode_answers(&out, &batch, &zero_act_head(2), &temps, &qs, 0).unwrap();

        assert_eq!(answers["int"]["choice"], json!(2));
        assert!(answers["int"]["choice"].is_i64());
        assert_eq!(answers["str"]["choice"], json!("b"));
        assert_eq!(answers["bool"]["choice"], json!(true));
        assert!(answers["bool"]["choice"].is_boolean());
        assert_eq!(answers["float"]["choice"], json!(1.5));
        assert!(answers["float"]["choice"].is_f64());

        for a in ["int", "str", "bool", "float"] {
            let probs = answers[a]["probabilities"].as_object().unwrap();
            let keys: Vec<&String> = probs.keys().collect();
            assert_eq!(keys, ["2", "b", "true", "1.5"], "{a}");
        }
        // e^3 / (e^3 + 3) = 0.87005: the winning label's probability is the answer confidence.
        assert_eq!(answers["bool"]["probabilities"]["true"], 0.87);
        assert_eq!(answers["bool"]["answer_confidence"], 0.87);
        assert_eq!(answers["int"]["probabilities"]["2"], 0.87);
        assert_eq!(answers["float"]["probabilities"]["1.5"], 0.87);
    }

    /// Decode the questions of `text` (a `{qid: definition}` object) with temperatures 1 and
    /// the zero act head. Row `j` gets logit 3 at option `winners[j]` and 0 at the others.
    fn decode_with_winners(text: &str, winners: &[usize]) -> Result<Value> {
        let qs = crate::question::parse_questions(&serde_json::from_str(text).unwrap()).unwrap();
        let n = qs.len();
        let kmax = qs.iter().map(|q| q.options.len()).max().unwrap_or(0);
        let mut logits = vec![-1e4f32; n * kmax];
        for (r, q) in qs.iter().enumerate() {
            for k in 0..q.options.len() {
                logits[r * kmax + k] = if k == winners[r] { 3.0 } else { 0.0 };
            }
        }
        let batch = Batch {
            n,
            len: 1,
            kmax,
            input_ids: vec![0; n],
            attention_mask: vec![1; n],
            seq_lens: vec![1; n],
            marker_pos: vec![0; n * kmax],
            marker_count: qs.iter().map(|q| q.options.len()).collect(),
            qtype: qs.iter().map(|q| q.qtype.index() as u32).collect(),
        };
        let out = BackendOutput {
            logits,
            pooled: vec![0.0; n * 2],
        };
        let temps = Temperatures {
            by_type: [1.0; 3],
            by_options: Map::new(),
            rejected: vec![],
        };
        decode_answers(&out, &batch, &zero_act_head(2), &temps, &qs, 0)
    }

    /// A numeric label goes out as Python prints it, whatever the caller typed: `1E5` is
    /// `100000.0`, `1e16` is `1e+16`, `-0` is `0`, `1.10` is `1.1`, and an int of any size
    /// keeps its digits, so two labels beyond `u64::MAX` that differ in the last digit stay
    /// two answers. With `arbitrary_precision` a serializer writes the Number's text verbatim,
    /// so the text has to be Python's already. Expected strings are `json.dumps` output under
    /// Python 3.14.
    #[test]
    fn choice_label_is_printed_like_python() {
        let want = [
            "100000.0",
            "1e+16",
            "0",
            "1.1",
            "18446744073709551616",
            "18446744073709551617",
            "123456789012345678901234567890",
        ];
        let labels = "[1E5, 1e16, -0, 1.10, 18446744073709551616, 18446744073709551617, 123456789012345678901234567890]";
        let ids: Vec<String> = (0..want.len()).map(|i| format!("q{i}")).collect();
        let text = format!(
            "{{{}}}",
            ids.iter()
                .map(|id| format!(
                    r#""{id}": {{"type": "choice", "instructions": "x", "criteria": {labels}}}"#
                ))
                .collect::<Vec<_>>()
                .join(", ")
        );
        let winners: Vec<usize> = (0..want.len()).collect();
        let answers = decode_with_winners(&text, &winners).unwrap();
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(
                serde_json::to_string(&answers[id]["choice"]).unwrap(),
                want[i],
                "{id}"
            );
            // The probabilities keys are the same text, in label order.
            let keys: Vec<&String> = answers[id]["probabilities"]
                .as_object()
                .unwrap()
                .keys()
                .collect();
            assert_eq!(keys, want, "{id}");
        }
        // Strings and bools are untouched.
        assert_eq!(python_value(&json!("1E5")).unwrap(), json!("1E5"));
        assert_eq!(python_value(&json!(true)).unwrap(), json!(true));
    }

    /// The legend copies the score descriptions from the request, so every number in them,
    /// nested or not, is printed as Python would. Expected text is `json.dumps` under Python
    /// 3.14 with compact separators, as serde_json writes it.
    #[test]
    fn score_legend_is_printed_like_python() {
        let answers = decode_with_winners(
            r#"{"rate": {"type": "score", "instructions": "x",
                "criteria": [1E5, {"desc": 1.10, "n": -0}, [0.0, -0.0, 18446744073709551617], "1E5", true]}}"#,
            &[1],
        )
        .unwrap();
        assert_eq!(
            serde_json::to_string(&answers["rate"]["legend"]).unwrap(),
            r#"{"0":100000.0,"1":{"desc":1.1,"n":0},"2":[0.0,-0.0,18446744073709551617],"3":"1E5","4":true}"#
        );
    }

    /// A float label or legend entry with no finite value (`1e400`) passes validation and
    /// reaches the model as the text `inf`, as upstream, and then fails the response the way
    /// upstream's serializer does, with its message: the winner first (`choice` comes before
    /// `probabilities`), then the first such label in option order, then the legend in order.
    #[test]
    fn non_finite_request_numbers_fail_like_upstream_serialization() {
        let msg = |x: &str| format!("Out of range float values are not JSON compliant: {x}");
        let choice = |crit: &str| {
            format!(r#"{{"q": {{"type": "choice", "instructions": "x", "criteria": {crit}}}}}"#)
        };
        let err = |text: &str, winner: usize| {
            decode_with_winners(text, &[winner])
                .unwrap_err()
                .to_string()
        };
        // The winner is finite; the key of another label is not.
        assert_eq!(err(&choice("[1e400, \"b\"]"), 1), msg("inf"));
        assert_eq!(err(&choice("[\"a\", -1e400]"), 0), msg("-inf"));
        // The winner is not finite: it fails before the other label's key.
        assert_eq!(err(&choice("[1e400, -1e400]"), 1), msg("-inf"));
        assert_eq!(err(&choice("[1e400, -1e400]"), 0), msg("inf"));
        assert!(matches!(
            decode_with_winners(&choice("[1e400, \"b\"]"), &[1]),
            Err(Error::Question(_))
        ));
        let score = |crit: &str| {
            format!(r#"{{"q": {{"type": "score", "instructions": "x", "criteria": {crit}}}}}"#)
        };
        assert_eq!(err(&score("[\"low\", 1e400]"), 0), msg("inf"));
        assert_eq!(
            err(&score("[{\"desc\": [1, -1e400]}, 1e400]"), 0),
            msg("-inf")
        );
        // The question before it is decoded; the failing one stops the response.
        let two = r#"{"ok": {"type": "noul", "instructions": "x"},
                      "bad": {"type": "choice", "instructions": "x", "criteria": [1e400]}}"#;
        assert_eq!(
            decode_with_winners(two, &[0, 0]).unwrap_err().to_string(),
            msg("inf")
        );
    }
}
