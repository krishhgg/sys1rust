//! Equivalence of the work-reduction settings (`dense_upto=1024`, `headprune`, `unpad`, the
//! three together, and `fuserope` alone and with the three), of boolean masks (`mask=bool`),
//! and of the round 3 settings (`band`, `nax`), on the real checkpoints (ignored by default;
//! needs them in the HF cache, `source bench/env.sh` first).
//! Every checkpoint found is run, a missing one is skipped with a note (with
//! `SYS1_TEST_ALL_CHECKPOINTS=1`, a missing one fails the test). Pass criteria per
//! question: the same chosen answer (argmax choice, rounded score, noul side) and every
//! reported probability within 1e-3. The exact settings (`fuserope` against the plain path,
//! `band` and `nax` against the plain path and against the round 2 default) must give the
//! same answer JSON for every state and raw logits and pooled outputs equal bit for bit
//! (`f32::to_bits`). Every kernel setting must also be listed by `Backend::active_kernels`,
//! so a load that fell back to the MLX ops fails the test instead of passing as equal.
//!
//! The cases are the `bench/workloads/smoke.jsonl` requests plus built edge cases: one question
//! with 2 options and with 1 option, 20 options, a state past `max_len` (truncated), two states
//! of very different lengths in one request (heavy padding), the three answer types in one
//! request (padded marker slots), and a single question (no padding, so `unpad` is skipped).
//! Per checkpoint, states sized for its tokenizer add the lengths around `band=512`: one row of
//! 511, 512 and 513 tokens, and two rows padded to 512 and to 600 (see [`length_cases`]).
//!
//! Run from the repo root with:
//! `cargo test --manifest-path runtime/Cargo.toml -p laya-mlx --release --test settings -- --ignored --nocapture`

mod common;

use common::{assert_ran, bench_root, categorical, checkpoint_dir, prob_diff, read_jsonl, CHECKPOINTS};
use laya_core::{parse_questions, Agent, BackendOptions};
use serde_json::{json, Value};
use std::path::Path;
use std::sync::Mutex;

/// The settings the work reductions are measured against (`results/SPEED.md`). Their answers
/// against the upstream fp32 reference are checked in `tests/reference.rs`.
const BASE: &str = "f16gelu,cache=512,wired=2048";
/// What sys1d's round 2 default adds to `BASE`: `BASE` plus this is the default the round 3
/// settings are checked against, bit for bit.
const DEFAULT_ON: &str = "dense_upto=1024,headprune,unpad,fuserope";
const TOL: f64 = 1e-3;
/// Against the fp32 CPU reference of `bench/reference/<name>/smoke.jsonl`: the f16 GPU
/// tolerance of `tests/parity.rs`. typed-decisions measures 0.0008, multilingual 0.0102 on
/// one answer, with or without the work reductions.
const REFERENCE_TOL: f64 = 0.02;

/// Two agents at a time is the budget; the tests take turns so `cargo test` cannot load six.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

/// One request: the states (each against every question) and the questions object.
#[derive(Clone)]
struct Case {
    name: String,
    states: Vec<Value>,
    questions: Value,
}

fn words(n: usize) -> String {
    let stock = [
        "The", "customer", "reports", "the", "invoice", "total", "changed", "after", "the",
        "plan", "upgrade", "and", "asks", "for", "a", "refund", "of", "the", "difference",
        "before", "the", "next", "billing", "cycle", "starts", "on", "Monday",
    ];
    (0..n).map(|i| stock[i % stock.len()]).collect::<Vec<_>>().join(" ")
}

fn choice(instructions: &str, options: &[&str]) -> Value {
    json!({"type": "choice", "instructions": instructions, "criteria": options})
}

fn cases(smoke: &[Value]) -> Vec<Case> {
    let short_state = json!({"account": {"tier": "standard", "tenure_months": 3}, "thread": [{"role": "customer", "text": "My payment did not go through."}]});
    let long_state = Value::String(words(900));
    let over_max_state = Value::String(words(2600));
    let many: Vec<String> = (0..20).map(|i| format!("option_{i}: outcome number {i}")).collect();
    let many: Vec<&str> = many.iter().map(String::as_str).collect();
    let mixed = json!({
        "next": choice("What should the assistant do next?", &["answer_directly", "escalate", "refund", "ask_more"]),
        "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["none", "low", "medium", "high", "critical"]},
        "angry": {"type": "noul", "instructions": "Is the customer angry?"},
        "refund_ok": {"type": "noul", "instructions": "Is a refund warranted?", "criteria": {"true": "the customer was overcharged", "false": "no charge error"}, "labels": {"false": "deny", "true": "grant"}},
    });
    let mut v = vec![
        Case {
            name: "two_options".into(),
            states: vec![short_state.clone()],
            questions: json!({"q": choice("Is this about billing or delivery?", &["billing", "delivery"])}),
        },
        Case {
            name: "one_option".into(),
            states: vec![short_state.clone()],
            questions: json!({"q": choice("Pick the only option.", &["only"])}),
        },
        Case {
            name: "many_options".into(),
            states: vec![short_state.clone()],
            questions: json!({"q": choice("Which outcome fits?", &many)}),
        },
        Case {
            name: "over_max_len".into(),
            states: vec![over_max_state],
            questions: mixed.clone(),
        },
        Case {
            name: "heavy_padding".into(),
            states: vec![short_state.clone(), long_state, Value::String("Hi.".into())],
            questions: mixed.clone(),
        },
        Case {
            name: "mixed_types".into(),
            states: vec![short_state.clone()],
            questions: mixed,
        },
        Case {
            name: "single_question".into(),
            states: vec![short_state],
            questions: json!({"q": {"type": "noul", "instructions": "Is the account locked?"}}),
        },
    ];
    for row in smoke {
        v.push(Case {
            name: row["id"].as_str().unwrap().to_string(),
            states: vec![row["body"]["state"].clone()],
            questions: row["body"]["questions"].clone(),
        });
    }
    v
}

/// Everything one agent produces for the cases: the answers per state, and the raw backend
/// output (unmasked logits and pooled) for the collated batch.
struct Run {
    answers: Vec<Vec<Value>>,
    logits: Vec<Vec<f32>>,
    pooled: Vec<Vec<f32>>,
}

fn run(agent: &Agent, cases: &[Case]) -> Run {
    let mut r = Run { answers: Vec::new(), logits: Vec::new(), pooled: Vec::new() };
    for c in cases {
        let out = agent.predict_batch(&c.states, &c.questions, None).unwrap_or_else(|e| panic!("{}: {e}", c.name));
        r.answers.push(out.into_iter().map(|mut o| o["answers"].take()).collect());
        let qs = parse_questions(&c.questions).unwrap();
        let mut items = Vec::new();
        for st in &c.states {
            items.extend(agent.encode(st, &qs).unwrap());
        }
        let batch = agent.collate(&items);
        let bo = agent.backend().forward(&batch).unwrap();
        let mut logits = Vec::new();
        for row in 0..batch.n {
            logits.extend_from_slice(&bo.logits[row * batch.kmax..row * batch.kmax + batch.marker_count[row]]);
        }
        r.logits.push(logits);
        r.pooled.push(bo.pooled);
    }
    r
}

fn load(dir: &Path, tuning: &str) -> Agent {
    let opts = BackendOptions { tuning: Some(tuning.into()), ..Default::default() };
    Agent::load(dir, &opts, Box::new(laya_mlx::make_backend)).unwrap()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// The largest magnitude in `a` and the f16 spacing (ulp) at that magnitude, to read an
/// absolute diff against: the hidden state has outlier dimensions past 1,000, where one f16
/// ulp is 1.0, and a one-ulp change there shows up whole wherever the residual and the FFN
/// output nearly cancel.
fn f16_scale(a: &[f32]) -> (f32, f32) {
    let scale = a.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    let ulp = if scale > 0.0 { 2f32.powi(scale.log2().floor() as i32 - 10) } else { 0.0 };
    (scale, ulp)
}

/// The edge cases really are what their names say, for this checkpoint's `max_len`.
fn check_shapes(agent: &Agent, cases: &[Case]) {
    let max_len = agent.cfg.agent.max_len;
    let batch_of = |name: &str| {
        let c = cases.iter().find(|c| c.name == name).unwrap();
        let qs = parse_questions(&c.questions).unwrap();
        let mut items = Vec::new();
        for st in &c.states {
            items.extend(agent.encode(st, &qs).unwrap());
        }
        agent.collate(&items)
    };
    let b = batch_of("over_max_len");
    assert_eq!(b.len, max_len, "over_max_len is padded to max_len");
    assert!(b.seq_lens.iter().all(|&l| l == max_len), "every row truncated to max_len: {:?}", b.seq_lens);
    let b = batch_of("heavy_padding");
    assert_eq!(b.n, 12);
    let (lo, hi) = (b.seq_lens.iter().min().unwrap(), b.seq_lens.iter().max().unwrap());
    assert!(*hi > 4 * *lo, "rows of very different lengths: {:?}", b.seq_lens);
    assert_eq!(b.kmax, 5);
    assert!(b.marker_count.contains(&2) && b.marker_count.contains(&4), "padded marker slots");
    let b = batch_of("single_question");
    assert_eq!((b.n, b.total_tokens()), (1, b.len), "no padding");
    let b = batch_of("many_options");
    assert_eq!(b.kmax, 20);
    assert_eq!(batch_of("one_option").kmax, 1);
}

/// The padded length of the batch that `states` against `questions` makes on this agent.
fn batch_len(agent: &Agent, states: &[Value], questions: &Value) -> usize {
    let qs = parse_questions(questions).unwrap();
    let mut items = Vec::new();
    for st in states {
        items.extend(agent.encode(st, &qs).unwrap());
    }
    agent.collate(&items).len
}

/// A state that makes a batch of exactly `target` tokens with `questions` on this agent's
/// tokenizer: the longest run of [`words`] that fits, then single periods. `None` when
/// `target` is past the checkpoint's `max_len`.
fn state_of_len(agent: &Agent, questions: &Value, target: usize) -> Option<Value> {
    if target > agent.cfg.agent.max_len {
        return None;
    }
    let len_of = |text: &str| batch_len(agent, &[Value::String(text.to_string())], questions);
    // Every word adds at least one token, so `target` words are too many (or truncated to
    // `max_len`, which is then `target`).
    let (mut lo, mut hi) = (0, target);
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if len_of(&words(mid)) <= target {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let mut text = words(lo);
    while len_of(&text) < target {
        text.push_str(" .");
    }
    assert_eq!(len_of(&text), target, "no state of {target} tokens: one more period overshoots");
    Some(Value::String(text))
}

/// Cases at the lengths around `band=512`, sized for this checkpoint's tokenizer: one row of
/// 511, 512 and 513 tokens (the one-row layout of the banded path from 512 up), and two rows
/// padded to 512 and to 600 (the multi-row layout, packed with `unpad`; 600 is not a multiple
/// of the 64-position chunk). A length past the checkpoint's `max_len` is skipped (english
/// has 512).
fn length_cases(agent: &Agent) -> Vec<Case> {
    let one = json!({"q": {"type": "noul", "instructions": "Is the account locked?"}});
    let two = json!({"q": choice("Is this about billing or delivery?", &["billing", "delivery"])});
    let mut v = Vec::new();
    for target in [511, 512, 513] {
        if let Some(st) = state_of_len(agent, &one, target) {
            v.push(Case { name: format!("len_{target}"), states: vec![st], questions: one.clone() });
        }
    }
    for (target, other) in [(512, 200), (600, 333)] {
        if let (Some(a), Some(b)) = (state_of_len(agent, &two, target), state_of_len(agent, &two, other)) {
            v.push(Case { name: format!("rows_{target}_{other}"), states: vec![a, b], questions: two.clone() });
        }
    }
    for c in &v {
        let want: usize = c.name.split('_').nth(1).unwrap().parse().unwrap();
        assert_eq!(batch_len(agent, &c.states, &c.questions), want, "{}", c.name);
    }
    v
}

/// The kernel names `Backend::active_kernels` reports for the kernel settings of `spec`.
fn kernel_settings(spec: &str) -> Vec<&'static str> {
    spec.split(',')
        .filter_map(|s| match s {
            "fuserope" | "fuserope=1" => Some("fuserope"),
            s if s.starts_with("band=") && s != "band=0" => Some("band"),
            s if s.starts_with("nax=") && s != "nax=0" => Some("nax"),
            _ => None,
        })
        .collect()
}

/// Load the base agent and the agent with `extra` on top, run both over the cases and compare.
/// Returns the max probability diff seen, after asserting every criterion.
/// The base settings are `BASE` plus `on` (`""`: `BASE` alone). With `exact`, every state's
/// answers must serialise to the same JSON and the raw logits and pooled outputs must be equal
/// bit for bit; otherwise the same choices and probabilities within `TOL`.
fn compare(name: &str, dir: &Path, cases: &[Case], on: &str, extra: &str, exact: bool) -> f64 {
    let base = if on.is_empty() { BASE.to_string() } else { format!("{BASE},{on}") };
    let base_agent = load(dir, &base);
    let tuned_agent = load(dir, &format!("{base},{extra}"));
    let label = if on.is_empty() { extra.to_string() } else { format!("{on} + {extra}") };
    // A kernel setting that fell back at load runs the same MLX ops as the base agent, and a
    // comparison would pass without touching the kernel. Refuse that run.
    if on.is_empty() {
        assert!(base_agent.backend().active_kernels().is_empty(), "{name}: the base settings `{BASE}` have a kernel active");
    }
    for (agent, which, spec) in [(&base_agent, "base", on), (&tuned_agent, "tuned", &format!("{on},{extra}")[..])] {
        for setting in kernel_settings(spec) {
            assert!(
                agent.backend().active_kernels().contains(&setting),
                "{name}/{label}: `{setting}` is not active on the {which} agent: it fell back to the MLX ops at load (laya-mlx printed why on stderr), so this run would not test it"
            );
        }
    }
    let base = run(&base_agent, cases);
    let tuned = run(&tuned_agent, cases);
    let (mut worst, mut worst_logit, mut worst_pooled, mut answers, mut exact_states) = (0.0f64, 0.0f32, 0.0f32, 0usize, 0usize);
    let (mut pooled_scale, mut pooled_ulp) = (0.0f32, 0.0f32);
    for (i, c) in cases.iter().enumerate() {
        worst_logit = worst_logit.max(max_abs(&base.logits[i], &tuned.logits[i]));
        worst_pooled = worst_pooled.max(max_abs(&base.pooled[i], &tuned.pooled[i]));
        let (scale, ulp) = f16_scale(&base.pooled[i]);
        if scale > pooled_scale {
            (pooled_scale, pooled_ulp) = (scale, ulp);
        }
        for (a, b) in base.answers[i].iter().zip(&tuned.answers[i]) {
            let same_json = serde_json::to_string(a).unwrap() == serde_json::to_string(b).unwrap();
            if same_json {
                exact_states += 1;
            }
            assert!(same_json || !exact, "{name}/{}/{label}: answers differ: {a} vs {b}", c.name);
            let (qa, qb) = (a.as_object().unwrap(), b.as_object().unwrap());
            assert_eq!(qa.len(), qb.len(), "{name}/{}/{label}", c.name);
            for (qid, ans) in qa {
                let other = &qb[qid];
                assert_eq!(categorical(ans), categorical(other), "{name}/{}/{qid}/{label}: {ans} vs {other}", c.name);
                // Same probability keys, all finite, or `prob_diff` says which one is not.
                let d = prob_diff(ans, other).unwrap_or_else(|e| panic!("{name}/{}/{qid}/{label}: {e}: {ans} vs {other}", c.name));
                assert!(d <= TOL, "{name}/{}/{qid}/{label}: probability diff {d}: {ans} vs {other}", c.name);
                // `answer_confidence` comes from the logits, `act_probability` from pooled.
                for key in [&["answer_confidence"][..], &["action", "act_probability"]] {
                    let (mut x, mut y) = (ans, other);
                    for k in key {
                        (x, y) = (&x[k], &y[k]);
                    }
                    let (Some(x), Some(y)) = (x.as_f64().filter(|v| v.is_finite()), y.as_f64().filter(|v| v.is_finite())) else {
                        panic!("{name}/{}/{qid}/{label}: {} is not a finite number on both sides: {x} vs {y}", c.name, key.join("."));
                    };
                    let dc = (x - y).abs();
                    assert!(dc <= TOL, "{name}/{}/{qid}/{label}: {} diff {dc}", c.name, key.join("."));
                    worst = worst.max(dc);
                }
                worst = worst.max(d);
                answers += 1;
            }
        }
    }
    let states: usize = base.answers.iter().map(Vec::len).sum();
    if exact {
        assert_eq!(exact_states, states, "{name}/{label}: not every state is byte-identical");
        assert_eq!((worst_logit, worst_pooled), (0.0, 0.0), "{name}/{label}: raw logits or pooled outputs differ");
        // Bit for bit, so a sign of zero or a NaN payload counts too.
        for (i, c) in cases.iter().enumerate() {
            for (what, a, b) in [("logits", &base.logits[i], &tuned.logits[i]), ("pooled", &base.pooled[i], &tuned.pooled[i])] {
                let same = a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x.to_bits() == y.to_bits());
                assert!(same, "{name}/{}/{label}: raw {what} differ in their bits", c.name);
            }
        }
    }
    eprintln!(
        "{name:<16} {label:<40} {answers} answers over {} cases: max prob diff {worst:.2e}, {exact_states}/{states} states byte-identical, raw logits {worst_logit:.2e}, pooled {worst_pooled:.2e} (one f16 ulp at its max |x| {pooled_scale:.0} is {pooled_ulp})",
        cases.len()
    );
    worst
}

/// Run `extra` against the base settings on every checkpoint in the cache; `exact` as in
/// [`compare`].
fn every_checkpoint(extra: &str, exact: bool) {
    every_checkpoint_on("", extra, exact);
}

/// [`every_checkpoint`] against `BASE` plus `on` (`""`: `BASE` alone), with the
/// [`length_cases`] of each checkpoint added to the shared cases.
fn every_checkpoint_on(on: &str, extra: &str, exact: bool) {
    let _turn = ONE_AT_A_TIME.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let smoke = read_jsonl(&bench_root().join("workloads/smoke.jsonl"));
    let shared = cases(&smoke);
    let mut ran = 0;
    for (name, repo) in CHECKPOINTS {
        let Some(dir) = checkpoint_dir(name, repo) else {
            continue;
        };
        let plain = load(&dir, BASE);
        check_shapes(&plain, &shared);
        let mut cases = shared.clone();
        cases.extend(length_cases(&plain));
        drop(plain);
        compare(name, &dir, &cases, on, extra, exact);
        ran += 1;
    }
    assert_ran(ran);
}

#[test]
#[ignore]
fn dense_upto_matches_plain() {
    every_checkpoint("dense_upto=1024", false);
}

#[test]
#[ignore]
fn headprune_matches_plain() {
    every_checkpoint("headprune", false);
}

#[test]
#[ignore]
fn unpad_matches_plain() {
    every_checkpoint("unpad", false);
}

#[test]
#[ignore]
fn all_three_match_plain() {
    every_checkpoint("dense_upto=1024,headprune,unpad", false);
}

/// The `fuserope` kernel on the padded layout (no `unpad`): the kernel is checked bit for bit
/// against the MLX ops in `split_rope.rs`; this checks the answers end to end, exactly. Every
/// state's answers serialise to the same JSON as the plain path's, and the raw logits and
/// pooled outputs are equal, so `fuserope` cannot change a chosen answer where the plain path
/// does not. The plain path's agreement with the upstream fp32 reference is checked by the
/// bench harness (`tests/reference.rs` for the smoke set): on the typed-decisions correctness
/// workload with the round 2 default (`fuserope` on), 1,498 of 1,500 answers agree (99.9%),
/// the same two near-tie flips as with the old default; see `results/SPEED.md`, round 2.
#[test]
#[ignore]
fn fuserope_matches_plain() {
    every_checkpoint("fuserope", true);
}

/// The `fuserope` kernel on the packed layout, the round 2 default of sys1d: with `unpad` the
/// kernel also does the expand through the packing's index.
#[test]
#[ignore]
fn all_four_match_plain() {
    every_checkpoint(DEFAULT_ON, false);
}

/// The banded local attention (`band`, round 3) against the plain path, exactly: from the
/// shortest length up (`band=1`, every case takes it) and from 512 up (`band=512`, the
/// shipped threshold: `len_511` and the smoke cases take the dense path, `len_512` and up the
/// banded one). The banded layout gives every query the keys the dense window mask leaves
/// it, in the same 32-key blocks, and the blocks it adds are fully masked.
#[test]
#[ignore]
fn band_matches_plain() {
    every_checkpoint("fuserope,band=1", true);
    every_checkpoint("fuserope,band=512", true);
}

/// `band` on top of the round 2 default, exactly: the packed layout (`unpad`), where the
/// banded output is compacted through its own index, and the chunked path above
/// `dense_upto=1024`, which `band` replaces.
#[test]
#[ignore]
fn band_with_the_default_matches_the_default() {
    every_checkpoint_on(DEFAULT_ON, "band=1", true);
    every_checkpoint_on(DEFAULT_ON, "band=512", true);
}

/// `nax=all`: the encoder's 4 linears and the head's 2 residual products on MLX's own NAX gemm
/// loop with other tiles and launch order, and wo2's split K in one launch. Exact: the same
/// sums in the same order. On the padded layout (the base settings) and on the packed one (the
/// round 2 default, `unpad`). On the padded layout the cases reach both sides of the tile
/// table's 1,024-row limit (two rows padded to 600 make 1,200). Needs a machine where MLX uses NAX (macOS 26.2 or later, GPU
/// generation 17 or later): elsewhere the load falls back and the kernel check fails the test.
#[test]
#[ignore]
fn nax_matches_plain() {
    every_checkpoint("nax=all", true);
    every_checkpoint_on(DEFAULT_ON, "nax=all", true);
}

/// A kernel setting the load turns off shows in `Backend::active_kernels`, which is how
/// `compare` refuses a silent fallback: `nax` with weights copied to `[in, out]` (`wcopy=t`)
/// and `band` with boolean masks fall back at load, the other kernels stay on.
#[test]
#[ignore]
fn fallback_shows_in_active_kernels() {
    let _turn = ONE_AT_A_TIME.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut ran = 0;
    for (name, repo) in CHECKPOINTS {
        let Some(dir) = checkpoint_dir(name, repo) else {
            continue;
        };
        for (extra, want) in [
            ("fuserope,band=512,nax=all", &["fuserope", "band", "nax"][..]),
            ("fuserope,band=512,nax=all,wcopy=t", &["fuserope", "band"][..]),
            ("fuserope,band=512,nax=all,mask=bool", &["fuserope", "nax"][..]),
        ] {
            let agent = load(&dir, &format!("{BASE},{extra}"));
            assert_eq!(agent.backend().active_kernels(), want, "{name}: {extra}");
            eprintln!("{name:<16} {extra:<40} active kernels {want:?}");
        }
        ran += 1;
    }
    assert_ran(ran);
}

/// The round 3 exact stack, `band=512` and `nax=all`, on the round 2 default, and with the
/// band from the shortest length up (`band=1`) so every case takes it.
#[test]
#[ignore]
fn stack_matches_plain() {
    every_checkpoint_on(DEFAULT_ON, "band=512,nax=all", true);
    every_checkpoint_on(DEFAULT_ON, "band=1,nax=all", true);
}

/// Boolean masks on every path: `over_max_len` and `heavy_padding` pad past `4 * window`, so
/// with the base settings they take the chunked local attention, whose boolean mask must give
/// the additive path's answers, padded query rows (all false would be NaN) included; the short
/// cases take the dense boolean mask.
#[test]
#[ignore]
fn bool_masks_match_additive() {
    every_checkpoint("mask=bool", false);
}

/// The smoke set against the fp32 CPU reference, per checkpoint, with the three work
/// reductions on top of the base settings: probabilities within `REFERENCE_TOL` and the same
/// decision on at least 99% of the answers. `tests/reference.rs` runs the same check with the
/// base settings; its numbers tell a precision regression here from the f16 gap.
#[test]
#[ignore]
fn smoke_matches_reference_with_all_three() {
    let _turn = ONE_AT_A_TIME.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let bench = bench_root();
    let smoke = read_jsonl(&bench.join("workloads/smoke.jsonl"));
    let extra = "dense_upto=1024,headprune,unpad";
    let mut ran = 0;
    for (name, repo) in CHECKPOINTS {
        let Some(dir) = checkpoint_dir(name, repo) else {
            continue;
        };
        let reference = read_jsonl(&bench.join(format!("reference/{name}/smoke.jsonl")));
        let agent = load(&dir, &format!("{BASE},{extra}"));
        let (mut worst, mut worst_at, mut agree, mut total) = (0.0f64, String::new(), 0usize, 0usize);
        for row in &smoke {
            let id = row["id"].as_str().unwrap();
            let want = &reference.iter().find(|r| r["id"] == *id).unwrap_or_else(|| panic!("{id} not in reference"))["answers"];
            let got = agent.predict(&row["body"]["state"], &row["body"]["questions"]).unwrap();
            let got = got["answers"].as_object().unwrap();
            assert_eq!(got.len(), want.as_object().unwrap().len(), "{name}/{id}: answered questions differ from the reference");
            for (qid, ans) in got {
                let d = prob_diff(ans, &want[qid]).unwrap_or_else(|e| panic!("{name}/{id}/{qid}: {e}"));
                assert!(d <= REFERENCE_TOL, "{name}/{id}/{qid}: probability diff {d} vs reference");
                if d > worst {
                    (worst, worst_at) = (d, format!("{id}/{qid}"));
                }
                total += 1;
                agree += usize::from(categorical(ans) == categorical(&want[qid]));
            }
        }
        let pct = 100.0 * agree as f64 / total as f64;
        eprintln!("{name:<16} reference, all three: {total} answers, max prob diff {worst:.4} at {worst_at}, agreement {agree}/{total} ({pct:.1}%)");
        assert!(pct >= 99.0, "{name}/{extra}: agreement {pct:.1}% < 99%");
        ran += 1;
    }
    assert_ran(ran);
}
