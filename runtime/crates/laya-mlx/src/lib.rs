//! MLX (Apple silicon) backend for the Laya decision model.
//!
//! Implements [`laya_core::Backend`] with [`mlx_rs`]: the ModernBERT encoder, the two
//! `nn.TransformerEncoderLayer` decision-head layers and the scorer, exactly as described in
//! `docs/MODEL.md`. Weights stay resident as MLX arrays (f16 by default, f32 with
//! [`BackendOptions::f32`]); the whole forward pass is built lazily and evaluated once.
//!
//! Changed in sys1rust from laya-r-mlx 914c9a7: `Knobs` (settings from
//! `BackendOptions::tuning` or `SYS1_MLX`), the `f16gelu` fix for the f16 -> f32 promotion in
//! GELU, length buckets with warm-up at load, MLX cache and wired limits, optional boolean
//! masks, and diagnostic timing. Work-reduction knobs, off unless asked for (measured in
//! `results/SPEED.md`): `dense_upto` (the dense/chunked local-attention switch), `headprune`
//! (last head layer only on the rows the scorer reads) and `unpad` (hidden states packed to
//! the real tokens outside attention). `fuserope` (round 2) runs the encoder's qkv split,
//! head reshape, RoPE and unpad expand as one custom Metal kernel (`split_rope.rs`). `band`
//! (round 3) runs the local layers' attention by chunks in a layout that kernel writes.
//! Loading settings (round 3): `directload` (f16 tensors copied into MLX from the checkpoint
//! as they are) and `sharehead` (the pruned head layer's projections as views of the full one).
//! [`MLX_ENV_DEFAULTS`] lists the MLX environment variables sys1d sets for itself.
//! Experiments that gained nothing (`split`, `rope1`, `splitk`) were removed after commit
//! 0495800; that commit has their code.

use laya_core::weights::{to_f16, to_f32};
use laya_core::{
    Backend, BackendOptions, BackendOutput, Batch, Device, Error, ModelConfig, Result, Weights,
};
use mlx_rs::error::Exception;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::transforms::compile::compile;
use mlx_rs::{fast, nn, ops, transforms, Array, Dtype, Stream};
use safetensors::tensor::TensorView;
use safetensors::SafeTensors;
use std::collections::HashSet;
use std::sync::Mutex;

pub mod metal_kernels;
mod nax_gemm;
mod split_rope;
use nax_gemm::NaxGemm;
use split_rope::SplitRope;

/// Finite "minus infinity" for additive attention masks (safe in f16, no NaN rows).
const MASK_NEG: f32 = -1e4;
/// Logit value reported for masked marker slots.
const LOGIT_MASKED: f32 = -1e4;
/// Distinct input shapes the per-shape GeGLU (`geglu=compiled`) compiles before new shapes take
/// the shapeless trace instead. MLX keeps every per-shape trace for the life of the process, so
/// this count bounds that memory; 64 covers a few length buckets times the row counts a server
/// sees, and the shapeless trace serves everything past it.
const GEGLU_MAX_SHAPES: usize = 64;

/// Backend settings, read once at load from `BackendOptions::tuning` or else `SYS1_MLX` (comma
/// list, e.g. `f16gelu,mask=bool`). Defaults reproduce laya-r-mlx 914c9a7.
///
/// Parsing is strict: an unknown setting or a value the setting cannot take is an
/// [`Error::Config`] naming it, so a run cannot be labeled with a setting that was never
/// applied. Flags take no value (`f16gelu`) or `=0`/`=1`; everything else needs `key=value`.
#[derive(Debug, Clone)]
struct Knobs {
    /// `shapeless` (default): split outside, GELU * gate compiled once for all shapes.
    /// `compiled`: split + GELU + gate compiled per input shape, one trace per distinct
    /// `(rows, len)`, or per packed token count with `unpad`, that MLX never frees, so at most
    /// [`GEGLU_MAX_SHAPES`] shapes are compiled this way and every new shape after that runs
    /// the shapeless trace; for experiments only. `plain`: no compile.
    geglu: String,
    /// Fuse residual adds into the gemm with `addmm` (default) or use matmul + add.
    addmm: bool,
    /// Boolean attention masks (as Python laya-mlx) instead of additive f16 masks, on every
    /// path of the forward: key padding, dense and chunked local attention, and the head.
    bool_mask: bool,
    /// Chunked sliding-window attention for long inputs (default) or dense masks always.
    windowed: bool,
    /// `mlx::clear_cache` after every forward.
    clear: bool,
    /// Print graph-build and eval times of every forward to stderr.
    trace: bool,
    /// Evaluate after every encoder layer and print per-layer times (slows the forward).
    layers: bool,
    /// Evaluate after every op of the encoder layers and print summed times per op.
    ops: bool,
    /// Linear weights: `view` keeps the host-loaded `[out, in]` array behind a transposed view
    /// (default); `gpu` copies it on the GPU first; `t` stores a GPU-written contiguous `[in, out]`.
    wcopy: String,
    /// Keep GELU in the compute dtype (fixes the upstream f16 -> f32 promotion). Off by
    /// default like every knob here, so that `BackendOptions::default()` reproduces upstream's
    /// numerics and the bench's `mlx-fp16` control variant measures the promotion. Everything
    /// that serves answers (`sys1d`, the other bench variants) sets `f16gelu`.
    f16gelu: bool,
    /// Pad the sequence length up to the first of these that fits (`buckets=128:256:512`).
    buckets: Vec<usize>,
    /// Otherwise (or past the last bucket) pad the length to a multiple of this.
    pad: usize,
    /// Row counts to run one forward for at every bucket length at load (`warm=1:4:10`).
    warm: Vec<usize>,
    /// MLX buffer-cache limit in MiB. MLX's default lets freed buffers pile up to about the
    /// size of RAM when request shapes vary, which pushes the machine into swap.
    cache_mb: Option<usize>,
    /// MLX wired-memory limit in MiB (keeps the weights resident).
    wired_mb: Option<usize>,
    /// Local attention runs dense (full `len x len` masks) while the padded length is at most
    /// this, chunked above it (`dense_upto=512`). `None` is the upstream `4 * window`.
    /// `windowed=0` still means never chunk.
    dense_upto: Option<usize>,
    /// Last head layer: queries, out_proj, norm2 and the FFN only for the rows the scorer
    /// reads (position 0 and every marker slot); keys and values still from every token.
    /// The other positions of that layer are never computed, so its output is a
    /// [`ScorerRows`] and nothing else: a consumer that needs per-token head output has to
    /// run with `headprune=0`.
    headprune: bool,
    /// Keep hidden states packed as `[T, d]` (real tokens only) through embeddings, norms,
    /// linears and GeGLU; expand to `[n, len, ...]` only around attention.
    unpad: bool,
    /// Encoder qkv split, head reshape, RoPE and the `unpad` expand in one custom Metal kernel
    /// ([`SplitRope`]) instead of MLX's split, reshape, rope and gather ops; exact, see
    /// `results/SPEED.md` round 2. The kernel is compiled and checked against the MLX ops at
    /// load; if that fails (no GPU, a head dim that is not a multiple of 8, a kernel that does
    /// not compile or does not match the MLX ops on this machine) the load prints one line to
    /// stderr and the forward runs the MLX ops instead.
    fuserope: bool,
    /// Banded local attention from this padded length up (`band=512`; 0, the default, is off).
    /// The local layers attend by chunks of `window` queries against the `3 * window` keys
    /// their window can reach, in a layout the `fuserope` kernel writes directly (see
    /// [`SplitRope::apply_band`]), instead of the dense `len x len` attention with a window
    /// mask, or the gathered chunks above `dense_upto`. It takes precedence over both, and over
    /// `windowed=0`. Exact: every query sees the same keys in the same key blocks of MLX's
    /// attention kernel (32 keys at head dim 64 in MLX 0.32.2; the windows start at multiples
    /// of 64 positions), and the blocks the layout adds are fully masked and add exact zeros
    /// (checked bit for bit in `tests/settings.rs`). Needs `fuserope` and additive masks;
    /// without them the load prints why and the local layers run as if `band` were 0.
    band: usize,
    /// The encoder's 4 linears (wqkv, wo, wi, wo2) and the decision head's 2 residual products
    /// (out_proj, linear2) on MLX's own NAX gemm loop, launched with smaller tiles, row tiles
    /// first, and wo2's split K in one launch ([`NaxGemm`], `nax=all`; `nax=0`, the default,
    /// is off). The residual products take the kernel with `addmm` (the default) only. Exact: each output element is the same sum in the same order as MLX's gemm. At
    /// load the kernels run only where MLX itself uses NAX (macOS 26.2 or later and GPU
    /// architecture generation 17 or later, which the M5 is), with f16 weights on the GPU as
    /// transposed views (any `wcopy` but `t`), and only after each one matches MLX's gemm bit
    /// for bit at every weight shape of the model, a check that runs on every load. Otherwise
    /// the load prints why and MLX's gemms run. The load prints the check's gemm count and time.
    nax: bool,
    /// Copy each F16 tensor of the checkpoint into MLX as it is, instead of through a converted
    /// `Vec` first (see [`Loader::direct`]). Exact: the same bits; a tensor of another dtype,
    /// or an f32 model, takes the conversion path.
    directload: bool,
    /// With `headprune`, take the pruned last head layer's q and k|v projections as views of
    /// its full projection instead of loading them again. With the default `wcopy=view` each
    /// copy keeps a whole checkpoint tensor alive, so this frees 2 copies of that weight and
    /// bias: 12.0 MiB of MLX memory on the 1,024-wide checkpoints and 6.8 MiB on multilingual
    /// (research round 2). Exact: the same values. With `wcopy=view` or `gpu` the views have
    /// the copies' strides; with `wcopy=t` they keep the full projection's row stride, 3 times
    /// a copy's. All three are checked bit for bit in `tests/settings.rs`.
    sharehead: bool,
}

impl Knobs {
    /// The settings of `tuning`, or of `SYS1_MLX` when `tuning` is `None`.
    fn from_spec(tuning: Option<&str>) -> Result<Self> {
        let spec = match tuning {
            Some(t) => t.to_string(),
            None => std::env::var("SYS1_MLX").unwrap_or_default(),
        };
        Self::parse(&spec)
    }

    /// Parse a comma-separated spec. The last mention of a setting wins.
    fn parse(spec: &str) -> Result<Self> {
        let mut k = Knobs {
            geglu: "shapeless".into(),
            addmm: true,
            bool_mask: false,
            windowed: true,
            clear: false,
            trace: false,
            layers: false,
            ops: false,
            wcopy: "view".into(),
            f16gelu: false,
            buckets: Vec::new(),
            pad: 1,
            warm: Vec::new(),
            cache_mb: None,
            wired_mb: None,
            dense_upto: None,
            headprune: false,
            unpad: false,
            fuserope: false,
            band: 0,
            nax: false,
            directload: false,
            sharehead: false,
        };
        for kv in spec.split(',').filter(|s| !s.is_empty()) {
            let (key, val) = match kv.split_once('=') {
                Some((key, val)) => (key, Some(val)),
                None => (kv, None),
            };
            let bad = |what: String| Error::Config(format!("mlx settings: `{kv}`: {what}"));
            // A flag is on when bare or `=1`, off when `=0`; nothing else.
            let flag = || match val {
                None | Some("1") => Ok(true),
                Some("0") => Ok(false),
                Some(v) => Err(bad(format!("`{v}` is not 0 or 1"))),
            };
            let value = || val.ok_or_else(|| bad("needs a value (`key=value`)".into()));
            let one_of = |allowed: &[&str]| -> Result<String> {
                let v = value()?;
                if allowed.contains(&v) {
                    Ok(v.to_string())
                } else {
                    Err(bad(format!("`{v}` is not one of {}", allowed.join(", "))))
                }
            };
            let number = |v: &str| -> Result<usize> {
                v.parse().map_err(|_| bad(format!("`{v}` is not a whole number")))
            };
            let list = || -> Result<Vec<usize>> { value()?.split(':').map(number).collect() };
            match key {
                "geglu" => k.geglu = one_of(&["shapeless", "compiled", "plain"])?,
                "addmm" => k.addmm = flag()?,
                "mask" => k.bool_mask = one_of(&["bool", "additive"])? == "bool",
                "windowed" => k.windowed = flag()?,
                "clear" => k.clear = flag()?,
                "trace" => k.trace = flag()?,
                "layers" => k.layers = flag()?,
                "ops" => k.ops = flag()?,
                "wcopy" => k.wcopy = one_of(&["view", "gpu", "t"])?,
                "f16gelu" => k.f16gelu = flag()?,
                "buckets" => {
                    k.buckets = list()?;
                    k.buckets.sort_unstable();
                }
                "pad" => {
                    k.pad = number(value()?)?;
                    if k.pad == 0 {
                        return Err(bad("pad must be at least 1".into()));
                    }
                }
                "warm" => k.warm = list()?,
                "cache" => k.cache_mb = Some(number(value()?)?),
                "wired" => k.wired_mb = Some(number(value()?)?),
                "dense_upto" => k.dense_upto = Some(number(value()?)?),
                "headprune" => k.headprune = flag()?,
                "unpad" => k.unpad = flag()?,
                "fuserope" => k.fuserope = flag()?,
                "band" => k.band = number(value()?)?,
                "nax" => k.nax = one_of(&["0", "all"])? == "all",
                "directload" => k.directload = flag()?,
                "sharehead" => k.sharehead = flag()?,
                // Read by `laya_core::Agent::load` ([`BackendOptions::parallel_load`]), which loads
                // the tokenizer on a second thread; checked here so a bad value fails here too.
                "parallel_load" => {
                    flag()?;
                }
                _ => return Err(Error::Config(format!("mlx settings: unknown setting `{key}` in `{spec}`"))),
            }
        }
        Ok(k)
    }
}

/// Check a settings spec (`BackendOptions::tuning`, `SYS1_MLX`) without loading a model: `Err`
/// names the first unknown setting or bad value, as loading with it would. Callers that label
/// a run with its settings (sys1-bench, sys1-probe) call this at startup.
pub fn check_settings(spec: &str) -> Result<()> {
    Knobs::parse(spec).map(|_| ())
}

/// MLX environment variables that sys1d (and sys1-bench's `mlx-fp16-lean` variant) set for
/// themselves when the user has not set them. These are process-wide MLX limits, not backend
/// settings, so they live outside `Knobs`.
///
/// `MLX_MAX_MB_PER_BUFFER`: MLX commits a Metal command buffer once the inputs of the ops in it
/// pass this many Mi elements. The name says MB, but MLX 0.32.2 adds `array::data_size()`, an
/// element count, once per input buffer. MLX's default is 40 on this M5 (50 on Max and Ultra
/// GPUs). With 10 on top of `band=512,nax=all`, s512_q1 took 0.931 of the time and s512_q10
/// 0.981, with the same answers (`results/SPEED.md`, round 3). MLX reads the variable once,
/// when it creates the Metal device at the first GPU operation, so it must be set before that.
pub const MLX_ENV_DEFAULTS: [(&str, &str); 1] = [("MLX_MAX_MB_PER_BUFFER", "10")];

/// The entries of [`MLX_ENV_DEFAULTS`] whose variable `get` reports as unset. A value the user
/// set, even an empty one, is kept.
pub fn mlx_env_unset(get: impl Fn(&str) -> Option<std::ffi::OsString>) -> Vec<(&'static str, &'static str)> {
    MLX_ENV_DEFAULTS.into_iter().filter(|(key, _)| get(key).is_none()).collect()
}

/// Set the [`MLX_ENV_DEFAULTS`] the environment does not have yet and return them.
///
/// Call it first thing in `main`, before any MLX call (MLX reads the variables once) and before
/// the process starts a thread: setting a variable while another thread reads the environment
/// is a data race on macOS.
pub fn set_mlx_env_defaults() -> Vec<(&'static str, &'static str)> {
    let unset = mlx_env_unset(|key| std::env::var_os(key));
    for (key, value) in &unset {
        std::env::set_var(key, value);
    }
    unset
}

/// Convert an MLX exception into a `laya_core::Error::Backend`.
pub(crate) trait Lx<T> {
    fn lx(self) -> Result<T>;
}
impl<T> Lx<T> for std::result::Result<T, mlx_rs::error::Exception> {
    fn lx(self) -> Result<T> {
        self.map_err(|e| Error::Backend(e.to_string()))
    }
}

/// Whether two float arrays have the same dtype, shape and bits. The load-time kernel checks
/// use it: `==` would let a -0 pass for a 0 and fail a NaN against itself.
pub(crate) fn same_bits(a: &Array, b: &Array) -> Result<bool> {
    if a.dtype() != b.dtype() {
        return Ok(false);
    }
    let bits = |x: &Array| x.view_dtype(if x.dtype() == Dtype::Float32 { Dtype::Uint32 } else { Dtype::Uint16 });
    bits(a).lx()?.eq_exact(bits(b).lx()?).lx()
}

/// The kernel-fallback rule for `fuserope` and `band`: both are decided here, at load, never
/// in a request. Returns the kernel and the padded length from which the local layers take the
/// banded path. `build` builds the kernel and runs its load check, of the banded launches too
/// when passed `true` ([`SplitRope::new`]). A kernel that cannot be built or whose plain
/// launches do not reproduce the MLX ops turns both off; banded launches that do not turn off
/// `band` only, and the kernel serves the shorter batches. Each fallback prints one line.
fn rope_kernels(
    knobs: &Knobs,
    cpu: bool,
    build: impl FnOnce(bool) -> Result<(SplitRope, Option<Error>)>,
) -> (Option<SplitRope>, Option<usize>) {
    let (split_rope, band_off) = match (knobs.fuserope, cpu) {
        (false, _) => (None, None),
        (true, true) => {
            eprintln!("laya-mlx: fuserope needs the GPU; the CPU backend runs the MLX split and rope ops instead");
            (None, None)
        }
        (true, false) => match build(knobs.band > 0 && !knobs.bool_mask) {
            Ok((k, band_off)) => (Some(k), band_off),
            Err(e) => {
                eprintln!("laya-mlx: fuserope is off for this load, the MLX split and rope ops run instead: {e}");
                (None, None)
            }
        },
    };
    // `band` needs the kernel's banded launches, checked by `build`, and additive masks.
    let band_from = match (knobs.band, &split_rope, band_off) {
        (0, _, _) => None,
        _ if knobs.bool_mask => {
            eprintln!("laya-mlx: band needs additive masks; with mask=bool the local layers run the dense or chunked attention instead");
            None
        }
        (_, None, _) => {
            eprintln!("laya-mlx: band needs the fuserope kernel, which is off for this load; the local layers run the dense or chunked attention instead");
            None
        }
        (_, Some(_), Some(e)) => {
            eprintln!("laya-mlx: band is off for this load, the local layers run the dense or chunked attention instead: {e}");
            None
        }
        (b, Some(_), None) => Some(b),
    };
    (split_rope, band_from)
}

/// A compiled MLX function `&[Array] -> Vec<Array>`.
type CompiledFn = Box<dyn for<'a> FnMut(&'a [Array]) -> std::result::Result<Vec<Array>, Exception>>;

/// `gelu_erf(input) * gate` over `x = [input | gate]` (torch `chunk(2, dim=-1)` order), fused
/// into a single Metal kernel by `mlx_rs::transforms::compile`.
///
/// Unfused, the split + five elementwise passes cost ~0.8 ms per layer at 4x512 tokens;
/// compiled they cost ~0.08 ms. The compiled state is not thread-safe, hence the mutex.
///
/// The default is the shapeless trace: GELU * gate is elementwise, so one trace serves every
/// input shape and the split (which needs concrete shapes) happens outside as two views. This
/// matters with `unpad`, where the input shape is the request's total token count. Paired A/B
/// on the timing workload against the per-shape trace, sys1d's settings: 1.001 typed-decisions,
/// 1.013 and 1.004 (sides swapped) multilingual, identical answers. The per-shape mode
/// (`geglu=compiled`) stays for experiments; MLX keeps one trace per distinct input shape for
/// the life of the process, so the mode compiles at most [`GEGLU_MAX_SHAPES`] shapes and hands
/// every new shape after that to the shapeless trace ([`ShapeBudget`]).
struct GeGlu {
    mode: String,
    keep: bool,
    traces: Mutex<GeGluTraces>,
}

/// The compiled traces behind [`GeGlu`]'s mutex.
struct GeGluTraces {
    /// GELU * gate over the two halves; one trace for every shape.
    shapeless: CompiledFn,
    /// Split + GELU * gate, one trace per input shape (`geglu=compiled` only).
    per_shape: Option<CompiledFn>,
    /// The shapes `per_shape` has compiled.
    shapes: ShapeBudget,
}

/// The distinct shapes a per-shape cache may hold, at most `cap` of them.
struct ShapeBudget {
    cap: usize,
    seen: HashSet<Vec<i32>>,
}

impl ShapeBudget {
    fn new(cap: usize) -> Self {
        Self { cap, seen: HashSet::new() }
    }

    /// Whether `shape` may take the per-shape path: yes when it is already cached, or when the
    /// cache has room (the shape is then counted); no once `cap` shapes are cached.
    fn admit(&mut self, shape: &[i32]) -> bool {
        if self.seen.contains(shape) {
            return true;
        }
        if self.seen.len() >= self.cap {
            return false;
        }
        self.seen.insert(shape.to_vec());
        true
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.seen.len()
    }
}

// SAFETY: the compiled closures own no thread-affine resources; calls are serialised by the
// mutex and MLX's scheduler is thread-safe.
unsafe impl Send for GeGlu {}
unsafe impl Sync for GeGlu {}

impl GeGlu {
    fn new(mode: &str, keep: bool) -> Self {
        Self::with_cap(mode, keep, GEGLU_MAX_SHAPES)
    }

    /// `cap` is the number of shapes `geglu=compiled` compiles per shape (tests pass a small one).
    fn with_cap(mode: &str, keep: bool, cap: usize) -> Self {
        let shapeless: CompiledFn = Box::new(compile(
            move |a: &[Array]| -> Vec<Array> {
                vec![ops::multiply(gelu_erf_as(&a[0], keep).expect("gelu"), &a[1]).expect("geglu gate")]
            },
            true,
        ));
        let per_shape: Option<CompiledFn> = (mode == "compiled").then(|| -> CompiledFn {
            Box::new(compile(
                move |a: &[Array]| -> Vec<Array> {
                    let ig = a[0].split_equal(2, -1).expect("geglu split");
                    vec![ops::multiply(gelu_erf_as(&ig[0], keep).expect("gelu"), &ig[1]).expect("geglu gate")]
                },
                // Not shapeless: `split` needs concrete shapes; MLX caches one trace per input
                // shape and never drops one, which is why this is not the default and why
                // `apply` stops sending new shapes here after `cap` of them.
                false,
            ))
        });
        Self {
            mode: mode.to_string(),
            keep,
            traces: Mutex::new(GeGluTraces {
                shapeless,
                per_shape,
                shapes: ShapeBudget::new(cap),
            }),
        }
    }

    fn apply(&self, x: &Array) -> Result<Array> {
        if self.mode == "plain" {
            let ig = x.split_equal(2, -1).lx()?;
            return ops::multiply(gelu_erf_as(&ig[0], self.keep).lx()?, &ig[1]).lx();
        }
        // A panic inside the compiled closure (mlx-rs re-raises it after MLX returns) would
        // poison the lock; recover the guard so one failed request cannot fail every later one.
        let mut guard = self.traces.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let t = &mut *guard;
        let mut out = match &mut t.per_shape {
            Some(f) if t.shapes.admit(x.shape()) => f(std::slice::from_ref(x)).lx()?,
            _ => {
                let ig = x.split_equal(2, -1).lx()?;
                (t.shapeless)(ig.as_slice()).lx()?
            }
        };
        Ok(out.remove(0))
    }

    /// How many shapes the per-shape trace has compiled (0 unless `geglu=compiled`).
    #[cfg(test)]
    fn per_shape_traces(&self) -> usize {
        self.traces.lock().unwrap_or_else(std::sync::PoisonError::into_inner).shapes.len()
    }
}

/// Exact GELU, `0.5 * x * (1 + erf(x / sqrt 2))` (torch `nn.GELU()` default).
///
/// Upstream (laya-r-mlx 914c9a7) multiplies by f32 scalar arrays, which promotes f16 input to
/// f32: from the first MLP on, the residual stream and every gemm run in f32 with the f16
/// weights cast on each forward. `keep_dtype` casts the scalars to the input dtype instead.
fn gelu_erf_as(x: &Array, keep_dtype: bool) -> std::result::Result<Array, Exception> {
    let c = |v: f32| -> std::result::Result<Array, Exception> {
        let a = Array::from_f32(v);
        if keep_dtype {
            a.as_dtype(x.dtype())
        } else {
            Ok(a)
        }
    };
    let e = ops::erf(ops::multiply(x, c(std::f32::consts::FRAC_1_SQRT_2)?)?)?;
    ops::multiply(ops::multiply(x, c(0.5)?)?, ops::add(&e, c(1.0)?)?)
}

/// The chunk-local band, `[S, 3S]` row-major: query `i` of a chunk may see gathered key slot
/// `j` (global offset `j - S` relative to the chunk start) iff `|S + i - j| <= S`.
fn band_allows(s: usize) -> Vec<bool> {
    let ks = 3 * s;
    let mut vals = vec![false; s * ks];
    for i in 0..s {
        for j in i..=i + 2 * s {
            vals[i * ks + j] = true;
        }
    }
    vals
}

/// Additive chunk-local band mask `[1, 1, 1, S, 3S]` (0 where [`band_allows`], else `MASK_NEG`).
fn band_mask(s: usize, dtype: Dtype) -> Result<Array> {
    let vals: Vec<f32> = band_allows(s).iter().map(|&ok| if ok { 0.0 } else { MASK_NEG }).collect();
    let m = Array::from_slice(&vals, &[1, 1, 1, s as i32, 3 * s as i32])
        .as_dtype(dtype)
        .lx()?;
    m.eval().lx()?;
    Ok(m)
}

/// Boolean chunk-local band mask `[1, 1, 1, S, 3S]` (true where [`band_allows`]), for `mask=bool`.
fn band_mask_bool(s: usize) -> Result<Array> {
    let m = Array::from_slice(&band_allows(s), &[1, 1, 1, s as i32, 3 * s as i32]);
    m.eval().lx()?;
    Ok(m)
}

/// Key validity of the chunked path, `[n, nc, 3S]` row-major: gathered slot `j` of chunk `c`
/// holds position `(c - 1) * S + j`, valid when that lies in `0..len` and is not padding.
fn window_key_valid(attention_mask: &[u32], n: usize, len: usize, s: usize, nc: usize) -> Vec<bool> {
    let ks = 3 * s;
    let mut ok = vec![false; n * nc * ks];
    for b in 0..n {
        for c in 0..nc {
            for j in 0..ks {
                let pos = (c as i64 - 1) * s as i64 + j as i64;
                ok[(b * nc + c) * ks + j] =
                    pos >= 0 && (pos as usize) < len && attention_mask[b * len + pos as usize] == 1;
            }
        }
    }
    ok
}

/// Padded queries of the chunked path, `[n, nc, S]` row-major: query `i` of chunk `c` is
/// position `c * S + i`, padded when it is past `len` (the tail of the last chunk when `len` is
/// not a multiple of `S`) or padding in its row.
fn window_query_padded(attention_mask: &[u32], n: usize, len: usize, s: usize, nc: usize) -> Vec<bool> {
    let mut padded = vec![true; n * nc * s];
    for b in 0..n {
        for pos in 0..len {
            padded[b * nc * s + pos] = attention_mask[b * len + pos] != 1;
        }
    }
    padded
}

/// Attention scale of the decision-head layers, `1 / sqrt(hidden / heads)`: upstream builds
/// them as `nn.TransformerEncoderLayer(d, nhead=max(1, d // 64))`, whose `MultiheadAttention`
/// scales by the square root of its head dim `d // nhead`. That is 64 (a scale of 1/8) for the
/// published 768- and 1,024-wide checkpoints, but not for a `d` that 64 does not divide.
fn head_scale(hidden: usize, heads: usize) -> f32 {
    1.0 / ((hidden / heads) as f32).sqrt()
}

/// Cast to f32 and force a row-contiguous layout so the result can be read from the host.
fn to_f32_contiguous(a: &Array) -> Result<Array> {
    a.as_dtype(Dtype::Float32).lx()?.contiguous().lx()
}

/// Copy an evaluated, row-contiguous f32 array to the host.
fn host_f32(a: &Array) -> Result<Vec<f32>> {
    a.try_as_slice::<f32>()
        .map(|s| s.to_vec())
        .map_err(|e| Error::Backend(format!("host copy: {e}")))
}

/// Build the MLX backend from a checkpoint. See [`laya_core::Backend`].
pub fn make_backend(
    w: &Weights,
    cfg: &ModelConfig,
    opts: &BackendOptions,
) -> Result<Box<dyn Backend>> {
    Ok(Box::new(MlxBackend::new(w, cfg, opts)?))
}

/// Factory for [`laya_core::testing::run_parity`].
pub fn factory() -> laya_core::testing::Factory {
    Box::new(make_backend)
}

/// `y = x @ W^T (+ b)`; `wt` is the transposed weight view `[in, out]`.
struct Linear {
    wt: Array,
    b: Option<Array>,
}

impl Linear {
    fn apply(&self, x: &Array) -> Result<Array> {
        match &self.b {
            Some(b) => ops::addmm(b, x, &self.wt, None, None).lx(),
            None => ops::matmul(x, &self.wt).lx(),
        }
    }

    /// `residual + x @ W^T (+ b)`, with the residual add fused into the gemm epilogue.
    fn apply_add(&self, x: &Array, residual: &Array) -> Result<Array> {
        let y = ops::addmm(residual, x, &self.wt, None, None).lx()?;
        match &self.b {
            Some(b) => ops::add(&y, b).lx(),
            None => Ok(y),
        }
    }
}

/// LayerNorm over the last axis, optionally with bias.
struct Norm {
    w: Array,
    b: Option<Array>,
    eps: f32,
}

impl Norm {
    fn apply(&self, x: &Array) -> Result<Array> {
        fast::layer_norm(x, &self.w, self.b.as_ref(), self.eps).lx()
    }
}

struct EncoderLayer {
    /// `None` for layer 0 (identity).
    attn_norm: Option<Norm>,
    wqkv: Linear,
    wo: Linear,
    mlp_norm: Norm,
    wi: Linear,
    wo2: Linear,
    rope_theta: f32,
    local: bool,
}

struct HeadLayer {
    norm1: Norm,
    in_proj: Linear,
    /// `in_proj` cut into its q rows and its k|v rows, for the pruned last layer (`headprune`).
    split_proj: Option<(Linear, Linear)>,
    out_proj: Linear,
    norm2: Norm,
    linear1: Linear,
    linear2: Linear,
}

/// Reads tensors from the safetensors view into MLX arrays of the compute dtype.
struct Loader<'a> {
    st: SafeTensors<'a>,
    dtype: Dtype,
    wcopy: String,
    /// The `directload` setting.
    directload: bool,
}

impl Loader<'_> {
    /// Whether [`Self::get`] copies `t`'s bytes into MLX as they are (`directload`): an F16
    /// tensor into an f16 model, on a little-endian machine (safetensors stores little-endian),
    /// whose data starts at an even address (MLX reads it through an f16 pointer) and holds
    /// exactly its shape's elements. Every other tensor goes through a converted `Vec` first,
    /// which for an F16 tensor gives the same bits with one more copy.
    fn direct(&self, t: &TensorView<'_>) -> bool {
        let data = t.data();
        self.directload
            && cfg!(target_endian = "little")
            && self.dtype == Dtype::Float16
            && t.dtype() == safetensors::Dtype::F16
            && data.as_ptr().align_offset(std::mem::align_of::<u16>()) == 0
            && data.len() == 2 * t.shape().iter().product::<usize>()
    }

    fn get(&self, name: &str) -> Result<Array> {
        let t = self
            .st
            .tensor(name)
            .map_err(|e| Error::Weights(format!("{name}: {e}")))?;
        let shape: Vec<i32> = t.shape().iter().map(|&d| d as i32).collect();
        if self.direct(&t) {
            // SAFETY: `direct` checked that the data is 2-byte aligned and holds exactly the
            // shape's f16 elements. `mlx_array_new_data` copies them into a new MLX buffer
            // before it returns, as for `Array::from_slice`, so the array never points into
            // the checkpoint's mapping.
            let a = unsafe { Array::from_raw_data(t.data().as_ptr().cast(), &shape, Dtype::Float16) };
            // A failed `mlx_array_new_data` returns an empty handle; dropping it frees nothing.
            if a.as_ptr().ctx.is_null() {
                return Err(Error::Weights(format!("{name}: MLX could not create the array")));
            }
            return Ok(a);
        }
        let arr = match self.dtype {
            Dtype::Float32 => Array::from_slice(&to_f32(&t)?, &shape),
            _ => Array::from_slice(&to_f16(&t)?, &shape),
        };
        Ok(arr)
    }

    fn linear(&self, w: &str, b: Option<&str>) -> Result<Linear> {
        let b = b.map(|n| self.get(n)).transpose()?;
        self.linear_from(self.get(w)?, b)
    }

    /// Output rows `rows` of a linear layer as a layer of their own.
    fn linear_rows(&self, w: &str, b: Option<&str>, rows: std::ops::Range<i32>) -> Result<Linear> {
        let w = self.get(w)?.index((rows.clone(), ..));
        let b = b.map(|n| self.get(n)).transpose()?.map(|b| b.index(rows));
        self.linear_from(w, b)
    }

    fn linear_from(&self, w: Array, b: Option<Array>) -> Result<Linear> {
        let wt = match self.wcopy.as_str() {
            "t" => ops::transpose(&w).lx()?.contiguous().lx()?,
            "gpu" => {
                let one = Array::from_f32(1.0).as_dtype(self.dtype).lx()?;
                ops::transpose(ops::multiply(&w, &one).lx()?).lx()?
            }
            _ => ops::transpose(&w).lx()?,
        };
        Ok(Linear { wt, b })
    }

    fn norm(&self, w: &str, b: Option<&str>, eps: f32) -> Result<Norm> {
        Ok(Norm {
            w: self.get(w)?,
            b: b.map(|n| self.get(n)).transpose()?,
            eps,
        })
    }
}

/// The resident model plus per-shape caches.
pub struct MlxBackend {
    cpu: bool,
    dtype: Dtype,
    hidden: usize,
    n_heads: usize,
    head_dim: usize,
    head_nheads: usize,
    window: usize,
    tok_emb: Array,
    emb_norm: Norm,
    layers: Vec<EncoderLayer>,
    final_norm: Norm,
    type_emb: Array,
    head: Vec<HeadLayer>,
    scorer_norm: Norm,
    scorer1: Linear,
    scorer3: Linear,
    /// Chunk-local band mask `[1, 1, 1, S, 3S]` for the windowed attention path: additive, or
    /// boolean with `mask=bool`.
    band: Array,
    caches: Mutex<Caches>,
    geglu: GeGlu,
    /// The `fuserope` kernel when the setting is on and its plain launches passed their
    /// load-time check.
    split_rope: Option<SplitRope>,
    /// The padded length from which the local layers take the banded path: `band` when it is
    /// set, masks are additive and `split_rope` passed its banded check at load; else `None`.
    band_from: Option<usize>,
    /// The `nax` kernels when the setting is on and they passed their load-time check.
    nax: Option<NaxGemm>,
    knobs: Knobs,
}

/// Constants reused across forwards. Nothing here is keyed by request shape.
///
/// The dense window masks are one array each, for the longest length seen so far; shorter
/// lengths take a view of it (see [`MlxBackend::window_mask`]). One mask per distinct length
/// would pile up: with `dense_upto=1024` and no length buckets, a server seeing every length
/// up to 1,024 would hold about 700 MB of masks that nothing frees. The gather indices of the
/// windowed path are built inside the forward graph instead ([`window_key_idx`]), so they need
/// no cache at all. The compiled GeGLU trace is shapeless (see [`GeGlu`]).
#[derive(Default)]
struct Caches {
    /// Dense sliding-window additive mask `[1, 1, L, L]` for the longest `L` so far.
    window_mask: Option<Array>,
    /// Boolean sliding-window mask `[1, 1, L, L]` for the longest `L` so far (`mask=bool`).
    window_bool: Option<Array>,
}

/// Flat key gather indices `[rows * nc * 3S]` into the `[rows * len, hd]` view of the keys for
/// the windowed path: chunk `c` of row `r` gathers positions `(c - 1) * S + j` for `j < 3S`,
/// clamped into `0..len` (the mask hides the clamped slots), as `r * len + pos`. Two small host
/// tables and one broadcast add on the device, part of the forward graph: no per-shape cache
/// and no early evaluation.
fn window_key_idx(s: usize, rows: usize, len: usize, nc: usize) -> Result<Array> {
    let ks = 3 * s;
    let pos: Vec<u32> = (0..nc)
        .flat_map(|c| (0..ks).map(move |j| (c as i64 - 1) * s as i64 + j as i64))
        .map(|p| p.clamp(0, len as i64 - 1) as u32)
        .collect();
    let pos = Array::from_slice(&pos, &[1, nc as i32, ks as i32]);
    let base: Vec<u32> = (0..rows as u32).map(|r| r * len as u32).collect();
    let base = Array::from_slice(&base, &[rows as i32, 1, 1]);
    ops::add(&base, &pos).lx()?.reshape(&[(rows * nc * ks) as i32]).lx()
}

/// How local (sliding-window) layers attend for the current batch shape. Every mask here is
/// additive, or boolean (true = may attend) with `mask=bool`.
enum LocalAttn {
    /// Full `len x len` attention with a pad + window mask `[n, 1, len, len]`.
    Dense(Array),
    /// Chunked attention: every 64-query chunk attends to the 192 keys that can fall inside
    /// its window (previous, own and next chunk); see [`MlxBackend::windowed_attention`].
    Windowed(Windowed),
    /// Chunked attention in the layout the `fuserope` kernel writes (`band`); see [`Banded`].
    Banded(Banded),
}

/// Constants of the banded local attention for one batch shape: sdpa runs over `n * nc` chunk
/// batches of `H` heads, `S` queries and `3S` keys, straight from [`SplitRope::apply_band`].
struct Banded {
    /// Additive mask `[n * nc, 1, S, 3S]`: at the keys a chunk holds, the sum of the window
    /// mask and the pad mask the dense path uses; at its slots outside the sequence, `MASK_NEG`
    /// plus the window mask. sdpa broadcasts it over the heads.
    mask: Array,
    n_chunks: i32,
    /// With `unpad`, the `[T]` positions `r * nc * S + pos` of the real tokens in the
    /// `[n, nc * S, d]` output: the packing's compaction for this layout.
    pack: Option<Array>,
}

/// Constants of the chunked sliding-window attention for one batch shape.
struct Windowed {
    /// Flat key gather indices `[n * H * n_chunks * 3S]` into `[n * H * len, hd]` rows.
    key_idx: Array,
    /// Pad + band mask `[n * H, n_chunks, S, 3S]`.
    mask: Array,
    n_chunks: i32,
}

/// Masks shared by every encoder layer of one forward pass.
struct AttnCtx {
    /// Key-padding mask `[n, 1, 1, len]`.
    pad: Array,
    local: LocalAttn,
}

/// Decision-head output as the scorer reads it.
///
/// Every consumer has to match all three variants: with `headprune` the last head layer is
/// computed at the scorer's rows only, so `Rows` holds no per-token output and cannot be
/// turned into one. Anything that needs every token's head output (per-token embeddings,
/// say) must run with `headprune=0` and take `Full` or `Packed`.
enum HeadOut {
    /// Every token, `[n, len, d]`.
    Full(Array),
    /// Every real token, `[T, d]` in [`Packing`] order (`unpad`).
    Packed(Array),
    /// The scorer's rows only (`headprune`).
    Rows(ScorerRows),
}

/// The last head layer at the rows the scorer reads and nowhere else: position 0 followed by
/// the `kmax` marker slots of each row, `[n, 1 + kmax, d]`. Padded marker slots hold position
/// 0's output (their logits are masked afterwards, as in the full layer). The array is private
/// so that it cannot be mistaken for a `[n, len, d]` per-token output.
struct ScorerRows(Array);

impl ScorerRows {
    /// Position 0 of every row, `[n, d]`: a range view of the `[n, 1 + kmax, d]` array with its
    /// middle axis squeezed (an integer index would gather a copy).
    fn pooled(&self) -> Result<Array> {
        self.0.index((.., ..1, ..)).squeeze_axes(&[1]).lx()
    }

    /// The marker slots of every row, `[n * kmax, d]`.
    fn markers(&self, n: i32, kmax: i32, d: i32) -> Result<Array> {
        self.0.index((.., 1.., ..)).reshape(&[n * kmax, d]).lx()
    }
}

/// Token packing for `unpad`: hidden states live as `[T, d]` over the real tokens only and
/// are expanded to the padded `[n, len, ...]` layout just for attention. Every row has at
/// least one real token (its `[CLS]`), which the padding positions borrow.
struct Packing {
    /// `[T]` padded positions `r * len + pos` of the packed tokens, row-major.
    pack: Array,
    /// `[n * len]` packed index of every padded position. Padding points at its row's token 0,
    /// so expanded values stay finite; they are masked as keys and dropped as queries.
    unpack: Array,
    /// `[T]` row of every packed token.
    row: Array,
    /// Packed index of position 0 of every row.
    offsets: Vec<u32>,
}

impl Packing {
    fn new(batch: &Batch) -> Self {
        let mut pack: Vec<u32> = Vec::with_capacity(batch.total_tokens());
        let mut unpack: Vec<u32> = Vec::with_capacity(batch.n * batch.len);
        let mut row: Vec<u32> = Vec::with_capacity(batch.total_tokens());
        let mut offsets = Vec::with_capacity(batch.n);
        for r in 0..batch.n {
            let first = pack.len() as u32;
            offsets.push(first);
            for pos in 0..batch.len {
                let flat = r * batch.len + pos;
                if batch.attention_mask[flat] == 1 {
                    unpack.push(pack.len() as u32);
                    pack.push(flat as u32);
                    row.push(r as u32);
                } else {
                    unpack.push(first);
                }
            }
        }
        Self {
            pack: Array::from_slice(&pack, &[pack.len() as i32]),
            unpack: Array::from_slice(&unpack, &[unpack.len() as i32]),
            row: Array::from_slice(&row, &[row.len() as i32]),
            offsets,
        }
    }

    /// `[T, c] -> [n, len, c]`.
    fn expand(&self, x: &Array, n: i32, len: i32) -> Result<Array> {
        x.take_axis(&self.unpack, 0).lx()?.reshape(&[n, len, x.dim(-1)]).lx()
    }

    /// `[n, len, c] -> [T, c]`.
    fn compact(&self, x: &Array, n: i32, len: i32) -> Result<Array> {
        x.reshape(&[n * len, x.dim(-1)]).lx()?.take_axis(&self.pack, 0).lx()
    }
}

/// Encoder output for one batch.
struct Encoded {
    /// `last_hidden_state`: `[n, len, d]`, or `[T, d]` when `packing` is set.
    h: Array,
    /// Additive key-padding mask `[n, 1, 1, len]`.
    pad: Array,
    packing: Option<Packing>,
}

// SAFETY: `mlx_rs::Array` is a reference-counted handle to immutable, already-evaluated
// data; MLX's scheduler is thread-safe and every forward selects its own stream. The
// mask cache is behind a `Mutex`.
unsafe impl Sync for MlxBackend {}

impl MlxBackend {
    fn new(w: &Weights, cfg: &ModelConfig, opts: &BackendOptions) -> Result<Self> {
        let cpu = matches!(opts.device, Device::Cpu);
        let dtype = if opts.f32 {
            Dtype::Float32
        } else {
            Dtype::Float16
        };
        let enc = &cfg.encoder;
        let eps = enc.norm_eps as f32;
        let knobs = Knobs::from_spec(opts.tuning.as_deref())?;
        let loader = Loader {
            st: w.view()?,
            dtype,
            wcopy: knobs.wcopy.clone(),
            directload: knobs.directload,
        };

        if std::env::var_os("SYS1_MLX").is_some() {
            eprintln!("laya-mlx knobs: {knobs:?}");
        }
        if let Some(mb) = knobs.cache_mb {
            mlx_rs::memory::set_cache_limit(mb << 20).lx()?;
        }
        if let Some(mb) = knobs.wired_mb {
            mlx_rs::memory::set_wired_limit(mb << 20).lx()?;
        }
        let stream = if cpu { Stream::cpu() } else { Stream::gpu() };
        mlx_rs::with_stream(&stream, || -> Result<Self> {
            let mut layers = Vec::with_capacity(enc.num_hidden_layers);
            for i in 0..enc.num_hidden_layers {
                let p = format!("encoder.layers.{i}");
                let local = enc.layer_is_local[i];
                layers.push(EncoderLayer {
                    attn_norm: if i == 0 {
                        None
                    } else {
                        Some(loader.norm(&format!("{p}.attn_norm.weight"), None, eps)?)
                    },
                    wqkv: loader.linear(&format!("{p}.attn.Wqkv.weight"), None)?,
                    wo: loader.linear(&format!("{p}.attn.Wo.weight"), None)?,
                    mlp_norm: loader.norm(&format!("{p}.mlp_norm.weight"), None, eps)?,
                    wi: loader.linear(&format!("{p}.mlp.Wi.weight"), None)?,
                    wo2: loader.linear(&format!("{p}.mlp.Wo.weight"), None)?,
                    rope_theta: if local {
                        enc.local_rope_theta
                    } else {
                        enc.global_rope_theta
                    } as f32,
                    local,
                });
            }
            let mut head = Vec::with_capacity(cfg.agent.head_layers);
            let d = enc.hidden_size as i32;
            for j in 0..cfg.agent.head_layers {
                let p = format!("head.layers.{j}");
                let (in_w, in_b) = (
                    format!("{p}.self_attn.in_proj_weight"),
                    format!("{p}.self_attn.in_proj_bias"),
                );
                let in_proj = loader.linear(&in_w, Some(&in_b))?;
                let split_proj = match (knobs.headprune && j + 1 == cfg.agent.head_layers, knobs.sharehead) {
                    (false, _) => None,
                    // Columns of `in_proj`'s `[in, 3d]` weight and its bias: views, no copy.
                    (true, true) => {
                        let cols = |c: std::ops::Range<i32>| Linear {
                            wt: in_proj.wt.index((.., c.clone())),
                            b: in_proj.b.as_ref().map(|b| b.index(c)),
                        };
                        Some((cols(0..d), cols(d..3 * d)))
                    }
                    (true, false) => Some((
                        loader.linear_rows(&in_w, Some(&in_b), 0..d)?,
                        loader.linear_rows(&in_w, Some(&in_b), d..3 * d)?,
                    )),
                };
                head.push(HeadLayer {
                    norm1: loader.norm(
                        &format!("{p}.norm1.weight"),
                        Some(&format!("{p}.norm1.bias")),
                        1e-5,
                    )?,
                    in_proj,
                    split_proj,
                    out_proj: loader.linear(
                        &format!("{p}.self_attn.out_proj.weight"),
                        Some(&format!("{p}.self_attn.out_proj.bias")),
                    )?,
                    norm2: loader.norm(
                        &format!("{p}.norm2.weight"),
                        Some(&format!("{p}.norm2.bias")),
                        1e-5,
                    )?,
                    linear1: loader.linear(
                        &format!("{p}.linear1.weight"),
                        Some(&format!("{p}.linear1.bias")),
                    )?,
                    linear2: loader.linear(
                        &format!("{p}.linear2.weight"),
                        Some(&format!("{p}.linear2.bias")),
                    )?,
                });
            }
            let (split_rope, band_from) = rope_kernels(&knobs, cpu, |band| {
                SplitRope::new(
                    enc.num_attention_heads,
                    enc.head_dim(),
                    dtype,
                    enc.global_rope_theta as f32,
                    enc.local_rope_theta as f32,
                    enc.local_attention / 2,
                    band,
                )
            });
            // `nax` is decided at load too. The kernels are checked against MLX's gemm at
            // every weight shape they may run: the encoder's plain products (wqkv, wi), its
            // residual products (wo, wo2) and the head's residual products (out_proj, linear2,
            // whose bias is added after the gemm, with or without the kernel). The head's other
            // linears add their bias inside MLX's gemm and stay with MLX.
            let nax = match (knobs.nax, cpu || dtype != Dtype::Float16 || knobs.wcopy == "t") {
                (false, _) => None,
                (true, true) => {
                    eprintln!("laya-mlx: nax needs an f16 model on the GPU with transposed-view weights (not wcopy=t); MLX's gemms run instead");
                    None
                }
                (true, false) => {
                    let mut mm: Vec<(i32, i32)> = Vec::new();
                    let mut addmm: Vec<(i32, i32)> = Vec::new();
                    let add = |list: &mut Vec<(i32, i32)>, l: &Linear| {
                        let shape = (l.wt.dim(1), l.wt.dim(0));
                        if !list.contains(&shape) {
                            list.push(shape);
                        }
                    };
                    for l in &layers {
                        add(&mut mm, &l.wqkv);
                        add(&mut mm, &l.wi);
                        add(&mut addmm, &l.wo);
                        add(&mut addmm, &l.wo2);
                    }
                    for h in &head {
                        add(&mut addmm, &h.out_proj);
                        add(&mut addmm, &h.linear2);
                    }
                    match NaxGemm::new(&mm, &addmm) {
                        Ok((k, note)) => {
                            eprintln!("laya-mlx: {note}");
                            Some(k)
                        }
                        Err(e) => {
                            eprintln!("laya-mlx: nax is off for this load, MLX's gemms run instead: {e}");
                            None
                        }
                    }
                }
            };
            let this = Self {
                cpu,
                dtype,
                hidden: enc.hidden_size,
                n_heads: enc.num_attention_heads,
                head_dim: enc.head_dim(),
                head_nheads: cfg.head_nheads(),
                window: enc.local_attention / 2,
                tok_emb: loader.get("encoder.embeddings.tok_embeddings.weight")?,
                emb_norm: loader.norm("encoder.embeddings.norm.weight", None, eps)?,
                layers,
                final_norm: loader.norm("encoder.final_norm.weight", None, eps)?,
                type_emb: loader.get("type_emb.weight")?,
                head,
                scorer_norm: loader.norm("scorer.0.weight", Some("scorer.0.bias"), 1e-5)?,
                scorer1: loader.linear("scorer.1.weight", Some("scorer.1.bias"))?,
                scorer3: loader.linear("scorer.3.weight", Some("scorer.3.bias"))?,
                band: if knobs.bool_mask {
                    band_mask_bool(enc.local_attention / 2)?
                } else {
                    band_mask(enc.local_attention / 2, dtype)?
                },
                caches: Mutex::new(Caches::default()),
                geglu: GeGlu::new(&knobs.geglu, knobs.f16gelu),
                split_rope,
                band_from,
                nax,
                knobs: knobs.clone(),
            };
            // Materialise every weight (and transposed view) once, up front.
            let all: Vec<&Array> = this.all_params();
            transforms::eval(all).lx()?;
            this.warm_buckets()?;
            Ok(this)
        })
    }

    /// One forward per (bucket length, warm row count), so the first real request of each
    /// shape finds its masks, gather indices, compiled traces and buffers ready.
    fn warm_buckets(&self) -> Result<()> {
        for &len in &self.knobs.buckets {
            for &n in &self.knobs.warm {
                let kmax = 2;
                let batch = Batch {
                    n,
                    len,
                    kmax,
                    input_ids: vec![0; n * len],
                    attention_mask: vec![1; n * len],
                    seq_lens: vec![len; n],
                    marker_pos: (0..n).flat_map(|_| [1u32, 2]).collect(),
                    marker_count: vec![kmax; n],
                    qtype: vec![0; n],
                };
                self.forward(&batch)?;
            }
        }
        Ok(())
    }

    fn all_params(&self) -> Vec<&Array> {
        fn push_lin<'a>(v: &mut Vec<&'a Array>, l: &'a Linear) {
            v.push(&l.wt);
            v.extend(l.b.as_ref());
        }
        fn push_norm<'a>(v: &mut Vec<&'a Array>, n: &'a Norm) {
            v.push(&n.w);
            v.extend(n.b.as_ref());
        }
        let mut v = vec![
            &self.tok_emb,
            &self.emb_norm.w,
            &self.final_norm.w,
            &self.type_emb,
        ];
        for l in &self.layers {
            if let Some(n) = &l.attn_norm {
                push_norm(&mut v, n);
            }
            push_lin(&mut v, &l.wqkv);
            push_lin(&mut v, &l.wo);
            push_norm(&mut v, &l.mlp_norm);
            push_lin(&mut v, &l.wi);
            push_lin(&mut v, &l.wo2);
        }
        for h in &self.head {
            push_norm(&mut v, &h.norm1);
            push_lin(&mut v, &h.in_proj);
            if let Some((q, kv)) = &h.split_proj {
                push_lin(&mut v, q);
                push_lin(&mut v, kv);
            }
            push_lin(&mut v, &h.out_proj);
            push_norm(&mut v, &h.norm2);
            push_lin(&mut v, &h.linear1);
            push_lin(&mut v, &h.linear2);
        }
        push_norm(&mut v, &self.scorer_norm);
        push_lin(&mut v, &self.scorer1);
        push_lin(&mut v, &self.scorer3);
        v
    }

    fn stream(&self) -> Stream {
        if self.cpu {
            Stream::cpu()
        } else {
            Stream::gpu()
        }
    }

    /// Additive key-padding mask `[n, 1, 1, len]` (0 for tokens, `MASK_NEG` for padding).
    fn pad_mask(&self, batch: &Batch) -> Result<Array> {
        let vals: Vec<f32> = batch
            .attention_mask
            .iter()
            .map(|&m| if m == 1 { 0.0 } else { MASK_NEG })
            .collect();
        let a = Array::from_slice(&vals, &[batch.n as i32, 1, 1, batch.len as i32]);
        a.as_dtype(self.dtype).lx()
    }

    /// Masks for this batch: dense window masks for short inputs, chunked gather indices and
    /// masks once the sequence is long enough for windowing to pay off, or the banded masks
    /// from the `band` length up. With `mask=bool` every mask is boolean, whichever path the
    /// length takes (`band` is off then). `packed` says whether the encoder packs this batch.
    fn attn_ctx(&self, batch: &Batch, packed: bool) -> Result<AttnCtx> {
        let has_local = self.layers.iter().any(|l| l.local);
        let dense_upto = self.knobs.dense_upto.unwrap_or(4 * self.window);
        let windowed = self.knobs.windowed && batch.len > dense_upto;
        let banded = self.band_from.is_some_and(|from| batch.len >= from);
        let (n, len) = (batch.n as i32, batch.len as i32);
        let pad = if self.knobs.bool_mask {
            Array::from_slice(&batch.attention_mask, &[n, 1, 1, len])
                .as_dtype(Dtype::Bool)
                .lx()?
        } else {
            self.pad_mask(batch)?
        };
        let local = if !has_local {
            LocalAttn::Dense(pad.clone())
        } else if banded {
            LocalAttn::Banded(self.banded_ctx(batch, packed)?)
        } else if windowed {
            LocalAttn::Windowed(self.windowed_ctx(batch)?)
        } else if self.knobs.bool_mask {
            // Padded queries may see every valid key so no softmax row is fully masked
            // (as Python laya-mlx); they are never used as keys or outputs.
            let pad_q = ops::logical_not(&pad.reshape(&[n, 1, len, 1]).lx()?).lx()?;
            let band = self.window_mask_bool(batch.len)?;
            LocalAttn::Dense(ops::logical_and(&ops::logical_or(&band, &pad_q).lx()?, &pad).lx()?)
        } else {
            LocalAttn::Dense(ops::add(&pad, &self.window_mask(batch.len)?).lx()?)
        };
        Ok(AttnCtx { pad, local })
    }

    /// The chunked path's constants for this batch: the key gather indices and the pad + band
    /// mask `[n * H, n_chunks, S, 3S]`, additive or boolean as `mask=` says.
    fn windowed_ctx(&self, batch: &Batch) -> Result<Windowed> {
        let (n, len, s) = (batch.n, batch.len, self.window);
        let nc = len.div_ceil(s);
        let ks = 3 * s;
        let key_idx = window_key_idx(s, n * self.n_heads, len, nc)?;
        let key_ok = window_key_valid(&batch.attention_mask, n, len, s, nc);
        let (n, nc, ks, s) = (n as i32, nc as i32, ks as i32, s as i32);
        let mask = if self.knobs.bool_mask {
            let valid = Array::from_slice(&key_ok, &[n, 1, nc, 1, ks]);
            // A padded query (padding in its row, or past `len` in the last chunk) may see every
            // gathered slot: an all-false row is NaN with boolean masks, and the outputs of these
            // rows are never read. The dense boolean path does the same with `pad_q`.
            let pad_q = window_query_padded(&batch.attention_mask, batch.n, len, self.window, nc as usize);
            let pad_q = Array::from_slice(&pad_q, &[n, 1, nc, s, 1]);
            ops::logical_or(&ops::logical_and(&valid, &self.band).lx()?, &pad_q).lx()?
        } else {
            let vals: Vec<f32> = key_ok.iter().map(|&ok| if ok { 0.0 } else { MASK_NEG }).collect();
            let valid = Array::from_slice(&vals, &[n, 1, nc, 1, ks])
                .as_dtype(self.dtype)
                .lx()?;
            ops::add(&valid, &self.band).lx()?
        };
        let h = self.n_heads as i32;
        let mask = ops::broadcast_to(&mask, &[n, h, nc, s, ks])
            .lx()?
            .reshape(&[n * h, nc, s, ks])
            .lx()?;
        Ok(Windowed {
            key_idx,
            mask,
            n_chunks: nc,
        })
    }

    /// The banded path's constants for this batch (see [`Banded`]). The mask holds the dense
    /// path's values: [`window_key_valid`] is the pad mask at each slot's position, `MASK_NEG`
    /// outside the sequence, and `self.band` is the window mask at each slot's offset.
    fn banded_ctx(&self, batch: &Batch, packed: bool) -> Result<Banded> {
        let (n, len, s) = (batch.n, batch.len, self.window);
        let nc = len.div_ceil(s);
        let ks = 3 * s;
        let key_ok = window_key_valid(&batch.attention_mask, n, len, s, nc);
        let vals: Vec<f32> = key_ok.iter().map(|&ok| if ok { 0.0 } else { MASK_NEG }).collect();
        let valid = Array::from_slice(&vals, &[(n * nc) as i32, 1, 1, ks as i32]).as_dtype(self.dtype).lx()?;
        let band = self.band.reshape(&[1, 1, s as i32, ks as i32]).lx()?;
        let mask = ops::add(&valid, &band).lx()?;
        let pack = packed.then(|| {
            let pack: Vec<u32> = (0..n)
                .flat_map(|r| {
                    (0..len)
                        .filter(move |&pos| batch.attention_mask[r * len + pos] == 1)
                        .map(move |pos| (r * nc * s + pos) as u32)
                })
                .collect();
            Array::from_slice(&pack, &[pack.len() as i32])
        });
        Ok(Banded {
            mask,
            n_chunks: nc as i32,
            pack,
        })
    }

    fn lock_caches(&self) -> Result<std::sync::MutexGuard<'_, Caches>> {
        self.caches
            .lock()
            .map_err(|_| Error::Backend("mask cache poisoned".into()))
    }

    /// Boolean sliding-window mask `[1, 1, len, len]` (true = may attend); see
    /// [`Self::window_mask`] for the caching.
    fn window_mask_bool(&self, len: usize) -> Result<Array> {
        let mut caches = self.lock_caches()?;
        if caches.window_bool.as_ref().is_none_or(|m| (m.dim(2) as usize) < len) {
            let mut vals = vec![false; len * len];
            for i in 0..len {
                for j in 0..len {
                    vals[i * len + j] = i.abs_diff(j) <= self.window;
                }
            }
            let m = Array::from_slice(&vals, &[1, 1, len as i32, len as i32]);
            m.eval().lx()?;
            caches.window_bool = Some(m);
        }
        Ok(Self::window_view(caches.window_bool.as_ref().expect("set above"), len))
    }

    /// `x @ W^T` for a linear without bias: the `nax` kernel where it has a configuration for
    /// the shape, MLX's gemm otherwise.
    fn mm(&self, l: &Linear, x: &Array) -> Result<Array> {
        if let (Some(kernel), None) = (&self.nax, &l.b) {
            // The kernel takes the weight as stored, `[out, in]`: `wt` is a transposed view of
            // it, so this transpose is a view too.
            if let Some(y) = kernel.matmul(x, &ops::transpose(&l.wt).lx()?)? {
                return Ok(y);
            }
        }
        l.apply(x)
    }

    /// `residual + x @ W^T (+ b)`, fused into the gemm unless `addmm=0`.
    fn lin_add(&self, l: &Linear, x: &Array, residual: &Array) -> Result<Array> {
        if self.knobs.addmm {
            if let Some(kernel) = &self.nax {
                // As in `mm`, a view of the weight as stored. The bias goes on after the gemm,
                // as in `Linear::apply_add`.
                if let Some(y) = kernel.addmm(x, &ops::transpose(&l.wt).lx()?, residual)? {
                    return match &l.b {
                        Some(b) => ops::add(&y, b).lx(),
                        None => Ok(y),
                    };
                }
            }
            return l.apply_add(x, residual);
        }
        ops::add(&l.apply(x)?, residual).lx()
    }

    /// Additive sliding-window mask `[1, 1, len, len]`. Whether `i` may see `j` depends on
    /// `|i - j|` alone, so the mask for `len` is the top-left corner of any longer one: one
    /// array for the longest length so far is kept and shorter lengths get a view of it.
    ///
    /// The new mask is evaluated here, on purpose: the cached array is shared by every later
    /// forward and must hold data, not a graph node those forwards would all point into. This
    /// happens once per new longest length, a few times in the life of a server.
    fn window_mask(&self, len: usize) -> Result<Array> {
        let mut caches = self.lock_caches()?;
        if caches.window_mask.as_ref().is_none_or(|m| (m.dim(2) as usize) < len) {
            let mut vals = vec![0f32; len * len];
            for i in 0..len {
                for j in 0..len {
                    if i.abs_diff(j) > self.window {
                        vals[i * len + j] = MASK_NEG;
                    }
                }
            }
            let m = Array::from_slice(&vals, &[1, 1, len as i32, len as i32])
                .as_dtype(self.dtype)
                .lx()?;
            m.eval().lx()?;
            caches.window_mask = Some(m);
        }
        Ok(Self::window_view(caches.window_mask.as_ref().expect("set above"), len))
    }

    /// The `[1, 1, len, len]` corner of a `[1, 1, L, L]` window mask, `L >= len`.
    fn window_view(m: &Array, len: usize) -> Array {
        if m.dim(2) as usize == len {
            m.clone()
        } else {
            m.index((.., .., ..len as i32, ..len as i32))
        }
    }

    /// Sliding-window attention by chunks. Queries `[n, H, len, hd]` are padded to a multiple
    /// of `S = window` and viewed as `[n*H, chunks, S, hd]`; keys/values are gathered into
    /// `[n*H, chunks, 3S, hd]` (chunks c-1, c, c+1, clamped, masked when out of range), so
    /// each chunk runs a small fused attention instead of a `len x len` one.
    fn windowed_attention(
        &self,
        q: &Array,
        k: &Array,
        v: &Array,
        w: &Windowed,
        scale: f32,
    ) -> Result<Array> {
        let (n, h, len, hd) = (q.dim(0), q.dim(1), q.dim(2), q.dim(3));
        let s = self.window as i32;
        let n_chunks = w.n_chunks;
        let lp = n_chunks * s;
        let q = if lp != len {
            ops::pad(
                q,
                &[(0, 0), (0, 0), (0, lp - len), (0, 0)],
                None::<Array>,
                None::<ops::PadMode>,
            )
            .lx()?
        } else {
            q.clone()
        };
        let q = q.reshape(&[n * h, n_chunks, s, hd]).lx()?;
        // Gather whole `hd` rows from the flattened `[n*H*len, hd]` view: several times faster
        // than `take_axis` on axis 2.
        let gather = |x: &Array| -> Result<Array> {
            x.reshape(&[n * h * len, hd])
                .lx()?
                .take_axis(&w.key_idx, 0)
                .lx()?
                .reshape(&[n * h, n_chunks, 3 * s, hd])
                .lx()
        };
        let (k, v) = (gather(k)?, gather(v)?);
        let att =
            fast::scaled_dot_product_attention(&q, &k, &v, scale, &w.mask, None::<&Array>).lx()?;
        let att = att.reshape(&[n, h, lp, hd]).lx()?;
        Ok(if lp != len {
            att.index((.., .., ..len, ..))
        } else {
            att
        })
    }

    /// `[n, len, heads * hd] -> [n, heads, len, hd]`.
    fn split_heads(&self, x: &Array, n: i32, len: i32, heads: usize) -> Result<Array> {
        let hd = x.dim(-1) / heads as i32;
        x.reshape(&[n, len, heads as i32, hd])
            .lx()?
            .transpose_axes(&[0, 2, 1, 3])
            .lx()
    }

    /// `[n, heads, len, hd] -> [n, len, d]`.
    fn merge_heads(&self, x: &Array, n: i32, len: i32) -> Result<Array> {
        x.transpose_axes(&[0, 2, 1, 3])
            .lx()?
            .reshape(&[n, len, self.hidden as i32])
            .lx()
    }

    /// Encoder `qkv [n, len, 3d]` -> roped `q`, roped `k` and `v`, each `[n, H, len, hd]`, with
    /// the MLX ops (the reference the `fuserope` kernel is checked against).
    fn qkv_rope(&self, qkv: &Array, n: i32, len: i32, theta: f32) -> Result<(Array, Array, Array)> {
        split_rope::mlx_path(qkv, n, len, self.n_heads as i32, self.head_dim as i32, theta)
    }

    /// Token index of position 0 of every row: `r * len` in the padded layout, the packing
    /// offsets when the hidden states are packed.
    fn row_starts(batch: &Batch, packing: Option<&Packing>) -> Vec<u32> {
        match packing {
            Some(p) => p.offsets.clone(),
            None => (0..batch.n as u32).map(|r| r * batch.len as u32).collect(),
        }
    }

    /// Flat token indices of the rows the scorer reads: position 0 of every row followed by
    /// its `kmax` marker slots (padded slots point at position 0, as in `Batch`).
    fn scorer_rows(batch: &Batch, starts: &[u32]) -> Array {
        let kmax = batch.kmax;
        let flat: Vec<u32> = (0..batch.n)
            .flat_map(|r| {
                let base = starts[r];
                std::iter::once(base)
                    .chain(batch.marker_pos[r * kmax..(r + 1) * kmax].iter().map(move |&p| base + p))
            })
            .collect();
        Array::from_slice(&flat, &[flat.len() as i32])
    }

    /// ModernBERT encoder; returns `last_hidden_state` (`[n, len, d]`, or packed `[T, d]` when
    /// `unpad` is allowed and the batch has padding) and the pad mask.
    fn encode(&self, batch: &Batch, unpad: bool) -> Result<Encoded> {
        let (n, len) = (batch.n as i32, batch.len as i32);
        let d = self.hidden as i32;
        // `unpad` only pays off when there is padding; a batch without any runs the plain path.
        let packing = if unpad && batch.total_tokens() < batch.n * batch.len {
            Some(Packing::new(batch))
        } else {
            None
        };
        let mut h = match &packing {
            Some(_) => {
                let ids: Vec<u32> = batch
                    .input_ids
                    .iter()
                    .zip(&batch.attention_mask)
                    .filter(|(_, &m)| m == 1)
                    .map(|(&id, _)| id)
                    .collect();
                let ids = Array::from_slice(&ids, &[ids.len() as i32]);
                self.tok_emb.take_axis(&ids, 0).lx()?
            }
            None => {
                let ids = Array::from_slice(&batch.input_ids, &[batch.n as i32 * len]);
                self.tok_emb.take_axis(&ids, 0).lx()?.reshape(&[n, len, d]).lx()?
            }
        };
        h = self.emb_norm.apply(&h)?;

        let ctx = self.attn_ctx(batch, packing.is_some())?;
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let mut lt: Vec<u128> = Vec::new();
        if self.knobs.layers {
            let t = std::time::Instant::now();
            transforms::eval([&h, &ctx.pad]).lx()?;
            if let LocalAttn::Dense(m) = &ctx.local {
                m.eval().lx()?;
            }
            lt.push(t.elapsed().as_micros());
        }

        let mut op_us: Vec<(&str, u128)> = Vec::new();
        let mut t_op = std::time::Instant::now();
        let mut mark = |name: &'static str, arrs: &[&Array]| -> Result<()> {
            if self.knobs.ops {
                transforms::eval(arrs.iter().copied()).lx()?;
                let us = t_op.elapsed().as_micros();
                match op_us.iter_mut().find(|(k, _)| *k == name) {
                    Some(e) => e.1 += us,
                    None => op_us.push((name, us)),
                }
                t_op = std::time::Instant::now();
            }
            Ok(())
        };
        mark("emb+masks", &[&h])?;
        // The `fuserope` kernel with `[n, len, nc]` as its `dims` input (`nc` is the chunk count
        // of the banded layout, read by the banded launches only): the batch shape goes in as
        // data, so one pipeline serves every shape. Nothing is built when the kernel is off.
        let fused = self.split_rope.as_ref().map(|kernel| {
            let nc = match &ctx.local {
                LocalAttn::Banded(b) => b.n_chunks,
                _ => 1,
            };
            (kernel, Array::from_slice(&[n, len, nc], &[3]))
        });
        for layer in &self.layers {
            let t = std::time::Instant::now();
            let a = match &layer.attn_norm {
                Some(norm) => norm.apply(&h)?,
                None => h.clone(),
            };
            mark("attn_norm", &[&a])?;
            let qkv = self.mm(&layer.wqkv, &a)?;
            // Attention output as `[n, len, d]`, or `[T, d]` when packed.
            let att = match (&ctx.local, layer.local, &fused) {
                (LocalAttn::Banded(bd), true, Some((kernel, dims))) => {
                    mark("wqkv", &[&qkv])?;
                    let unpack = packing.as_ref().map(|p| &p.unpack);
                    let (nc, s) = (bd.n_chunks, self.window as i32);
                    let (q, k, v) = if n == 1 {
                        // One row: every position written once, the chunk windows as views.
                        let (q, k, v) = kernel.apply_band(&qkv, unpack, dims, layer.local, n, nc, true)?;
                        split_rope::band_views(&q, &k, &v, nc, s, self.n_heads as i32, self.head_dim as i32)?
                    } else {
                        kernel.apply_band(&qkv, unpack, dims, layer.local, n, nc, false)?
                    };
                    mark("split+rope", &[&q, &k, &v])?;
                    let att = fast::scaled_dot_product_attention(&q, &k, &v, scale, &bd.mask, None::<&Array>).lx()?;
                    mark("sdpa_local", &[&att])?;
                    // `[n * nc, H, S, hd]` -> `[n, nc * S, d]`: heads merged, chunks back in
                    // position order.
                    let att = att.transpose_axes(&[0, 2, 1, 3]).lx()?.reshape(&[n, nc * s, d]).lx()?;
                    match &bd.pack {
                        Some(pack) => att.reshape(&[n * nc * s, d]).lx()?.take_axis(pack, 0).lx()?,
                        None if nc * s == len => att,
                        None => att.index((.., ..len, ..)),
                    }
                }
                (LocalAttn::Banded(_), true, None) => {
                    return Err(Error::Backend("band: the banded layout needs the fuserope kernel".into()));
                }
                _ => {
                    let (q, k, v) = match &fused {
                        // The kernel reads the packed rows through the unpack index itself.
                        Some((kernel, dims)) => {
                            mark("wqkv", &[&qkv])?;
                            kernel.apply(&qkv, packing.as_ref().map(|p| &p.unpack), dims, layer.local, n, len)?
                        }
                        None => {
                            // Packed: back to `[n, len, 3d]` for attention.
                            let qkv = match &packing {
                                Some(p) => p.expand(&qkv, n, len)?,
                                None => qkv,
                            };
                            mark("wqkv", &[&qkv])?;
                            self.qkv_rope(&qkv, n, len, layer.rope_theta)?
                        }
                    };
                    mark("split+rope", &[&q, &k, &v])?;
                    let att = match (&ctx.local, layer.local) {
                        (LocalAttn::Windowed(w), true) => self.windowed_attention(&q, &k, &v, w, scale)?,
                        (LocalAttn::Dense(mask), true) => {
                            fast::scaled_dot_product_attention(&q, &k, &v, scale, mask, None::<&Array>)
                                .lx()?
                        }
                        (LocalAttn::Banded(_), true) => {
                            return Err(Error::Backend("band: a local layer missed the banded path".into()));
                        }
                        (_, false) => {
                            fast::scaled_dot_product_attention(&q, &k, &v, scale, &ctx.pad, None::<&Array>)
                                .lx()?
                        }
                    };
                    mark(if layer.local { "sdpa_local" } else { "sdpa_global" }, &[&att])?;
                    let att = self.merge_heads(&att, n, len)?;
                    match &packing {
                        Some(p) => p.compact(&att, n, len)?,
                        None => att,
                    }
                }
            };
            h = self.lin_add(&layer.wo, &att, &h)?;
            mark("merge+wo", &[&h])?;

            let m = layer.mlp_norm.apply(&h)?;
            mark("mlp_norm", &[&m])?;
            let wi = self.mm(&layer.wi, &m)?;
            mark("wi", &[&wi])?;
            let act = self.geglu.apply(&wi)?;
            mark("geglu", &[&act])?;
            h = self.lin_add(&layer.wo2, &act, &h)?;
            mark("wo2", &[&h])?;
            if self.knobs.layers {
                h.eval().lx()?;
                lt.push(t.elapsed().as_micros());
            }
        }
        if self.knobs.ops {
            eprintln!("ops_us n={} len={} h_dtype={:?} {:?}", batch.n, batch.len, h.dtype(), op_us);
        }
        if self.knobs.layers {
            eprintln!("layers_us n={} len={} emb+masks={} layers={:?}", batch.n, batch.len, lt[0], &lt[1..]);
        }
        Ok(Encoded {
            h: self.final_norm.apply(&h)?,
            pad: ctx.pad,
            packing,
        })
    }

    /// Decision-head transformer layers on top of the encoder output. With `headprune` the
    /// last layer is only computed at the scorer's rows, and the result says so in its type
    /// (see [`HeadOut`]).
    fn head_forward(&self, batch: &Batch, enc: &Encoded) -> Result<HeadOut> {
        let (n, len) = (batch.n as i32, batch.len as i32);
        let d = self.hidden as i32;
        let packing = enc.packing.as_ref();
        let qtype = Array::from_slice(&batch.qtype, &[n]);
        let te = self.type_emb.take_axis(&qtype, 0).lx()?;
        let mut h = match packing {
            // Packed tokens take their row's type embedding row by row.
            Some(p) => ops::add(&enc.h, &te.take_axis(&p.row, 0).lx()?).lx()?,
            None => ops::add(&enc.h, &te.reshape(&[n, 1, d]).lx()?).lx()?,
        };
        let scale = head_scale(self.hidden, self.head_nheads);
        let sdpa = |q: &Array, k: &Array, v: &Array| -> Result<Array> {
            fast::scaled_dot_product_attention(q, k, v, scale, &enc.pad, None::<&Array>).lx()
        };
        // Around attention only: `[T, c] -> [n, len, c]` and back (identity when not packed).
        let expand = |x: Array| -> Result<Array> {
            match packing {
                Some(p) => p.expand(&x, n, len),
                None => Ok(x),
            }
        };
        let compact = |x: Array| -> Result<Array> {
            match packing {
                Some(p) => p.compact(&x, n, len),
                None => Ok(x),
            }
        };
        for layer in &self.head {
            let x = layer.norm1.apply(&h)?;
            if let Some((wq, wkv)) = &layer.split_proj {
                // `headprune` (last layer only): every token still supplies a key and a value,
                // but queries, out_proj, norm2 and the FFN run only for the rows the scorer
                // reads. A token's output depends on the other tokens only through attention,
                // so the picked rows come out the same as in the full layer.
                let kv = expand(wkv.apply(&x)?)?;
                let parts = kv.split_equal(2, -1).lx()?;
                let k = self.split_heads(&parts[0], n, len, self.head_nheads)?;
                let v = self.split_heads(&parts[1], n, len, self.head_nheads)?;
                let r = 1 + batch.kmax as i32;
                let idx = Self::scorer_rows(batch, &Self::row_starts(batch, packing));
                let pick = |a: &Array| -> Result<Array> {
                    a.reshape(&[-1, d]).lx()?.take_axis(&idx, 0).lx()?.reshape(&[n, r, d]).lx()
                };
                let q = self.split_heads(&wq.apply(&pick(&x)?)?, n, r, self.head_nheads)?;
                let att = self.merge_heads(&sdpa(&q, &k, &v)?, n, r)?;
                let hs = self.lin_add(&layer.out_proj, &att, &pick(&h)?)?;
                let x = layer.norm2.apply(&hs)?;
                let x = nn::relu(&layer.linear1.apply(&x)?).lx()?;
                return Ok(HeadOut::Rows(ScorerRows(self.lin_add(&layer.linear2, &x, &hs)?)));
            }
            let qkv = expand(layer.in_proj.apply(&x)?)?;
            let parts = qkv.split_equal(3, -1).lx()?;
            let q = self.split_heads(&parts[0], n, len, self.head_nheads)?;
            let k = self.split_heads(&parts[1], n, len, self.head_nheads)?;
            let v = self.split_heads(&parts[2], n, len, self.head_nheads)?;
            let att = compact(self.merge_heads(&sdpa(&q, &k, &v)?, n, len)?)?;
            h = self.lin_add(&layer.out_proj, &att, &h)?;

            let x = layer.norm2.apply(&h)?;
            let x = nn::relu(&layer.linear1.apply(&x)?).lx()?;
            h = self.lin_add(&layer.linear2, &x, &h)?;
        }
        Ok(match packing {
            Some(_) => HeadOut::Packed(h),
            None => HeadOut::Full(h),
        })
    }

    /// Scorer logits `[n * kmax]` (unmasked) and pooled `[n, d]` as f32 arrays.
    fn score(&self, batch: &Batch, h: &HeadOut, packing: Option<&Packing>) -> Result<(Array, Array)> {
        let (n, len, kmax) = (batch.n as i32, batch.len as i32, batch.kmax as i32);
        let d = self.hidden as i32;
        let (pooled, m) = match h {
            HeadOut::Full(h) => {
                let pooled = h
                    .take_axis(Array::from_slice(&[0u32], &[1]), 1)
                    .lx()?
                    .reshape(&[n, d])
                    .lx()?;
                let flat: Vec<u32> = (0..batch.n)
                    .flat_map(|r| (0..batch.kmax).map(move |k| (r, k)))
                    .map(|(r, k)| r as u32 * len as u32 + batch.marker_pos[r * batch.kmax + k])
                    .collect();
                let idx = Array::from_slice(&flat, &[n * kmax]);
                (pooled, h.reshape(&[n * len, d]).lx()?.take_axis(&idx, 0).lx()?)
            }
            HeadOut::Packed(h) => {
                let starts = Self::row_starts(batch, packing);
                let pooled = h.take_axis(Array::from_slice(&starts, &[n]), 0).lx()?;
                let flat: Vec<u32> = (0..batch.n)
                    .flat_map(|r| (0..batch.kmax).map(move |k| (r, k)))
                    .map(|(r, k)| starts[r] + batch.marker_pos[r * batch.kmax + k])
                    .collect();
                (pooled, h.take_axis(Array::from_slice(&flat, &[n * kmax]), 0).lx()?)
            }
            HeadOut::Rows(rows) => (rows.pooled()?, rows.markers(n, kmax, d)?),
        };
        let s = self.scorer_norm.apply(&m)?;
        let s = gelu_erf_as(&self.scorer1.apply(&s)?, self.knobs.f16gelu).lx()?;
        let logits = self.scorer3.apply(&s)?.reshape(&[n * kmax]).lx()?;
        Ok((to_f32_contiguous(&logits)?, to_f32_contiguous(&pooled)?))
    }
}

impl Backend for MlxBackend {
    fn name(&self) -> String {
        let dev = if self.cpu { "cpu" } else { "gpu" };
        let dt = if self.dtype == Dtype::Float32 {
            "f32"
        } else {
            "f16"
        };
        format!("mlx({dev},{dt})")
    }

    fn active_kernels(&self) -> &'static [&'static str] {
        // One static list per combination: nothing is allocated per load.
        match (&self.split_rope, self.band_from, &self.nax) {
            (Some(_), Some(_), Some(_)) => &["fuserope", "band", "nax"],
            (Some(_), Some(_), None) => &["fuserope", "band"],
            (Some(_), None, Some(_)) => &["fuserope", "nax"],
            (Some(_), None, None) => &["fuserope"],
            (None, _, Some(_)) => &["nax"],
            (None, _, None) => &[],
        }
    }

    fn padded_len(&self, len: usize, _rows: usize) -> usize {
        if let Some(&b) = self.knobs.buckets.iter().find(|&&b| b >= len) {
            return b;
        }
        len.div_ceil(self.knobs.pad) * self.knobs.pad
    }

    fn forward(&self, batch: &Batch) -> Result<BackendOutput> {
        let stream = self.stream();
        mlx_rs::with_stream(&stream, || {
            let t0 = std::time::Instant::now();
            let enc = self.encode(batch, self.knobs.unpad)?;
            let h = self.head_forward(batch, &enc)?;
            let (logits, pooled) = self.score(batch, &h, enc.packing.as_ref())?;
            let t1 = std::time::Instant::now();
            transforms::eval([&logits, &pooled]).lx()?;
            if self.knobs.trace {
                eprintln!(
                    "forward n={} len={} build_us={} eval_us={}",
                    batch.n,
                    batch.len,
                    (t1 - t0).as_micros(),
                    t1.elapsed().as_micros()
                );
            }
            let mut logits = host_f32(&logits)?;
            for r in 0..batch.n {
                for k in batch.marker_count[r]..batch.kmax {
                    logits[r * batch.kmax + k] = LOGIT_MASKED;
                }
            }
            let pooled = host_f32(&pooled)?;
            if self.knobs.clear {
                mlx_rs::memory::clear_cache().lx()?;
            }
            Ok(BackendOutput { logits, pooled })
        })
    }

    /// The full `[n, len, d]` last hidden state, computed at every position including padding.
    /// This debug hook therefore runs the encoder in the padded layout whatever `unpad` says:
    /// the packed path never computes the padding positions, and filling them in with some
    /// other token's state would look like a computed result to the parity harness.
    fn encoder_hidden(&self, batch: &Batch) -> Result<Option<Vec<f32>>> {
        let stream = self.stream();
        mlx_rs::with_stream(&stream, || {
            let enc = self.encode(batch, false)?;
            debug_assert!(enc.packing.is_none());
            let h = to_f32_contiguous(&enc.h)?;
            h.eval().lx()?;
            Ok(Some(host_f32(&h)?))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `same_bits` tells a -0 from a 0 and passes a NaN against the same NaN, in f32, f16 and
    /// bf16; a different dtype or shape with the same bits is not the same.
    #[test]
    fn same_bits_tells_signed_zeros_apart() {
        let pos = Array::from_slice(&[1.5f32, 0.0, f32::NAN], &[3]);
        let neg = Array::from_slice(&[1.5f32, -0.0, f32::NAN], &[3]);
        for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
            let (p, n) = (pos.as_dtype(dtype).unwrap(), neg.as_dtype(dtype).unwrap());
            assert!(ops::eq(&p, &n).unwrap().index(1).item_exact::<bool>(), "{dtype:?}: == passes -0 for 0");
            assert!(!same_bits(&p, &n).unwrap(), "{dtype:?}: -0 against 0");
            assert!(same_bits(&p, &p).unwrap(), "{dtype:?}: NaN against itself");
        }
        let h = pos.as_dtype(Dtype::Float16).unwrap();
        assert!(!same_bits(&h, &h.view_dtype(Dtype::Bfloat16).unwrap()).unwrap(), "f16 against bf16 with the same bits");
        assert!(!same_bits(&pos, &pos.reshape(&[3, 1]).unwrap()).unwrap(), "[3] against [3, 1]");
    }

    /// The load's `fuserope` and `band` decision. A kernel that passes its check serves both.
    /// One whose banded launches fail keeps `fuserope` with `band` off, so `active_kernels`
    /// lists fuserope alone. One that fails to build or whose plain launch fails turns both
    /// off. With `mask=bool` the banded launches are not built or checked, and a CPU load
    /// builds no kernel. Uses the GPU.
    #[test]
    fn a_failed_band_check_keeps_fuserope() {
        let knobs = Knobs::parse("fuserope,band=512").unwrap();
        let t = 10_000.0;
        let (k, band_from) = rope_kernels(&knobs, false, |band| SplitRope::new(2, 8, Dtype::Float16, t, t, 4, band));
        assert!(k.is_some_and(|k| k.band_checked()) && band_from == Some(512));
        let (k, band_from) = rope_kernels(&knobs, false, |band| SplitRope::with_wrong_band(2, 8, Dtype::Float16, t, t, 4, band));
        assert!(k.is_some_and(|k| !k.band_checked()) && band_from.is_none());
        let (k, band_from) = rope_kernels(&knobs, false, |_| Err(Error::Backend("fuserope self-check: the plain launch differs".into())));
        assert!(k.is_none() && band_from.is_none());
        let (k, band_from) = rope_kernels(&knobs, true, |_| panic!("a CPU load builds no kernel"));
        assert!(k.is_none() && band_from.is_none());
        let bool_mask = Knobs::parse("fuserope,band=512,mask=bool").unwrap();
        let (k, band_from) = rope_kernels(&bool_mask, false, |band| {
            assert!(!band, "mask=bool needs no banded launches");
            SplitRope::new(2, 8, Dtype::Float16, t, t, 4, band)
        });
        assert!(k.is_some() && band_from.is_none());
    }

    #[test]
    fn knobs_default_off_and_parse() {
        let k = Knobs::from_spec(Some("")).unwrap();
        assert_eq!(k.dense_upto, None);
        assert!(!k.headprune && !k.unpad && !k.fuserope);
        let k = Knobs::from_spec(Some("dense_upto=512,headprune,unpad,fuserope")).unwrap();
        assert_eq!(k.dense_upto, Some(512));
        assert!(k.headprune && k.unpad && k.fuserope);
        let k = Knobs::from_spec(Some("headprune=0,unpad=0,fuserope=0")).unwrap();
        assert!(!k.headprune && !k.unpad && !k.fuserope);
        assert!(Knobs::from_spec(Some("fuserope=1")).unwrap().fuserope);
        assert_eq!(k.band, 0);
        assert_eq!(Knobs::from_spec(Some("band=512")).unwrap().band, 512);
        assert_eq!(Knobs::from_spec(Some("band=512,band=0")).unwrap().band, 0);
        assert!(Knobs::from_spec(Some("band")).is_err());
        assert!(Knobs::from_spec(Some("band=-1")).is_err());
        assert!(!k.nax);
        assert!(Knobs::from_spec(Some("nax=all")).unwrap().nax);
        assert!(!Knobs::from_spec(Some("nax=all,nax=0")).unwrap().nax);
        assert!(Knobs::from_spec(Some("nax")).is_err());
        assert!(Knobs::from_spec(Some("nax=split")).is_err());
        assert!(!k.directload && !k.sharehead);
        let k = Knobs::from_spec(Some("directload,sharehead,parallel_load")).unwrap();
        assert!(k.directload && k.sharehead);
        let k = Knobs::from_spec(Some("directload=1,sharehead=1,directload=0,sharehead=0,parallel_load=0")).unwrap();
        assert!(!k.directload && !k.sharehead);
        for bad in ["directload=2", "sharehead=yes", "parallel_load=2"] {
            assert!(Knobs::from_spec(Some(bad)).is_err(), "{bad}");
        }
    }

    /// `directload` gives the same arrays as the conversion path, bit for bit: an F16 tensor
    /// with zeros of both signs, a subnormal, the largest finite f16, an infinity and a NaN,
    /// and the BF16 and F32 tensors that take the conversion path anyway. Only the F16 tensor
    /// into an f16 model, at an even address, is copied directly; the same file placed at an
    /// odd address puts every tensor there and takes the conversion path.
    #[test]
    fn directload_gives_the_same_bits() {
        let le16 = |v: &[u16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let f16 = le16(&[0x0000, 0x8000, 0x0001, 0x3c00, 0xbc00, 0x7bff, 0x7c00, 0x7e01]);
        let bf16 = le16(&[0x3f80, 0xc000, 0x0000, 0x4049]);
        let f32: Vec<u8> = [1.5f32, -0.0, 3.25e-5, 65504.0].iter().flat_map(|x| x.to_le_bytes()).collect();
        use safetensors::Dtype as St;
        let views = [
            ("h", TensorView::new(St::F16, vec![2, 4], &f16).unwrap()),
            ("b", TensorView::new(St::BF16, vec![4], &bf16).unwrap()),
            ("f", TensorView::new(St::F32, vec![2, 2], &f32).unwrap()),
        ];
        let file = safetensors::serialize(views, None).unwrap();
        // Every tensor's data starts at `8 + header + offset`, and every offset is even.
        let data_start = 8 + u64::from_le_bytes(file[..8].try_into().unwrap()) as usize;
        for odd in [false, true] {
            let mut storage = vec![0u8; file.len() + 1];
            let base_odd = (storage.as_ptr() as usize + data_start) % 2 == 1;
            let at = usize::from(base_odd != odd);
            storage[at..at + file.len()].copy_from_slice(&file);
            let bytes = &storage[at..at + file.len()];
            for dtype in [Dtype::Float16, Dtype::Float32] {
                let loader = |directload| Loader {
                    st: SafeTensors::deserialize(bytes).unwrap(),
                    dtype,
                    wcopy: "view".into(),
                    directload,
                };
                let (plain, direct) = (loader(false), loader(true));
                for name in ["h", "b", "f"] {
                    let t = direct.st.tensor(name).unwrap();
                    let want = !odd && dtype == Dtype::Float16 && name == "h";
                    assert_eq!(direct.direct(&t), want, "{name}, {dtype:?}, odd address {odd}");
                    assert!(!plain.direct(&t));
                    let (x, y) = (plain.get(name).unwrap(), direct.get(name).unwrap());
                    assert_eq!((x.shape(), x.dtype()), (y.shape(), dtype));
                    assert_eq!(y.dtype(), dtype);
                    assert!(same_bits(&x, &y).unwrap(), "{name}, {dtype:?}, odd address {odd}");
                }
            }
        }
    }

    /// The padding rows of a batch go through the packing as their row's token 0, and every
    /// scorer row (position 0 and the marker slots) lands on the right packed token.
    #[test]
    fn packing_indexes_real_tokens_and_scorer_rows() {
        let batch = Batch {
            n: 3,
            len: 4,
            kmax: 2,
            input_ids: vec![1, 2, 0, 0, 3, 4, 5, 6, 7, 0, 0, 0],
            attention_mask: vec![1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            seq_lens: vec![2, 4, 1],
            marker_pos: vec![1, 0, 1, 3, 0, 0],
            marker_count: vec![1, 2, 0],
            qtype: vec![0, 1, 2],
        };
        let p = Packing::new(&batch);
        assert_eq!(p.offsets, vec![0, 2, 6]);
        assert_eq!(p.pack.as_slice::<u32>(), &[0, 1, 4, 5, 6, 7, 8]);
        assert_eq!(p.unpack.as_slice::<u32>(), &[0, 1, 0, 0, 2, 3, 4, 5, 6, 6, 6, 6]);
        assert_eq!(p.row.as_slice::<u32>(), &[0, 0, 1, 1, 1, 1, 2]);
        // Padded layout: row start `r * len`; packed: the offsets. Padded marker slots point
        // at position 0 in both.
        let padded = MlxBackend::scorer_rows(&batch, &MlxBackend::row_starts(&batch, None));
        assert_eq!(padded.as_slice::<u32>(), &[0, 1, 0, 4, 5, 7, 8, 8, 8]);
        let packed = MlxBackend::scorer_rows(&batch, &MlxBackend::row_starts(&batch, Some(&p)));
        assert_eq!(packed.as_slice::<u32>(), &[0, 1, 0, 2, 3, 5, 6, 6, 6]);
    }

    #[test]
    fn settings_parse_every_kind_of_value() {
        let k = Knobs::parse("f16gelu,addmm=0,mask=bool,geglu=compiled,wcopy=t,buckets=512:128:256,pad=8,warm=1:4,cache=512,wired=2048").unwrap();
        assert!(k.f16gelu && !k.addmm && k.bool_mask);
        assert_eq!((k.geglu.as_str(), k.wcopy.as_str()), ("compiled", "t"));
        assert_eq!((k.buckets, k.pad, k.warm), (vec![128, 256, 512], 8, vec![1, 4]));
        assert_eq!((k.cache_mb, k.wired_mb), (Some(512), Some(2048)));
        // Empty spec and empty items are the defaults; the last mention wins.
        let d = Knobs::parse("").unwrap();
        assert!(!d.f16gelu && d.addmm && d.geglu == "shapeless" && d.cache_mb.is_none());
        assert!(Knobs::parse(",f16gelu,,").unwrap().f16gelu);
        assert!(!Knobs::parse("f16gelu,f16gelu=0").unwrap().f16gelu);
        assert!(!Knobs::parse("mask=bool,mask=additive").unwrap().bool_mask);
    }

    /// An unknown setting or a bad value is an error that names the item, never a silent default.
    #[test]
    fn settings_reject_unknown_keys_and_bad_values() {
        let err = |spec: &str| Knobs::parse(spec).err().map(|e| e.to_string()).unwrap_or_else(|| panic!("{spec} was accepted"));
        assert!(err("f16gelu,cache=512x").contains("`cache=512x`"), "{}", err("f16gelu,cache=512x"));
        assert!(err("f16gelu,chache=512").contains("unknown setting `chache`"), "{}", err("f16gelu,chache=512"));
        assert!(err("geglu=fast").contains("`fast` is not one of shapeless, compiled, plain"));
        assert!(err("mask=1").contains("`mask=1`"));
        assert!(err("wcopy=copy").contains("`wcopy=copy`"));
        assert!(err("f16gelu=yes").contains("`yes` is not 0 or 1"));
        assert!(err("buckets=128:abc").contains("`abc` is not a whole number"));
        assert!(err("buckets=").contains("`buckets=`"));
        assert!(err("pad=0").contains("pad must be at least 1"));
        assert!(err("pad=-1").contains("`-1` is not a whole number"));
        assert!(err("cache").contains("needs a value"));
        assert!(err("dense_upto=abc").contains("`abc` is not a whole number"));
        assert!(err("dense_upto").contains("needs a value"));
        assert!(err("unpad=yes").contains("`yes` is not 0 or 1"));
        assert!(err("headprune=on").contains("`on` is not 0 or 1"));
        assert!(err("fuserope=split").contains("`split` is not 0 or 1"));
        assert!(err("wired=2048,f16gelu,cache=512 ").contains("`cache=512 `"));
        assert_eq!(check_settings("f16gelu,cache=512,wired=2048").ok(), Some(()));
        assert!(check_settings("f16gelu,cache=512x").is_err());
    }

    /// Only the MLX variables the environment lacks are set; any user value is kept.
    #[test]
    fn mlx_env_defaults_keep_the_users_values() {
        use std::ffi::OsString;
        assert_eq!(mlx_env_unset(|_| None), vec![("MLX_MAX_MB_PER_BUFFER", "10")]);
        assert_eq!(mlx_env_unset(|_| Some(OsString::from("40"))), vec![]);
        assert_eq!(mlx_env_unset(|_| Some(OsString::new())), vec![]);
        let asked = std::cell::RefCell::new(Vec::new());
        mlx_env_unset(|key| {
            asked.borrow_mut().push(key.to_string());
            None
        });
        assert_eq!(asked.into_inner(), vec!["MLX_MAX_MB_PER_BUFFER"]);
        // MLX parses the value with atoi, so it must be a plain integer.
        for (_, value) in MLX_ENV_DEFAULTS {
            assert!(value.parse::<i32>().is_ok_and(|v| v > 0), "{value}");
        }
    }

    /// The per-shape cache admits `cap` distinct shapes, then only the shapes it already holds.
    #[test]
    fn shape_budget_caps_distinct_shapes() {
        let mut b = ShapeBudget::new(3);
        assert!(b.admit(&[1, 4]) && b.admit(&[2, 4]) && b.admit(&[3, 4]));
        assert_eq!(b.len(), 3);
        assert!(!b.admit(&[4, 4]), "a fourth shape is refused");
        assert!(b.admit(&[2, 4]), "a cached shape stays admitted");
        assert!(!b.admit(&[4, 4]), "the refused shape is not counted");
        assert_eq!(b.len(), 3);
        assert!(!ShapeBudget::new(0).admit(&[1]));
    }

    /// `geglu=compiled` compiles at most `cap` shapes; past the cap new shapes run the shapeless
    /// trace, and every path computes the same values as the plain (uncompiled) GeGLU.
    #[test]
    fn per_shape_geglu_falls_back_past_the_cap() {
        let g = GeGlu::with_cap("compiled", true, 2);
        let plain = GeGlu::new("plain", true);
        for rows in [1i32, 2, 3, 1, 4] {
            let x = Array::from_iter((0..rows * 8).map(|i| i as f32 * 0.25 - 4.0), &[rows, 8]);
            let (got, want) = (g.apply(&x).unwrap(), plain.apply(&x).unwrap());
            assert_eq!(got.shape(), &[rows, 4]);
            let d = ops::abs(&ops::subtract(&got, &want).unwrap()).unwrap().max(None).unwrap();
            assert!(d.item_exact::<f32>() < 1e-6, "rows={rows}: max diff {d}");
        }
        assert_eq!(g.per_shape_traces(), 2, "shapes [1,8] and [2,8] compiled per shape, [3,8] and [4,8] fell back");
        assert_eq!(GeGlu::new("shapeless", true).per_shape_traces(), 0);
    }

    /// The head attention scales by its own head dim, `hidden / max(1, hidden / 64)`: 1/8 for
    /// the published widths, and `1 / sqrt(hidden)` when a single head takes the whole width.
    #[test]
    fn head_scale_follows_the_head_dim() {
        assert_eq!(head_scale(768, 12), 0.125);
        assert_eq!(head_scale(1024, 16), 0.125);
        assert_eq!(head_scale(512, 8), 0.125);
        assert_eq!(head_scale(96, 1), 1.0 / 96f32.sqrt());
        assert_eq!(head_scale(160, 2), 1.0 / 80f32.sqrt());
    }

    /// One window mask serves every shorter length as a view with the same values.
    #[test]
    fn window_view_is_the_corner_of_the_longer_mask() {
        let full = window_mask_values(8, 2);
        let m = Array::from_slice(&full, &[1, 1, 8, 8]);
        assert_eq!(MlxBackend::window_view(&m, 8).as_slice::<f32>(), &full[..]);
        let corner = MlxBackend::window_view(&m, 5).contiguous().unwrap();
        assert_eq!(corner.shape(), &[1, 1, 5, 5]);
        assert_eq!(corner.as_slice::<f32>(), &window_mask_values(5, 2)[..]);
    }

    /// Reference values of the additive window mask for `len` and window `w`.
    fn window_mask_values(len: usize, w: usize) -> Vec<f32> {
        let mut vals = vec![0f32; len * len];
        for i in 0..len {
            for j in 0..len {
                if i.abs_diff(j) > w {
                    vals[i * len + j] = MASK_NEG;
                }
            }
        }
        vals
    }

    /// The device-built gather indices are the ones the host loop used to build and cache:
    /// `r * len + clamp((c - 1) * S + j, 0, len - 1)`, row-major over `(r, c, j)`.
    #[test]
    fn window_key_idx_matches_the_host_formula() {
        for (s, rows, len) in [(2usize, 3usize, 7usize), (4, 1, 4), (64, 2, 130)] {
            let nc = len.div_ceil(s);
            let want: Vec<u32> = (0..rows)
                .flat_map(|r| (0..nc).map(move |c| (r, c)))
                .flat_map(|(r, c)| (0..3 * s).map(move |j| (r, (c as i64 - 1) * s as i64 + j as i64)))
                .map(|(r, pos)| (r * len) as u32 + pos.clamp(0, len as i64 - 1) as u32)
                .collect();
            let got = window_key_idx(s, rows, len, nc).unwrap();
            assert_eq!(got.shape(), &[(rows * nc * 3 * s) as i32]);
            assert_eq!(got.as_slice::<u32>(), &want[..], "s={s} rows={rows} len={len}");
        }
    }

    /// Two rows, `S = 2`, `len = 7` (the last chunk has one real slot): row 0 is full, row 1
    /// has 3 tokens and 4 of padding.
    fn padded_batch() -> (Vec<u32>, usize, usize, usize, usize) {
        let mask = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0];
        (mask, 2, 7, 2, 4)
    }

    /// The chunked masks, boolean and additive, are the same rule: key slot `j` of chunk `c` is
    /// position `(c - 1) * S + j`, and a padded query is one past `len` or masked in its row.
    #[test]
    fn chunked_mask_tables_follow_the_position_formula() {
        let (mask, n, len, s, nc) = padded_batch();
        let key_ok = window_key_valid(&mask, n, len, s, nc);
        let padded = window_query_padded(&mask, n, len, s, nc);
        for b in 0..n {
            for c in 0..nc {
                for j in 0..3 * s {
                    let pos = (c as i64 - 1) * s as i64 + j as i64;
                    let want = (0..len as i64).contains(&pos) && mask[b * len + pos as usize] == 1;
                    assert_eq!(key_ok[(b * nc + c) * 3 * s + j], want, "row {b} chunk {c} slot {j}");
                }
                for i in 0..s {
                    let pos = c * s + i;
                    let want = pos >= len || mask[b * len + pos] != 1;
                    assert_eq!(padded[(b * nc + c) * s + i], want, "row {b} chunk {c} query {i}");
                }
            }
        }
        // The band is the same rule the additive mask has always used.
        let band = band_allows(s);
        let additive = band_mask(s, Dtype::Float32).unwrap();
        let additive = additive.as_slice::<f32>();
        for i in 0..s {
            for j in 0..3 * s {
                assert_eq!(band[i * 3 * s + j], i + 2 * s >= j && j >= i, "band {i},{j}");
                assert_eq!(additive[i * 3 * s + j] == 0.0, band[i * 3 * s + j], "additive band {i},{j}");
            }
        }
        assert_eq!(band_mask_bool(s).unwrap().as_slice::<bool>(), &band[..]);
    }

    /// `(key valid & band) | padded query` over the tables: a real query sees exactly the valid
    /// keys within `S` of it, and no query row is all false (which would be a NaN softmax row
    /// with boolean masks). Row 1's fourth chunk is all padding, so it relies on the `padded`
    /// term alone.
    #[test]
    fn chunked_bool_mask_leaves_no_query_row_all_false() {
        let (mask, n, len, s, nc) = padded_batch();
        let key_ok = window_key_valid(&mask, n, len, s, nc);
        let padded = window_query_padded(&mask, n, len, s, nc);
        let band = band_allows(s);
        for b in 0..n {
            for c in 0..nc {
                for i in 0..s {
                    let q = c * s + i;
                    let row: Vec<bool> = (0..3 * s)
                        .map(|j| (key_ok[(b * nc + c) * 3 * s + j] && band[i * 3 * s + j]) || padded[(b * nc + c) * s + i])
                        .collect();
                    assert!(row.iter().any(|&v| v), "row {b} query {q} attends to nothing");
                    if q < len && mask[b * len + q] == 1 {
                        for (j, &v) in row.iter().enumerate() {
                            let pos = (c as i64 - 1) * s as i64 + j as i64;
                            let want = (0..len as i64).contains(&pos)
                                && mask[b * len + pos as usize] == 1
                                && (pos - q as i64).abs() <= s as i64;
                            assert_eq!(v, want, "row {b} query {q} key slot {j} (position {pos})");
                        }
                    }
                }
            }
        }
        // The chunk that motivates the padded-query term.
        assert!(!key_ok[((nc + 3) * 3 * s)..((nc + 4) * 3 * s)].iter().any(|&v| v));
    }

    /// `pooled` is position 0 of every row, as a view (same values as a gather would give).
    #[test]
    fn scorer_rows_pooled_is_position_zero_of_every_row() {
        let (n, r, d) = (2, 3, 4);
        let vals: Vec<f32> = (0..(n * r * d)).map(|x| x as f32).collect();
        let rows = ScorerRows(Array::from_slice(&vals, &[n, r, d]));
        let pooled = rows.pooled().unwrap();
        assert_eq!(pooled.shape(), &[n, d]);
        let want: Vec<f32> = (0..n).flat_map(|b| (0..d).map(move |k| (b * r * d + k) as f32)).collect();
        assert_eq!(pooled.contiguous().unwrap().as_slice::<f32>(), &want[..]);
        let markers = rows.markers(n, r - 1, d).unwrap();
        assert_eq!(markers.shape(), &[n * (r - 1), d]);
    }
}
