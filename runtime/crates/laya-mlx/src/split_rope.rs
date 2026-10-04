//! `fuserope`: the encoder's qkv split, head reshape, RoPE of q and k, and the `unpad` expand
//! in one Metal kernel (`kernels/split_rope.metal`), in place of MLX's `split_equal`, three
//! reshape copies, two `fast::rope` launches and the `take_axis` gather. The kernel computes
//! what those ops compute, with MLX's own rope arithmetic, and [`mlx_path`] is that chain, kept
//! here as the reference the kernel is checked against: once at model load on small inputs
//! (every pipeline a request can hit), and in this module's tests at several shapes.
//!
//! Instantiations are bounded: the template arguments are the element type, the head count,
//! the head dim, whether the batch is packed, the output layout (plain, chunked or padded) and
//! the local window, all fixed for a loaded model except the layout. MLX also picks the address
//! space of each input by its element count (fewer than 8 elements: `constant`, otherwise
//! `device`; `max_constant_array_size` in `mlx/backend/common/metal_kernel.cpp` of MLX 0.32.2)
//! and names the pipeline after that choice. The qkv rows, `dims` and `lbase` never cross that
//! limit; the packing's `unpack` index does. So a model has the 3 pipelines of [`Variant`], 9
//! with the two banded layouts of `band`, and the load-time check runs every one. The batch
//! shape travels in the `dims` input, never in a template argument.
//!
//! The banded layout ([`SplitRope::apply_band`]) is the chunked form the `band` local
//! attention consumes: queries by chunks of `S` positions and, for every chunk, the `3S` key
//! and value positions its window can reach, zero outside the sequence. [`mlx_path_band`] is
//! that layout built with MLX pad, slice and stack ops from [`mlx_path`], the reference the
//! banded launch is checked against. The padded layout (`apply_band` with `padded`) holds the
//! same positions once, in position order with `S` zero rows before the keys, so that the
//! chunk windows are overlapping views of it ([`band_views`], one row only); [`mlx_path_padded`]
//! is its reference. A kernel whose banded launches fail the load check while its plain ones
//! pass is kept, with the banded launches off.

use crate::metal_kernels::{MetalKernel, TemplateArg};
use crate::Lx;
use laya_core::{Error, Result};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{fast, ops, transforms, Array, Dtype};

const SOURCE: &str = include_str!("kernels/split_rope.metal");
/// Pairs of elements each thread rotates; the kernel loads and stores `vec<T, 4>`.
const PAIRS_PER_THREAD: i32 = 4;
const THREADGROUP: i32 = 256;
/// MLX passes an input with fewer elements than this in the `constant` address space, as a
/// pipeline of its own (`max_constant_array_size`, `mlx/backend/common/metal_kernel.cpp`).
const MLX_CONSTANT_LIMIT: usize = 8;

/// The MLX chain the kernel replaces: `qkv [n, len, 3 * heads * hd]` -> roped `q`, roped `k`
/// and `v`, each `[n, heads, len, hd]`.
pub(crate) fn mlx_path(qkv: &Array, n: i32, len: i32, heads: i32, hd: i32, theta: f32) -> Result<(Array, Array, Array)> {
    let heads_of = |x: &Array| -> Result<Array> {
        x.reshape(&[n, len, heads, hd]).lx()?.transpose_axes(&[0, 2, 1, 3]).lx()
    };
    let rope = |x: &Array| -> Result<Array> { fast::rope(x, hd, false, theta, 1.0, 0, None::<&Array>).lx() };
    let parts = qkv.split_equal(3, -1).lx()?;
    let q = heads_of(&parts[0])?;
    let k = heads_of(&parts[1])?;
    let v = heads_of(&parts[2])?;
    Ok((rope(&q)?, rope(&k)?, v))
}

/// The banded layout from the MLX chain: `qkv [n, len, 3 * heads * hd]` -> roped `q`
/// `[n * nc, heads, s, hd]` (chunk `c` of row `b` at index `b * nc + c` holds positions
/// `c * s ..`, zero past `len`), roped `k` and `v` `[n * nc, heads, 3 * s, hd]` (slot `j` of
/// chunk `c` holds position `(c - 1) * s + j`, zero outside `0..len`), with `nc = ceil(len / s)`.
pub(crate) fn mlx_path_band(qkv: &Array, n: i32, len: i32, heads: i32, hd: i32, theta: f32, s: i32) -> Result<(Array, Array, Array)> {
    let (q, k, v) = mlx_path(qkv, n, len, heads, hd, theta)?;
    let nc = chunks(len, s);
    let lp = nc * s;
    // `[n, heads, len, hd]`, padded by `lo` positions before and `hi` after, cut into `nc`
    // windows of `width` positions starting at `c * s`, as `[n * nc, heads, width, hd]`.
    let chunks = |x: &Array, lo: i32, hi: i32, width: i32| -> Result<Array> {
        let x = ops::pad(x, &[(0, 0), (0, 0), (lo, hi), (0, 0)], None::<Array>, None::<ops::PadMode>).lx()?;
        let parts: Vec<Array> = (0..nc).map(|c| x.index((.., .., c * s..c * s + width, ..))).collect();
        let x = ops::stack(&parts, 2).lx()?;
        x.transpose_axes(&[0, 2, 1, 3, 4]).lx()?.reshape(&[n * nc, heads, width, hd]).lx()
    };
    let q = chunks(&q, 0, lp - len, s)?;
    let k = chunks(&k, s, lp + s - len, 3 * s)?;
    let v = chunks(&v, s, lp + s - len, 3 * s)?;
    Ok((q, k, v))
}

/// The padded layout from the MLX chain: `qkv [n, len, 3 * heads * hd]` -> roped `q`
/// `[n, heads, nc * s, hd]` (position `p` at row `p`, zero past `len`), roped `k` and `v`
/// `[n, heads, (nc + 2) * s, hd]` (position `p` at row `p + s`, zero outside `0..len`).
pub(crate) fn mlx_path_padded(qkv: &Array, n: i32, len: i32, heads: i32, hd: i32, theta: f32, s: i32) -> Result<(Array, Array, Array)> {
    let (q, k, v) = mlx_path(qkv, n, len, heads, hd, theta)?;
    let nc = chunks(len, s);
    let lp = nc * s;
    let pad = |x: &Array, lo: i32, hi: i32| -> Result<Array> {
        ops::pad(x, &[(0, 0), (0, 0), (lo, hi), (0, 0)], None::<Array>, None::<ops::PadMode>).lx()
    };
    Ok((pad(&q, 0, lp - len)?, pad(&k, s, lp + s - len)?, pad(&v, s, lp + s - len)?))
}

/// The chunk windows of the padded layout of one row (`n == 1`), as views: `q [1, heads,
/// nc * s, hd]` -> `[nc, heads, s, hd]`, `k` and `v [1, heads, (nc + 2) * s, hd]` ->
/// `[nc, heads, 3s, hd]` with chunk `c` starting at row `c * s` (windows overlap by `2s`
/// rows). Element for element these are [`mlx_path_band`]'s arrays; no copy is made, and
/// `scaled_dot_product_attention` reads them through their strides.
pub(crate) fn band_views(q: &Array, k: &Array, v: &Array, nc: i32, s: i32, heads: i32, hd: i32) -> Result<(Array, Array, Array)> {
    if q.shape() != [1, heads, nc * s, hd] || k.shape() != [1, heads, (nc + 2) * s, hd] || v.shape() != k.shape() {
        return Err(Error::Backend(format!("band_views: one row expected, got q {:?} k {:?} v {:?}", q.shape(), k.shape(), v.shape())));
    }
    let (s64, hd64, nc64) = (s as i64, hd as i64, nc as i64);
    let qv = q.as_strided(&[nc, heads, s, hd][..], &[s64 * hd64, nc64 * s64 * hd64, hd64, 1][..], 0).lx()?;
    let kstrides = [s64 * hd64, (nc64 + 2) * s64 * hd64, hd64, 1];
    let kv = k.as_strided(&[nc, heads, 3 * s, hd][..], &kstrides[..], 0).lx()?;
    let vv = v.as_strided(&[nc, heads, 3 * s, hd][..], &kstrides[..], 0).lx()?;
    Ok((qv, kv, vv))
}

/// `ceil(len / s)`, the chunk count of the banded layout.
pub(crate) fn chunks(len: i32, s: i32) -> i32 {
    (len + s - 1) / s
}

/// `[log2(theta)]` as the kernel's `lbase` input, computed as MLX's rope does it (`log2` of the
/// base as an f32).
fn log2_base(theta: f32) -> Array {
    Array::from_slice(&[theta.log2()], &[1])
}

/// The pipelines one model can launch. Which one a request hits depends on the batch: the
/// padded layout, or the packed layout with its `unpack` index below or at MLX's `constant`
/// limit. The load-time check runs every one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Variant {
    /// No packing: the kernel reads the padded rows in place.
    Unpacked,
    /// Packed, and `unpack` has fewer than [`MLX_CONSTANT_LIMIT`] elements.
    PackedConstant,
    /// Packed, and `unpack` has at least [`MLX_CONSTANT_LIMIT`] elements.
    PackedDevice,
}

impl Variant {
    const ALL: [Variant; 3] = [Variant::Unpacked, Variant::PackedConstant, Variant::PackedDevice];

    /// The pipeline a launch with this `unpack` input hits, by MLX's rule.
    fn of(unpack: Option<&Array>) -> Variant {
        match unpack {
            None => Variant::Unpacked,
            Some(u) if u.size() < MLX_CONSTANT_LIMIT => Variant::PackedConstant,
            Some(_) => Variant::PackedDevice,
        }
    }
}

/// `pack` (padded index of every real token) and `unpack` (packed index of every padded
/// position, padding borrowing its row's token 0), as `Packing` builds them, for `n` rows of
/// `len` with the given real lengths.
fn packing(n: i32, len: i32, lens: &[i32]) -> (Vec<u32>, Vec<u32>) {
    assert_eq!(lens.len(), n as usize);
    let (mut pack, mut unpack) = (Vec::new(), Vec::new());
    for (r, &l) in lens.iter().enumerate() {
        let first = pack.len() as u32;
        for pos in 0..len {
            if pos < l {
                unpack.push(pack.len() as u32);
                pack.push((r as i32 * len + pos) as u32);
            } else {
                unpack.push(first);
            }
        }
    }
    (pack, unpack)
}

/// One batch of the load-time check: its pipeline, its shape, the real length of every row
/// (ignored for the padded layout) and which rope base to use.
struct CheckCase {
    variant: Variant,
    n: i32,
    len: i32,
    lens: &'static [i32],
    local: bool,
}

impl CheckCase {
    /// The `unpack` index of this case, `None` for the padded layout.
    fn index(&self) -> Option<Array> {
        match self.variant {
            Variant::Unpacked => None,
            _ => {
                let (_, unpack) = packing(self.n, self.len, self.lens);
                Some(Array::from_slice(&unpack, &[unpack.len() as i32]))
            }
        }
    }
}

/// The batches the load-time check runs, at least one per [`Variant`]: 6 padded positions;
/// one padded row of 70 positions, on which the banded check also compares the chunk views of
/// the padded layout; 6 packed positions with 5 real tokens (an index of 6 elements,
/// `constant`); 10 packed positions with 8 real tokens (an index of 10 elements, `device`);
/// and 266 packed positions with 203 real tokens. The two long cases span 2 and 3 chunks of
/// the banded layout at the published models' window of 64 (more at a smaller one).
/// `load_check_covers_every_kernel_variant` checks that each case hits the pipeline it is
/// listed for. With `band` every case also runs the chunked and the padded launch.
const CHECK_CASES: [CheckCase; 5] = [
    CheckCase { variant: Variant::Unpacked, n: 2, len: 3, lens: &[3, 3], local: false },
    CheckCase { variant: Variant::Unpacked, n: 1, len: 70, lens: &[70], local: true },
    CheckCase { variant: Variant::PackedConstant, n: 2, len: 3, lens: &[3, 2], local: true },
    CheckCase { variant: Variant::PackedDevice, n: 2, len: 5, lens: &[5, 3], local: false },
    CheckCase { variant: Variant::PackedDevice, n: 2, len: 133, lens: &[133, 70], local: true },
];

/// The fused kernel for one model's head layout and rope bases.
pub(crate) struct SplitRope {
    kernel: MetalKernel,
    heads: i32,
    hd: i32,
    /// The local window `S` of the banded layout, the kernel's `S` template argument (fixed per
    /// model, passed on every launch so that one set of pipelines serves both layouts).
    window: i32,
    /// Whether the banded launches passed their check at load and may be used.
    band: bool,
    /// Placeholder for the `unpack` input when the batch is not packed; the kernel never reads
    /// it (`PACKED` is false), and MLX passes it in the `constant` address space.
    no_index: Array,
    /// The global and the local rope base, and their `lbase` inputs, indexed by the layer's
    /// `local` flag.
    theta: [f32; 2],
    lbase: [Array; 2],
}

impl SplitRope {
    /// Build the kernel and run every pipeline of [`Variant`] once on a small batch against
    /// [`mlx_path`], so that a kernel that does not compile, or does not reproduce the MLX ops
    /// on this MLX build, is an `Err` here at load and never inside a request. The run also
    /// compiles the pipelines before the first request. `hd` must be a multiple of 8 (the
    /// kernel's vector width times two halves). `window` is the local window `S` of the banded
    /// layout; with `band` the banded launches are checked too (and refused otherwise). If
    /// only they fail, the kernel is returned with them off, and with the reason.
    pub(crate) fn new(heads: usize, hd: usize, dtype: Dtype, global_theta: f32, local_theta: f32, window: usize, band: bool) -> Result<(Self, Option<Error>)> {
        Self::build(heads, hd, global_theta, local_theta, window, band)?.checked(dtype)
    }

    /// The kernel, unchecked.
    fn build(heads: usize, hd: usize, global_theta: f32, local_theta: f32, window: usize, band: bool) -> Result<Self> {
        if hd == 0 || !hd.is_multiple_of(2 * PAIRS_PER_THREAD as usize) {
            return Err(Error::Config(format!("fuserope needs a head dim that is a multiple of 8, this model has {hd}")));
        }
        if window == 0 {
            return Err(Error::Config("fuserope needs a local window of at least 1".into()));
        }
        let kernel = MetalKernel::new("sys1_split_rope", &["qkv", "unpack", "dims", "lbase"], &["q", "k", "v"], SOURCE).lx()?;
        let lbase = [log2_base(global_theta), log2_base(local_theta)];
        transforms::eval([&lbase[0], &lbase[1]]).lx()?;
        Ok(Self {
            kernel,
            heads: heads as i32,
            hd: hd as i32,
            window: window as i32,
            band,
            no_index: Array::from_slice(&[0u32], &[1]),
            theta: [global_theta, local_theta],
            lbase,
        })
    }

    /// The load check ([`Self::self_check`]): `Err` if a plain launch fails it; else the
    /// kernel, with `band` off and the reason if a banded launch failed it.
    fn checked(mut self, dtype: Dtype) -> Result<(Self, Option<Error>)> {
        let band_off = self.self_check(dtype)?;
        if band_off.is_some() {
            self.band = false;
        }
        Ok((self, band_off))
    }

    /// Test hook: [`Self::new`] with a kernel whose banded launches write 7 to `k[0]`, a slot
    /// outside the sequence, and whose plain launch is the real one. In both banded layouts
    /// thread 0 is the only thread that writes `k[0]`, so the extra write is not a race.
    #[cfg(test)]
    pub(crate) fn with_wrong_band(heads: usize, hd: usize, dtype: Dtype, global_theta: f32, local_theta: f32, window: usize, band: bool) -> Result<(Self, Option<Error>)> {
        let mut this = Self::build(heads, hd, global_theta, local_theta, window, band)?;
        let wrong = format!("{SOURCE}\nif (BAND != 0 && g == 0) k[0] = T(7);\n");
        this.kernel = MetalKernel::new("sys1_split_rope_wrong_band", &["qkv", "unpack", "dims", "lbase"], &["q", "k", "v"], &wrong).lx()?;
        this.checked(dtype)
    }

    /// Whether the banded launches passed their check.
    #[cfg(test)]
    pub(crate) fn band_checked(&self) -> bool {
        self.band
    }

    /// Every case of [`CHECK_CASES`] in `dtype`, each compared with [`mlx_path`] bit for bit
    /// ([`crate::same_bits`], so a -0 does not pass for a 0), and each checked to hit the
    /// pipeline it is listed for. A plain launch that fails is an `Err`. With `band`, a banded
    /// launch that fails (or cannot run) is `Ok(Some(reason))`, and the cases after it check
    /// the plain launch only.
    fn self_check(&self, dtype: Dtype) -> Result<Option<Error>> {
        for v in Variant::ALL {
            if !CHECK_CASES.iter().any(|c| c.variant == v) {
                return Err(Error::Backend(format!("fuserope self-check: no case for the {v:?} launch")));
            }
        }
        let width = 3 * self.heads * self.hd;
        let mut band_off = None;
        for case in &CHECK_CASES {
            let (n, len) = (case.n, case.len);
            let vals: Vec<f32> = (0..n * len * width).map(|i| ((i64::from(i) * 7919) % 2003) as f32 / 1001.0 - 1.0).collect();
            let padded = Array::from_slice(&vals, &[n, len, width]).as_dtype(dtype).lx()?;
            let unpack = case.index();
            if Variant::of(unpack.as_ref()) != case.variant {
                return Err(Error::Backend(format!("fuserope self-check: the {:?} case builds a {:?} launch", case.variant, Variant::of(unpack.as_ref()))));
            }
            // The rows the kernel reads and the padded layout the MLX ops read.
            let (rows, expanded) = match &unpack {
                None => (padded.clone(), padded),
                Some(unpack) => {
                    let (pack, _) = packing(n, len, case.lens);
                    let rows = padded.reshape(&[n * len, width]).lx()?.take_axis(Array::from_slice(&pack, &[pack.len() as i32]), 0).lx()?;
                    let expanded = rows.take_axis(unpack, 0).lx()?.reshape(&[n, len, width]).lx()?;
                    (rows, expanded)
                }
            };
            let nc = chunks(len, self.window);
            let dims = Array::from_slice(&[n, len, nc], &[3]);
            let theta = self.theta[case.local as usize];
            let same = |name: &str, want: &Array, have: &Array, launch: &str| -> Result<()> {
                if !crate::same_bits(want, have)? {
                    return Err(Error::Backend(format!("fuserope self-check: the kernel's {name} differs from the MLX ops ({:?} {launch} launch)", case.variant)));
                }
                Ok(())
            };
            let (q, k, v) = mlx_path(&expanded, n, len, self.heads, self.hd, theta)?;
            let got = self.apply(&rows, unpack.as_ref(), &dims, case.local, n, len)?;
            for (name, want, have) in [("q", &q, &got.0), ("k", &k, &got.1), ("v", &v, &got.2)] {
                same(name, want, have, "plain")?;
            }
            let banded = || -> Result<()> {
                let (q, k, v) = mlx_path_band(&expanded, n, len, self.heads, self.hd, theta, self.window)?;
                let got = self.apply_band(&rows, unpack.as_ref(), &dims, case.local, n, nc, false)?;
                for (name, want, have) in [("q", &q, &got.0), ("k", &k, &got.1), ("v", &v, &got.2)] {
                    same(name, want, have, "banded")?;
                }
                let (pq, pk, pv) = mlx_path_padded(&expanded, n, len, self.heads, self.hd, theta, self.window)?;
                let got = self.apply_band(&rows, unpack.as_ref(), &dims, case.local, n, nc, true)?;
                for (name, want, have) in [("q", &pq, &got.0), ("k", &pk, &got.1), ("v", &pv, &got.2)] {
                    same(name, want, have, "padded")?;
                }
                if n == 1 {
                    let views = band_views(&got.0, &got.1, &got.2, nc, self.window, self.heads, self.hd)?;
                    for (name, want, have) in [("q", &q, &views.0), ("k", &k, &views.1), ("v", &v, &views.2)] {
                        same(name, want, have, "padded, viewed by chunks")?;
                    }
                }
                Ok(())
            };
            if self.band && band_off.is_none() {
                band_off = banded().err();
            }
        }
        Ok(band_off)
    }

    /// `qkv` as `[R, 3 * heads * hd]` rows (or `[n, len, 3 * heads * hd]`, reshaped for free)
    /// -> roped `q`, roped `k` and `v`, each `[n, heads, len, hd]`. `unpack` is the packing's
    /// `[n * len]` index when the rows are packed, `dims` is `[n, len, nc]` i32 (`nc` is read
    /// by the banded launch only), and `local` picks the layer's rope base.
    pub(crate) fn apply(&self, qkv: &Array, unpack: Option<&Array>, dims: &Array, local: bool, n: i32, len: i32) -> Result<(Array, Array, Array)> {
        let shape = [n, self.heads, len, self.hd];
        let threads = n * len * self.heads * (self.hd / 2 / PAIRS_PER_THREAD);
        self.launch(qkv, unpack, dims, local, 0, &shape, &shape, threads)
    }

    /// The same inputs -> the banded layout of [`mlx_path_band`] for `nc = ceil(len / S)`
    /// chunks (`dims` must carry that `nc`): roped `q` `[n * nc, heads, S, hd]`, roped `k` and
    /// `v` `[n * nc, heads, 3S, hd]`; or, with `padded`, the padded layout of
    /// [`mlx_path_padded`]: `q` `[n, heads, nc * S, hd]`, `k` and `v` `[n, heads, (nc + 2) * S,
    /// hd]`, every position written once. Only after `new` checked the banded launches.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_band(&self, qkv: &Array, unpack: Option<&Array>, dims: &Array, local: bool, n: i32, nc: i32, padded: bool) -> Result<(Array, Array, Array)> {
        if !self.band {
            return Err(Error::Backend("fuserope: the banded layout was not checked at load".into()));
        }
        let s = self.window;
        let threads = n * (nc + 2) * s * self.heads * (self.hd / 2 / PAIRS_PER_THREAD);
        if padded {
            let qshape = [n, self.heads, nc * s, self.hd];
            let kshape = [n, self.heads, (nc + 2) * s, self.hd];
            return self.launch(qkv, unpack, dims, local, 2, &qshape, &kshape, threads);
        }
        let qshape = [n * nc, self.heads, s, self.hd];
        let kshape = [n * nc, self.heads, 3 * s, self.hd];
        self.launch(qkv, unpack, dims, local, 1, &qshape, &kshape, threads)
    }

    /// One launch: `band` picks the layout (0 plain, 1 chunked, 2 padded), `qshape` the shape
    /// of `q`, `kshape` that of `k` and `v`, `threads` the grid.
    #[allow(clippy::too_many_arguments)]
    fn launch(&self, qkv: &Array, unpack: Option<&Array>, dims: &Array, local: bool, band: i32, qshape: &[i32], kshape: &[i32], threads: i32) -> Result<(Array, Array, Array)> {
        let width = 3 * self.heads * self.hd;
        let rows = qkv.reshape(&[-1, width]).lx()?;
        let dtype = rows.dtype();
        let mut out = self
            .kernel
            .apply(
                &[&rows, unpack.unwrap_or(&self.no_index), dims, &self.lbase[local as usize]],
                &[(qshape, dtype), (kshape, dtype), (kshape, dtype)],
                &[
                    ("T", TemplateArg::Dtype(dtype)),
                    ("H", TemplateArg::Int(self.heads)),
                    ("D", TemplateArg::Int(self.hd)),
                    ("PACKED", TemplateArg::Bool(unpack.is_some())),
                    ("BAND", TemplateArg::Int(band)),
                    ("S", TemplateArg::Int(self.window)),
                ],
                [threads, 1, 1],
                [THREADGROUP, 1, 1],
            )
            .lx()?;
        let v = out.pop();
        let k = out.pop();
        let q = out.pop();
        match (q, k, v) {
            (Some(q), Some(k), Some(v)) => Ok((q, k, v)),
            _ => Err(Error::Backend("fuserope: the kernel returned fewer than 3 outputs".into())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::indexing::IndexOp;

    const THETA: f32 = 10_000.0;

    /// Deterministic values in about `[-2, 2]` with the spread of a real projection.
    fn values(count: usize) -> Vec<f32> {
        let mut x: u32 = 12345;
        (0..count)
            .map(|_| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (x >> 8) as f32 / (1u32 << 24) as f32 * 4.0 - 2.0
            })
            .collect()
    }

    /// Bit equality of two arrays, as the load check compares ([`crate::same_bits`]): a -0
    /// does not pass for a 0. A mismatch names the first element that differs (through f32,
    /// lossless for f16, so the sign of a zero shows).
    fn assert_same(what: &str, want: &Array, got: &Array) {
        assert_eq!(want.shape(), got.shape(), "{what}: shape");
        assert_eq!(want.dtype(), got.dtype(), "{what}: dtype");
        if crate::same_bits(want, got).unwrap() {
            return;
        }
        let w = want.as_dtype(Dtype::Float32).unwrap().contiguous().unwrap();
        let g = got.as_dtype(Dtype::Float32).unwrap().contiguous().unwrap();
        let (w, g) = (w.as_slice::<f32>(), g.as_slice::<f32>());
        let at = w.iter().zip(g).position(|(a, b)| a.to_bits() != b.to_bits());
        panic!("{what}: bits differ, first at element {at:?}: {:?} against {:?}", at.map(|i| w[i]), at.map(|i| g[i]));
    }

    /// The comparator the tests share with the load check fails on a signed zero, where the
    /// old `==` and largest-difference check passed it.
    #[test]
    #[should_panic(expected = "bits differ, first at element Some(1)")]
    fn assert_same_fails_on_a_signed_zero() {
        let pos = Array::from_slice(&[1.0f32, 0.0, 2.0], &[3]).as_dtype(Dtype::Float16).unwrap();
        let neg = Array::from_slice(&[1.0f32, -0.0, 2.0], &[3]).as_dtype(Dtype::Float16).unwrap();
        assert_same("signed zero", &pos, &neg);
    }

    /// A batch of `n` rows padded to `len`, with the given real lengths: the packed rows, the
    /// packing's `unpack` index and the expanded padded layout the MLX reference reads.
    fn batch(n: i32, len: i32, lens: &[i32], width: i32, dtype: Dtype) -> (Array, Array, Array) {
        let padded = Array::from_slice(&values((n * len * width) as usize), &[n, len, width]).as_dtype(dtype).unwrap();
        let (pack, unpack) = packing(n, len, lens);
        let rows = padded.reshape(&[n * len, width]).unwrap();
        let packed = rows.take_axis(Array::from_slice(&pack, &[pack.len() as i32]), 0).unwrap();
        let unpack = Array::from_slice(&unpack, &[unpack.len() as i32]);
        let expanded = packed.take_axis(&unpack, 0).unwrap().reshape(&[n, len, width]).unwrap();
        (packed, unpack, expanded)
    }

    /// The kernel against the MLX chain, padded and packed, at several shapes: one row and
    /// many, lengths that are not multiples of anything, the model's 16 x 64 heads and two
    /// smaller head dims, both element types, a packed index of fewer than 8 elements (MLX's
    /// `constant` variant) and the longest position the published models allow (8,192).
    #[test]
    fn kernel_matches_the_mlx_ops() {
        // (n, len, real lengths, heads, head dim, dtype)
        let cases: Vec<(i32, i32, Vec<i32>, i32, i32, Dtype)> = vec![
            (1, 7, vec![7], 4, 64, Dtype::Float16),
            (3, 13, vec![13, 5, 1], 4, 64, Dtype::Float16),
            (2, 3, vec![3, 2], 2, 8, Dtype::Float16),
            (2, 5, vec![4, 5], 3, 32, Dtype::Float32),
            (10, 226, vec![226, 197, 150, 226, 64, 3, 99, 226, 180, 17], 16, 64, Dtype::Float16),
            (1, 8192, vec![8192], 16, 64, Dtype::Float16),
            (2, 8192, vec![8192, 4097], 2, 64, Dtype::Float16),
        ];
        let mut kernels: Vec<((i32, i32, Dtype), SplitRope)> = Vec::new();
        let mut seen = Vec::new();
        for (n, len, lens, heads, hd, dtype) in cases {
            let what = format!("n={n} len={len} lens={lens:?} heads={heads} hd={hd} {dtype:?}");
            let key = (heads, hd, dtype);
            if !kernels.iter().any(|(k, _)| *k == key) {
                kernels.push((key, SplitRope::new(heads as usize, hd as usize, dtype, THETA, THETA, 64, false).unwrap().0));
            }
            let kernel = &kernels.iter().find(|(k, _)| *k == key).unwrap().1;
            let width = 3 * heads * hd;
            let (packed, unpack, expanded) = batch(n, len, &lens, width, dtype);
            let dims = Array::from_slice(&[n, len, 1], &[3]);
            // Padded layout: the expanded rows straight through, as a forward without `unpad`.
            let (q, k, v) = mlx_path(&expanded, n, len, heads, hd, THETA).unwrap();
            let (gq, gk, gv) = kernel.apply(&expanded, None, &dims, false, n, len).unwrap();
            assert_same(&format!("{what} padded q"), &q, &gq);
            assert_same(&format!("{what} padded k"), &k, &gk);
            assert_same(&format!("{what} padded v"), &v, &gv);
            seen.push(Variant::Unpacked);
            // Packed rows with the unpack index: the `unpad` forward.
            if packed.dim(0) < n * len {
                let (gq, gk, gv) = kernel.apply(&packed, Some(&unpack), &dims, false, n, len).unwrap();
                assert_same(&format!("{what} packed q"), &q, &gq);
                assert_same(&format!("{what} packed k"), &k, &gk);
                assert_same(&format!("{what} packed v"), &v, &gv);
                seen.push(Variant::of(Some(&unpack)));
            }
        }
        for v in Variant::ALL {
            assert!(seen.contains(&v), "no case launched {v:?}");
        }
    }

    /// Each load-time case hits the pipeline it is listed for under MLX's rule, and every
    /// pipeline is listed, so the check cannot silently stop covering one.
    #[test]
    fn load_check_covers_every_kernel_variant() {
        for v in Variant::ALL {
            let case = CHECK_CASES.iter().find(|c| c.variant == v).unwrap_or_else(|| panic!("{v:?} is not in the load check"));
            assert_eq!(Variant::of(case.index().as_ref()), v, "the {v:?} case builds another launch");
        }
        assert_eq!(Variant::of(Some(&Array::from_slice(&[0u32; 7], &[7]))), Variant::PackedConstant);
        assert_eq!(Variant::of(Some(&Array::from_slice(&[0u32; 8], &[8]))), Variant::PackedDevice);
    }

    /// The two rope bases of the published models, at a late position, stay exact, each
    /// through its own `lbase` input.
    #[test]
    fn both_rope_bases_match() {
        let (n, len, heads, hd) = (1, 1025, 2, 64);
        let kernel = SplitRope::new(heads as usize, hd as usize, Dtype::Float16, 10_000.0, 160_000.0, 64, false).unwrap().0;
        let width = 3 * heads * hd;
        let (_, _, expanded) = batch(n, len, &[len], width, Dtype::Float16);
        let dims = Array::from_slice(&[n, len, 1], &[3]);
        for (theta, local) in [(10_000.0f32, false), (160_000.0, true)] {
            let (q, k, _) = mlx_path(&expanded, n, len, heads, hd, theta).unwrap();
            let (gq, gk, _) = kernel.apply(&expanded, None, &dims, local, n, len).unwrap();
            assert_same(&format!("theta={theta} q"), &q, &gq);
            assert_same(&format!("theta={theta} k"), &k, &gk);
        }
    }

    /// The banded launch against the MLX pad, slice and stack chain, padded and packed, at
    /// windows that divide the length and ones that do not, one chunk and many, both element
    /// types; and the plain launch of the same kernel build still matches.
    #[test]
    fn band_kernel_matches_the_mlx_ops() {
        // (n, len, real lengths, heads, head dim, dtype, window)
        type BandCase = (i32, i32, Vec<i32>, i32, i32, Dtype, i32);
        let cases: Vec<BandCase> = vec![
            (1, 7, vec![7], 4, 64, Dtype::Float16, 4),
            (1, 8, vec![8], 2, 8, Dtype::Float16, 4),
            (3, 13, vec![13, 5, 1], 4, 64, Dtype::Float16, 4),
            (2, 5, vec![4, 5], 3, 32, Dtype::Float32, 64),
            (1, 151, vec![151], 16, 64, Dtype::Float16, 64),
            (4, 226, vec![226, 197, 64, 3], 16, 64, Dtype::Float16, 64),
            (1, 640, vec![640], 2, 64, Dtype::Float16, 64),
        ];
        for (n, len, lens, heads, hd, dtype, s) in cases {
            let what = format!("band n={n} len={len} lens={lens:?} heads={heads} hd={hd} {dtype:?} s={s}");
            let (kernel, band_off) = SplitRope::new(heads as usize, hd as usize, dtype, THETA, THETA, s as usize, true).unwrap();
            assert!(band_off.is_none(), "{what}: {band_off:?}");
            let nc = chunks(len, s);
            let width = 3 * heads * hd;
            let (packed, unpack, expanded) = batch(n, len, &lens, width, dtype);
            let dims = Array::from_slice(&[n, len, nc], &[3]);
            let (q, k, v) = mlx_path_band(&expanded, n, len, heads, hd, THETA, s).unwrap();
            assert_eq!(q.shape(), &[n * nc, heads, s, hd]);
            assert_eq!(k.shape(), &[n * nc, heads, 3 * s, hd]);
            let (gq, gk, gv) = kernel.apply_band(&expanded, None, &dims, false, n, nc, false).unwrap();
            assert_same(&format!("{what} unpacked q"), &q, &gq);
            assert_same(&format!("{what} unpacked k"), &k, &gk);
            assert_same(&format!("{what} unpacked v"), &v, &gv);
            // The padded layout against its reference, and (one row) its chunk views against
            // the chunked reference above.
            let (pq, pk, pv) = mlx_path_padded(&expanded, n, len, heads, hd, THETA, s).unwrap();
            assert_eq!(pq.shape(), &[n, heads, nc * s, hd]);
            assert_eq!(pk.shape(), &[n, heads, (nc + 2) * s, hd]);
            let (gq, gk, gv) = kernel.apply_band(&expanded, None, &dims, false, n, nc, true).unwrap();
            assert_same(&format!("{what} padded q"), &pq, &gq);
            assert_same(&format!("{what} padded k"), &pk, &gk);
            assert_same(&format!("{what} padded v"), &pv, &gv);
            if n == 1 {
                let (vq, vk, vv) = band_views(&gq, &gk, &gv, nc, s, heads, hd).unwrap();
                assert_same(&format!("{what} viewed q"), &q, &vq);
                assert_same(&format!("{what} viewed k"), &k, &vk);
                assert_same(&format!("{what} viewed v"), &v, &vv);
            } else {
                assert!(band_views(&gq, &gk, &gv, nc, s, heads, hd).is_err(), "{what}: views of several rows");
            }
            if packed.dim(0) < n * len {
                let (gq, gk, gv) = kernel.apply_band(&packed, Some(&unpack), &dims, false, n, nc, false).unwrap();
                assert_same(&format!("{what} packed q"), &q, &gq);
                assert_same(&format!("{what} packed k"), &k, &gk);
                assert_same(&format!("{what} packed v"), &v, &gv);
                let (gq, gk, gv) = kernel.apply_band(&packed, Some(&unpack), &dims, false, n, nc, true).unwrap();
                assert_same(&format!("{what} packed padded q"), &pq, &gq);
                assert_same(&format!("{what} packed padded k"), &pk, &gk);
                assert_same(&format!("{what} packed padded v"), &pv, &gv);
            }
            let (q, k, v) = mlx_path(&expanded, n, len, heads, hd, THETA).unwrap();
            let (gq, gk, gv) = kernel.apply(&expanded, None, &dims, false, n, len).unwrap();
            assert_same(&format!("{what} plain q"), &q, &gq);
            assert_same(&format!("{what} plain k"), &k, &gk);
            assert_same(&format!("{what} plain v"), &v, &gv);
        }
    }

    /// The banded reference itself: chunk `c` of `q` is positions `c * s ..`, slot `j` of chunk
    /// `c` of `k` is position `(c - 1) * s + j`, and the slots outside the sequence are zero.
    #[test]
    fn mlx_path_band_is_the_chunked_view() {
        let (n, len, heads, hd, s) = (2, 10, 2, 8, 4);
        let x = Array::from_slice(&values((n * len * 3 * heads * hd) as usize), &[n, len, 3 * heads * hd]);
        let (q, k, v) = mlx_path(&x, n, len, heads, hd, THETA).unwrap();
        let (bq, bk, bv) = mlx_path_band(&x, n, len, heads, hd, THETA, s).unwrap();
        let nc = chunks(len, s);
        assert_eq!(nc, 3);
        let zeros = |a: &Array| a.as_dtype(Dtype::Float32).unwrap().contiguous().unwrap().as_slice::<f32>().iter().all(|v| v.to_bits() == 0);
        for b in 0..n {
            for c in 0..nc {
                let r = b * nc + c;
                for i in 0..s {
                    let p = c * s + i;
                    let got = bq.index((r, .., i, ..));
                    if p < len {
                        assert_same(&format!("q b={b} c={c} i={i}"), &q.index((b, .., p, ..)), &got);
                    } else {
                        assert!(zeros(&got), "q b={b} c={c} i={i} past len is not zero");
                    }
                }
                for j in 0..3 * s {
                    let p = (c - 1) * s + j;
                    let (gk, gv) = (bk.index((r, .., j, ..)), bv.index((r, .., j, ..)));
                    if (0..len).contains(&p) {
                        assert_same(&format!("k b={b} c={c} j={j}"), &k.index((b, .., p, ..)), &gk);
                        assert_same(&format!("v b={b} c={c} j={j}"), &v.index((b, .., p, ..)), &gv);
                    } else {
                        assert!(zeros(&gk) && zeros(&gv), "k/v b={b} c={c} j={j} outside the sequence is not zero");
                    }
                }
            }
        }
    }

    /// A kernel whose banded launches differ from the MLX ops and whose plain launch does not:
    /// the load check keeps the kernel with `band` off and says why, the plain launch still
    /// matches, and the banded launch is refused. Without `band` the same kernel passes, as
    /// nothing banded is checked.
    #[test]
    fn a_failed_band_check_keeps_the_plain_kernel() {
        let (heads, hd, s) = (2, 8, 4);
        let (kernel, band_off) = SplitRope::with_wrong_band(heads as usize, hd as usize, Dtype::Float16, THETA, THETA, s as usize, true).unwrap();
        let why = band_off.expect("the banded launches must fail the check").to_string();
        eprintln!("band off as reported: {why}");
        assert!(why.contains("the kernel's k differs from the MLX ops (Unpacked banded launch)"), "{why}");
        assert!(!kernel.band);
        let (n, len) = (1, 7);
        let nc = chunks(len, s);
        let (_, _, expanded) = batch(n, len, &[len], 3 * heads * hd, Dtype::Float16);
        let dims = Array::from_slice(&[n, len, nc], &[3]);
        let (q, k, v) = mlx_path(&expanded, n, len, heads, hd, THETA).unwrap();
        let (gq, gk, gv) = kernel.apply(&expanded, None, &dims, false, n, len).unwrap();
        assert_same("plain q", &q, &gq);
        assert_same("plain k", &k, &gk);
        assert_same("plain v", &v, &gv);
        assert!(kernel.apply_band(&expanded, None, &dims, false, n, nc, false).is_err());
        assert!(kernel.apply_band(&expanded, None, &dims, false, n, nc, true).is_err());
        let (_, band_off) = SplitRope::with_wrong_band(heads as usize, hd as usize, Dtype::Float16, THETA, THETA, s as usize, false).unwrap();
        assert!(band_off.is_none(), "{band_off:?}");
    }

    /// A kernel built without `band` refuses the banded launch instead of running an
    /// unchecked pipeline.
    #[test]
    fn band_launch_needs_the_load_check() {
        let kernel = SplitRope::new(2, 8, Dtype::Float16, THETA, THETA, 4, false).unwrap().0;
        let x = Array::from_slice(&values(2 * 3 * 2 * 8), &[1, 2, 3 * 2 * 8]).as_dtype(Dtype::Float16).unwrap();
        let dims = Array::from_slice(&[1, 2, 1], &[3]);
        assert!(kernel.apply_band(&x, None, &dims, false, 1, 1, false).is_err());
        assert!(kernel.apply_band(&x, None, &dims, false, 1, 1, true).is_err());
    }

    /// A head dim that is not a multiple of 8 is refused at build time, with the number.
    #[test]
    fn odd_head_dim_is_a_config_error() {
        let err = SplitRope::new(4, 36, Dtype::Float16, THETA, THETA, 64, false).err().expect("36 is not a multiple of 8");
        assert!(err.to_string().contains("multiple of 8") && err.to_string().contains("36"), "{err}");
        assert!(SplitRope::new(4, 0, Dtype::Float16, THETA, THETA, 64, false).is_err());
        assert!(SplitRope::new(4, 64, Dtype::Float16, THETA, THETA, 0, false).is_err());
    }

    /// The reference chain itself: `v` is the plain head split and `q`, `k` are rotated, so a
    /// test that compares the kernel to it compares against real rope output.
    #[test]
    fn mlx_path_rotates_q_and_k_only() {
        let (n, len, heads, hd) = (1, 4, 2, 8);
        let x = Array::from_slice(&values((n * len * 3 * heads * hd) as usize), &[n, len, 3 * heads * hd]);
        let (q, k, v) = mlx_path(&x, n, len, heads, hd, THETA).unwrap();
        let parts = x.split_equal(3, -1).unwrap();
        let heads_of = |p: &Array| p.reshape(&[n, len, heads, hd]).unwrap().transpose_axes(&[0, 2, 1, 3]).unwrap();
        assert_same("v", &heads_of(&parts[2]), &v);
        let moved = |a: &Array, b: &Array| !ops::eq(a, b).unwrap().all(None).unwrap().item_exact::<bool>();
        assert!(moved(&heads_of(&parts[0]), &q) && moved(&heads_of(&parts[1]), &k));
        // Position 0 is never rotated.
        assert_same("q at position 0", &heads_of(&parts[0]).index((.., .., ..1, ..)), &q.index((.., .., ..1, ..)));
    }
}
