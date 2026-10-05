//! `nax=all`: the encoder's gemms and the head's residual products on MLX's own NAX gemm loop,
//! launched with a tile shape and order of this module's choosing (`kernels/nax_gemm.metal`,
//! with MLX's loop from the verbatim MLX 0.32.2 headers in `kernels/mlx/`).
//!
//! MLX 0.32.2 runs an f16 `matmul` or `addmm` of 16 rows or more on its NAX gemm. Each
//! simdgroup computes its block of the output alone, so the threadgroup tile and the launch
//! order change which blocks share a weight tile in cache, never the sums. This module launches
//! smaller tiles (more threadgroups for the GPU's 10 cores at a few hundred rows) and runs the
//! row tiles that read the same weight tile back to back (MLX runs the column tiles of a row
//! first). Linears whose K MLX splits in two (wo2 up to 1,024 rows) get MLX's split-K result
//! in one launch: both partial sums stay in registers and combine as MLX combines them,
//! without MLX's fp32 scratch buffer and second launch.
//!
//! Two guards run once, at load, and turn the setting off with an `Err` that the caller prints:
//! [`mlx_uses_nax`] (MLX's own condition for its NAX gemms; elsewhere MLX runs other kernels,
//! which this module does not reproduce), and a bit-for-bit check of every kernel against
//! MLX's `matmul` or `addmm` at every weight shape of the model ([`NaxGemm::new`]). The check
//! runs on every load, before a kernel serves a request.
//!
//! The kernel set is fixed and small: 3 kernels (plain, residual, split-K residual), 5 tile
//! configurations and 2 K alignments, at most 16 pipelines in all (10 for the typed-decisions
//! model), which MLX compiles once and keeps for the life of the process. A new kernel source
//! or a new executable path costs the macOS Metal compiler about 1.7 s for the 10 (0.12 to
//! 0.23 s each, measured on an M5); after that macOS's shader cache serves them in about 5 ms.

use crate::metal_kernels::{MetalKernel, TemplateArg};
use crate::Lx;
use laya_core::{Error, Result};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{ops, random, transforms, Array, Dtype, Stream};
use std::time::Instant;

/// The MLX headers the kernels include, in include order, copied byte for byte from MLX 0.32.2
/// (`kernels/mlx/README.md`). The test `mlx_headers_are_verbatim` compares them with the MLX
/// build the runtime links.
const MLX_HEADERS: [(&str, &str); 5] = [
    ("steel/defines.h", include_str!("kernels/mlx/steel/defines.h")),
    ("steel/utils/type_traits.h", include_str!("kernels/mlx/steel/utils/type_traits.h")),
    ("steel/utils/integral_constant.h", include_str!("kernels/mlx/steel/utils/integral_constant.h")),
    ("steel/gemm/nax.h", include_str!("kernels/mlx/steel/gemm/nax.h")),
    ("steel/gemm/gemm_nax.h", include_str!("kernels/mlx/steel/gemm/gemm_nax.h")),
];
const MLX_LICENSE: &str = include_str!("kernels/mlx/LICENSE");
const BODY: &str = include_str!("kernels/nax_gemm.metal");
/// Below this many rows (or output columns) MLX sends a gemm to its gemv kernels (one row, and
/// the wide gemv up to 15 rows on M3 and later), not to the NAX gemm.
const MIN_ROWS: i32 = 16;
/// MLX's split-K partition size for 2048 < K <= 4096 (`steel_gemm_splitk_axpby_nax`).
const PART: i32 = 2048;
/// The row counts the load-time check tries first, in this order; [`check_rows`] keeps the ones
/// a shape needs and scans for any it misses. 16 is MLX's first NAX row count; 37 cuts a row
/// tile inside a 16-row fragment; 128, 384, 1,152 and 1,408 fill every row tile (multiples of
/// 128, MLX's tile height, and of this module's 64; 384 and 1,152 of 96 too); 321 is the first
/// count of the split-K 64 x 128 tiles; 1,024 / 1,025 and 1,365 / 1,366 are both sides of MLX's
/// split-K limits for K up to 4,096 (1,024 rows, and K / 3 rows for K 4,096).
const CHECK_ROWS: [i32; 11] = [MIN_ROWS, 37, 128, 321, 384, 1024, 1025, 1152, 1365, 1366, 1408];
/// The check's output elements per MLX graph (one host wait each): 16 MB per f16 side.
const CHECK_BATCH: i64 = 1 << 23;

/// The header of the `nax` kernels: an attribution and MLX's license as comments, the Metal
/// includes MLX's headers need, then the 5 MLX headers in include order. Two kinds of line are
/// left out because MLX compiles a custom kernel from one string with no include path: each
/// `#pragma once` and each `#include "mlx/..."`. Nothing else changes.
fn header() -> String {
    let mut out = String::from(
        "// MLX 0.32.2's NAX gemm loop, for the `nax` kernels of laya-mlx (nax_gemm.rs).\n\
         // The 5 files below are verbatim copies of mlx/backend/metal/kernels/steel/{defines.h,\n\
         // utils/type_traits.h, utils/integral_constant.h, gemm/nax.h, gemm/gemm_nax.h} from MLX\n\
         // 0.32.2 (https://github.com/ml-explore/mlx), without their `#pragma once` and\n\
         // `#include \"mlx/...\"` lines. MLX's license:\n//\n",
    );
    for line in MLX_LICENSE.lines() {
        out.push_str(if line.is_empty() { "//" } else { "// " });
        out.push_str(line);
        out.push('\n');
    }
    out.push_str(
        "\n#include <metal_stdlib>\n#include <metal_simdgroup>\n#include <metal_simdgroup_matrix>\n\
         #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace metal;\n",
    );
    for (path, text) in MLX_HEADERS {
        out.push_str(&format!("\n// ---- {path} ----\n"));
        for line in text.lines() {
            let t = line.trim_start();
            if t.starts_with("#pragma once") || t.starts_with("#include \"mlx/") {
                continue;
            }
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

/// What MLX does with an f16 gemm of `x [m, k]` by `w [n, k]` (matmul.cpp, MLX 0.32.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Route {
    /// gemv, or a split-K shape this module does not reproduce: MLX runs it.
    Mlx,
    /// The regular NAX gemm.
    Regular,
    /// The NAX split-K gemm with two K ranges split at [`PART`].
    Split,
}

pub(crate) fn route(m: i32, n: i32, k: i32) -> Route {
    if m < MIN_ROWS || n < MIN_ROWS {
        return Route::Mlx;
    }
    let mx = m.max(n);
    if k >= 3 * mx || (mx <= 1024 && k > 2 * mx) {
        return if k > PART && k <= 2 * PART { Route::Split } else { Route::Mlx };
    }
    Route::Regular
}

/// MLX's own condition for its NAX gemms (`metal::is_nax_available`, device.cpp, MLX 0.32.2):
/// macOS 26.2 or later and a GPU architecture generation of 17 or more, 18 for phone GPUs
/// (architecture names ending in `p`). Where it does not hold, MLX runs other gemm kernels,
/// which this module does not reproduce. Whether MLX was built without NAX (the macOS 14 and 15
/// builds) is not visible from here; there the load-time bit check still compares the outputs
/// with the gemm MLX does run.
pub(crate) fn mlx_uses_nax() -> std::result::Result<(), String> {
    let arch = gpu_architecture()?;
    if !arch_has_nax(&arch) {
        return Err(format!("MLX does not use NAX on this GPU (architecture `{arch}`)"));
    }
    let os = macos_version()?;
    if os < (26, 2) {
        return Err(format!("MLX uses NAX from macOS 26.2, this is {}.{}", os.0, os.1));
    }
    Ok(())
}

/// The generation as MLX reads it from the architecture name (the two characters before the
/// last, digits or 0), against MLX's threshold for the name's last character.
fn arch_has_nax(arch: &str) -> bool {
    let b = arch.as_bytes();
    let Some(&last) = b.last() else { return false };
    let digit = |c: u8| if c.is_ascii_digit() { i32::from(c - b'0') } else { 0 };
    let gen = if b.len() >= 3 { digit(b[b.len() - 3]) * 10 + digit(b[b.len() - 2]) } else { 0 };
    gen >= if last == b'p' { 18 } else { 17 }
}

/// An `mlx_device`, freed when dropped.
struct DeviceHandle(mlx_sys::mlx_device);

impl Drop for DeviceHandle {
    fn drop(&mut self) {
        // SAFETY: `0` came from `mlx_device_new_type` and is freed only here (a null `ctx` is a
        // no-op). Freeing a device does not fail, so the status is not read.
        unsafe { mlx_sys::mlx_device_free(self.0) };
    }
}

/// An `mlx_device_info`, freed when dropped.
struct InfoHandle(mlx_sys::mlx_device_info);

impl Drop for InfoHandle {
    fn drop(&mut self) {
        // SAFETY: `0` came from `mlx_device_info_new` (null `ctx`, a no-op to free) and was
        // filled in place by `mlx_device_info_get`; freed only here. Freeing it does not fail,
        // so the status is not read.
        unsafe { mlx_sys::mlx_device_info_free(self.0) };
    }
}

/// The GPU's architecture name as MLX reports it (`device_info`, which honours MLX's
/// `MLX_METAL_GPU_ARCH` override as MLX's NAX check does).
fn gpu_architecture() -> std::result::Result<String, String> {
    // mlx-rs installs its error handler on its first checked call; before that, mlx-c's
    // default handler exits the process on any error. This checked call makes sure a failure
    // below is reported, not fatal.
    Stream::thread_local_or_default().get_index().map_err(|e| format!("MLX is not usable: {e}"))?;
    // SAFETY: a plain allocation; a null `ctx` means it failed.
    let dev = DeviceHandle(unsafe { mlx_sys::mlx_device_new_type(mlx_sys::mlx_device_type__MLX_GPU, 0) });
    if dev.0.ctx.is_null() {
        return Err("MLX has no GPU device".into());
    }
    // SAFETY: an empty handle that `mlx_device_info_get` fills in place.
    let mut info = InfoHandle(unsafe { mlx_sys::mlx_device_info_new() });
    // SAFETY: `info.0` and `dev.0` are live handles for the call.
    let status = unsafe { mlx_sys::mlx_device_info_get(&mut info.0, dev.0) };
    if status != 0 {
        return Err(format!("MLX cannot describe the GPU (mlx_device_info_get returned {status})"));
    }
    let mut value: *const std::os::raw::c_char = std::ptr::null();
    // SAFETY: `info.0` is live and the key is a C string; `value` points into `info`.
    let status = unsafe { mlx_sys::mlx_device_info_get_string(&mut value, info.0, c"architecture".as_ptr()) };
    if status != 0 || value.is_null() {
        return Err(format!("MLX reports no GPU architecture (mlx_device_info_get_string returned {status})"));
    }
    // SAFETY: `value` is a NUL-terminated string owned by `info`, copied here before `info` is
    // freed.
    Ok(unsafe { std::ffi::CStr::from_ptr(value) }.to_string_lossy().into_owned())
}

/// The running macOS version, major and minor (`kern.osproductversion`).
fn macos_version() -> std::result::Result<(u32, u32), String> {
    extern "C" {
        fn sysctlbyname(
            name: *const std::os::raw::c_char,
            oldp: *mut std::os::raw::c_void,
            oldlenp: *mut usize,
            newp: *mut std::os::raw::c_void,
            newlen: usize,
        ) -> std::os::raw::c_int;
    }
    let mut buf = [0u8; 64];
    let mut len = buf.len();
    // SAFETY: `buf` is writable for `len` bytes and sysctl writes at most that many.
    let status = unsafe {
        sysctlbyname(c"kern.osproductversion".as_ptr(), buf.as_mut_ptr().cast(), &mut len, std::ptr::null_mut(), 0)
    };
    let text = std::str::from_utf8(&buf[..len.min(buf.len())]).unwrap_or("").trim_end_matches('\0');
    let mut parts = text.split('.').map(|p| p.parse::<u32>());
    match (status, parts.next(), parts.next().unwrap_or(Ok(0))) {
        (0, Some(Ok(major)), Ok(minor)) => Ok((major, minor)),
        _ => Err(format!("cannot read the macOS version (`{text}`, status {status})")),
    }
}

/// Tile `bm x bn` over `wm x wn` simdgroups, K block `bk`, bands of `g` row tiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Cfg {
    bm: i32,
    bn: i32,
    wm: i32,
    wn: i32,
    bk: i32,
    g: i32,
}

const fn cfg(bm: i32, bn: i32, wm: i32, wn: i32, bk: i32, g: i32) -> Cfg {
    Cfg { bm, bn, wm, wn, bk, g }
}

/// The configuration for a gemm, or `None` where MLX's own launch was as fast. From the research
/// round 2 sweep on an M5 with each gemm waiting for the one before, as in the forward (row
/// counts 184 to 5,707, time against MLX's launch in brackets):
///
/// - split-K (wo2 up to 1,024 rows): 64 x 64 tiles up to 320 rows, 64 x 128 above (0.62 to 0.70)
/// - every other gemm up to 1,024 rows: 64 x 64 tiles, row tiles first (0.86 to 0.96)
/// - past 1,024 rows: 96 x 64 tiles in bands of 8 row tiles (0.93 to 1.01); wo2 (K 2,624)
///   64 x 128 in bands of 8 up to 3,000 rows (0.92 to 0.96), MLX's launch past that
///
/// Small tiles win here because one gemm runs alone on the GPU: more threadgroups keep the 10
/// cores busy, and the row-first order keeps a weight tile in cache for the row tiles that read
/// it.
fn select(route: Route, m: i32, k: i32) -> Option<Cfg> {
    match route {
        Route::Mlx => None,
        Route::Split if m <= 320 => Some(cfg(64, 64, 2, 2, 256, 64)),
        Route::Split => Some(cfg(64, 128, 2, 4, 256, 64)),
        Route::Regular if m <= 1024 => Some(cfg(64, 64, 2, 2, 512, 64)),
        Route::Regular if k > PART && m <= 3000 => Some(cfg(64, 128, 2, 4, 512, 8)),
        Route::Regular if k > PART => None,
        Route::Regular => Some(cfg(96, 64, 3, 2, 512, 8)),
    }
}

/// A kernel instantiation: its route, tile configuration and whether K (for split-K, the second
/// range) is a multiple of the K block.
type Inst = (Route, Cfg, bool);

/// The instantiation a gemm of `m` rows by a weight `(n, k)` runs on, with or without a
/// residual. `None` where MLX's own gemm runs.
fn pick(m: i32, n: i32, k: i32, residual: bool) -> Option<Inst> {
    let r = route(m, n, k);
    // MLX's split-K matmul without a residual is not reproduced: it stays with MLX.
    if r == Route::Split && !residual {
        return None;
    }
    let c = select(r, m, k)?;
    let aligned = if r == Route::Split { (k - PART) % c.bk == 0 } else { k % c.bk == 0 };
    Some((r, c, aligned))
}

/// The row counts the load-time check runs for a weight `(n, k)`, taken from [`CHECK_ROWS`] and
/// then from every count up to 3,001 in order (past 3,000 rows the pick no longer changes), at
/// counts where the kernel runs:
///
/// - each count next to a change of MLX's route (gemv to NAX gemm at 16 rows, split-K to the
///   regular gemm): the route is MLX's choice, so both sides of each edge are checked
/// - for each instantiation, the first count that cuts the last row tile inside a 16-row
///   fragment, and the first that fills every row tile, this module's and MLX's (a multiple of
///   128 and of the tile height)
///
/// A cut launch runs full and partial simdgroup blocks side by side, so each instantiation
/// meets both edge paths of the kernel, and MLX's gemm runs with and without its `align_M`.
/// The Laya checkpoints' shapes need no count beyond [`CHECK_ROWS`].
fn check_rows(n: i32, k: i32, residual: bool) -> Vec<i32> {
    let edge = |m: i32| route(m - 1, n, k) != route(m, n, k) || route(m, n, k) != route(m + 1, n, k);
    let mut rows = Vec::new();
    let mut seen: Vec<(Inst, bool)> = Vec::new();
    for m in CHECK_ROWS.into_iter().chain(MIN_ROWS..=3001) {
        let Some(p) = pick(m, n, k, residual) else { continue };
        if rows.contains(&m) {
            continue;
        }
        let full = m % full_rows(p.1.bm) == 0;
        let kind = (full || m % 16 != 0).then_some((p, full));
        let new = kind.is_some_and(|kind| !seen.contains(&kind));
        if new || edge(m) {
            rows.push(m);
            if let Some(kind) = kind.filter(|kind| !seen.contains(kind)) {
                seen.push(kind);
            }
        }
    }
    rows.sort_unstable();
    rows
}

/// The smallest row count that fills every row tile of height `bm` and of MLX's 128.
fn full_rows(bm: i32) -> i32 {
    let (mut a, mut b) = (bm, 128);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    bm / a * 128
}

/// The 3 `nax` kernels. Built and checked once per load; [`NaxGemm::matmul`] and
/// [`NaxGemm::addmm`] add one launch to MLX's lazy graph, or return `None` where MLX's own gemm
/// runs.
pub(crate) struct NaxGemm {
    mm: MetalKernel,
    addmm: MetalKernel,
    split: MetalKernel,
}

impl NaxGemm {
    /// Build the kernels and check them against MLX for every weight shape `(n, k)` in `mm`
    /// (plain products) and `addmm` (products plus a residual). Returns the kernels and a line
    /// for the load log: how many gemms the check ran and how long it took. An `Err` means the
    /// setting must not be used here: MLX does not use NAX on this machine ([`mlx_uses_nax`]), a
    /// kernel does not compile, or a kernel differs from MLX's gemm in any bit.
    pub(crate) fn new(mm: &[(i32, i32)], addmm: &[(i32, i32)]) -> Result<(Self, String)> {
        let this = Self::build()?;
        let mut shapes: Vec<(i32, i32, bool)> = Vec::new();
        for (list, residual) in [(mm, false), (addmm, true)] {
            for &(n, k) in list {
                if !shapes.contains(&(n, k, residual)) {
                    shapes.push((n, k, residual));
                }
            }
        }
        let t = Instant::now();
        let ran = this.check(&shapes)?;
        let note = format!("nax check ran: {ran} gemms bit-identical to MLX's in {:.0} ms", ms(t));
        Ok((this, note))
    }

    /// The gate and the 3 kernels, unchecked.
    fn build() -> Result<Self> {
        mlx_uses_nax().map_err(|e| Error::Backend(format!("nax: {e}")))?;
        let header = header();
        let build = |mode: i32, name: &str, inputs: &[&str]| {
            MetalKernel::with_header(name, inputs, &["out"], &source(mode), &header)
                .map_err(|e| Error::Backend(format!("nax: {e}")))
        };
        Ok(Self {
            mm: build(0, "laya_nax_mm", &["a", "w"])?,
            addmm: build(1, "laya_nax_addmm", &["a", "w", "c"])?,
            split: build(2, "laya_nax_split", &["a", "w", "c"])?,
        })
    }

    /// `a @ w^T` for `a [.., k]` and `w [n, k]` in f16, or `None` where MLX's own gemm runs.
    pub(crate) fn matmul(&self, a: &Array, w: &Array) -> Result<Option<Array>> {
        self.run(a, w, None)
    }

    /// `c + a @ w^T` (`c [.., n]` with `a`'s leading shape), or `None` where MLX's runs.
    pub(crate) fn addmm(&self, a: &Array, w: &Array, c: &Array) -> Result<Option<Array>> {
        self.run(a, w, Some(c))
    }

    fn run(&self, a: &Array, w: &Array, c: Option<&Array>) -> Result<Option<Array>> {
        let (n, k) = (w.dim(0), w.dim(1));
        if k == 0 || a.dim(-1) != k || [a.dtype(), w.dtype()].iter().any(|&d| d != Dtype::Float16) {
            return Ok(None);
        }
        let m = (a.size() as i32) / k;
        let Some(inst) = pick(m, n, k, c.is_some()) else {
            return Ok(None);
        };
        if let Some(c) = c {
            if c.dim(-1) != n || c.size() as i32 != m * n || c.dtype() != Dtype::Float16 {
                return Ok(None);
            }
        }
        let mut shape = a.shape().to_vec();
        *shape.last_mut().expect("a has at least one axis") = n;
        let a2 = a.reshape(&[m, k]).lx()?;
        let c2 = c.map(|c| c.reshape(&[m, n])).transpose().lx()?;
        let out = self.launch(inst, &a2, w, c2.as_ref())?;
        Ok(Some(out.reshape(&shape).lx()?))
    }

    /// One launch of the instantiation `(route, cfg, aligned)` on `a [m, k]`, `w [n, k]` and,
    /// for the residual kernels, `c [m, n]`: one threadgroup per output tile, exactly (the kernel
    /// maps each to its band position).
    fn launch(&self, (route, cfg, aligned): Inst, a: &Array, w: &Array, c: Option<&Array>) -> Result<Array> {
        let (m, n) = (a.dim(0), w.dim(0));
        let (kernel, inputs) = match (c, route) {
            (Some(c), Route::Split) => (&self.split, vec![a, w, c]),
            (Some(c), _) => (&self.addmm, vec![a, w, c]),
            (None, _) => (&self.mm, vec![a, w]),
        };
        let mut template = vec![
            ("BM", TemplateArg::Int(cfg.bm)),
            ("BN", TemplateArg::Int(cfg.bn)),
            ("WM", TemplateArg::Int(cfg.wm)),
            ("WN", TemplateArg::Int(cfg.wn)),
            ("BK", TemplateArg::Int(cfg.bk)),
            ("ALIGN_K", TemplateArg::Bool(aligned)),
            ("G", TemplateArg::Int(cfg.g)),
        ];
        if route == Route::Split {
            template.push(("PART", TemplateArg::Int(PART)));
        }
        let tiles = ((m + cfg.bm - 1) / cfg.bm) * ((n + cfg.bn - 1) / cfg.bn);
        let threads = 32 * cfg.wm * cfg.wn;
        let out = kernel
            .apply(&inputs, &[(&[m, n], Dtype::Float16)], &template, [tiles * threads, 1, 1], [threads, 1, 1])
            .lx()?;
        out.into_iter().next().ok_or_else(|| Error::Backend("nax: the kernel returned no output".into()))
    }

    /// The load check: for each weight shape `(n, k, residual)`, random activations, a weight
    /// scaled like the model's and (with `residual`) a residual with outlier columns past 1,000,
    /// at the row counts of [`check_rows`], compared bit for bit with MLX's gemm through a
    /// transposed view of the weight, as [`crate::Linear`] calls it. The inputs are shared: one
    /// weight per K at the largest N, one activation per K and one residual per N at the largest
    /// row count, each gemm taking the leading rows (a view of the same buffer). The comparisons
    /// run as one MLX graph per [`CHECK_BATCH`] output elements, one host wait each. Returns how
    /// many gemms ran on the kernels.
    fn check(&self, shapes: &[(i32, i32, bool)]) -> Result<usize> {
        let cases: Vec<(i32, i32, bool, i32)> =
            shapes.iter().flat_map(|&(n, k, r)| check_rows(n, k, r).into_iter().map(move |m| (n, k, r, m))).collect();
        // (k, largest n), (k, largest m), (n, largest m with a residual).
        let (mut wmax, mut amax, mut cmax) = (Vec::new(), Vec::new(), Vec::new());
        let grow = |v: &mut Vec<(i32, i32)>, key: i32, size: i32| match v.iter_mut().find(|e| e.0 == key) {
            Some(e) => e.1 = e.1.max(size),
            None => v.push((key, size)),
        };
        for &(n, k, residual, m) in &cases {
            grow(&mut wmax, k, n);
            grow(&mut amax, k, m);
            if residual {
                grow(&mut cmax, n, m);
            }
        }
        let normal = |shape: &[i32], scale: f32, seed: u64| -> Result<Array> {
            let key = random::key(seed).lx()?;
            random::normal::<f32>(shape, None::<f32>, scale, &key).lx()?.as_dtype(Dtype::Float16).lx()
        };
        let weights = wmax.iter().map(|&(k, n)| Ok((k, normal(&[n, k], 0.03, 7)?))).collect::<Result<Vec<_>>>()?;
        let acts = amax.iter().map(|&(k, m)| Ok((k, normal(&[m, k], 1.0, 8)?))).collect::<Result<Vec<_>>>()?;
        let residuals = cmax
            .iter()
            .map(|&(n, m)| {
                let c = normal(&[m, n], 8.0, 9)?;
                let outlier_cols: Vec<bool> = (0..n).map(|j| j % 97 == 5).collect();
                let outlier_cols = Array::from_slice(&outlier_cols, &[n]);
                let outliers = ops::multiply(&c, Array::from_f32(150.0).as_dtype(Dtype::Float16).lx()?).lx()?;
                Ok((n, ops::select(&outlier_cols, &outliers, &c).lx()?))
            })
            .collect::<Result<Vec<_>>>()?;
        // Generate the inputs first, so their f32 draws are freed before the gemms run.
        transforms::eval(weights.iter().chain(&acts).chain(&residuals).map(|e| &e.1)).lx()?;
        let find = |v: &[(i32, Array)], key: i32| v.iter().find(|e| e.0 == key).map(|e| e.1.clone()).expect("an input per key");
        // MLX's gemm and the kernel's for one case, or `None` where MLX's gemm runs.
        let gemms = |&(n, k, residual, m): &(i32, i32, bool, i32)| -> Result<Option<(Array, Array)>> {
            let w = find(&weights, k).index((..n, ..));
            let wt = ops::transpose(&w).lx()?;
            let a = find(&acts, k).index((..m, ..));
            Ok(if residual {
                let c = find(&residuals, n).index((..m, ..));
                self.addmm(&a, &w, &c)?.map(|got| ops::addmm(&c, &a, &wt, None, None).map(|want| (want, got))).transpose().lx()?
            } else {
                self.matmul(&a, &w)?.map(|got| ops::matmul(&a, &wt).map(|want| (want, got))).transpose().lx()?
            })
        };
        // Evaluate a batch of comparisons (one bool each, same shape and bits) with one host
        // wait; on a difference, recompute that case for the message.
        let settle = |batch: &mut Vec<(usize, Array)>| -> Result<()> {
            if batch.is_empty() {
                return Ok(());
            }
            let same = ops::stack(&batch.iter().map(|b| &b.1).collect::<Vec<_>>(), 0).lx()?;
            transforms::eval([&same]).lx()?;
            if let Some(j) = same.as_slice::<bool>().iter().position(|&s| !s) {
                let (n, k, residual, m) = cases[batch[j].0];
                let (want, got) = gemms(&cases[batch[j].0])?.expect("the kernel ran for this case");
                let diff = ops::abs(ops::subtract(&got, &want).lx()?).lx()?.max(None).lx()?;
                transforms::eval([&diff]).lx()?;
                return Err(Error::Backend(format!(
                    "nax: the kernel differs from MLX's {} ({m} x {k} by {n}, max diff {})",
                    if residual { "addmm" } else { "matmul" },
                    diff.as_dtype(Dtype::Float32).lx()?.try_item_exact::<f32>().unwrap_or(f32::NAN)
                )));
            }
            batch.clear();
            Ok(())
        };
        let bits = |x: &Array| x.view_dtype(Dtype::Uint16).lx();
        let mut ran = 0;
        let mut batch: Vec<(usize, Array)> = Vec::new();
        let mut size = 0i64;
        for (i, case) in cases.iter().enumerate() {
            let Some((want, got)) = gemms(case)? else { continue };
            ran += 1;
            batch.push((i, ops::array_eq(bits(&want)?, bits(&got)?, false).lx()?));
            size += i64::from(case.0) * i64::from(case.3);
            if size >= CHECK_BATCH {
                settle(&mut batch)?;
                size = 0;
            }
        }
        settle(&mut batch)?;
        Ok(ran)
    }
}

/// The full source of kernel `mode` (0 plain, 1 residual, 2 split-K residual), less the header.
fn source(mode: i32) -> String {
    format!("#define MODE {mode}\n{BODY}")
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The routing rule follows MLX's: gemv below 16 rows, split-K for wo2 (K 2624, N 1024)
    /// up to 1024 rows and for the head's linear2 (K 4096) up to 1365, the regular gemm else.
    #[test]
    fn route_follows_mlx() {
        assert_eq!(route(15, 1024, 2624), Route::Mlx);
        assert_eq!(route(16, 1024, 2624), Route::Split);
        assert_eq!(route(1024, 1024, 2624), Route::Split);
        assert_eq!(route(1025, 1024, 2624), Route::Regular);
        assert_eq!(route(1365, 1024, 4096), Route::Split);
        assert_eq!(route(1366, 1024, 4096), Route::Regular);
        assert_eq!(route(151, 3072, 1024), Route::Regular);
        assert_eq!(route(151, 1024, 1024), Route::Regular);
        assert_eq!(route(16, 1024, 8192), Route::Mlx);
    }

    /// The tile table: the pick at the sweep's row counts.
    #[test]
    fn select_follows_sweep() {
        let pick = |m, n, k| select(route(m, n, k), m, k).map(|c| (c.bm, c.bn, c.bk, c.g));
        assert_eq!(pick(184, 1024, 2624), Some((64, 64, 256, 64)));
        assert_eq!(pick(570, 1024, 2624), Some((64, 128, 256, 64)));
        assert_eq!(pick(184, 3072, 1024), Some((64, 64, 512, 64)));
        assert_eq!(pick(774, 1024, 1024), Some((64, 64, 512, 64)));
        assert_eq!(pick(1379, 1024, 2624), Some((64, 128, 512, 8)));
        assert_eq!(pick(5707, 1024, 2624), None);
        assert_eq!(pick(2250, 5248, 1024), Some((96, 64, 512, 8)));
        assert_eq!(pick(8, 1024, 2624), None);
    }

    /// Which of the 3 kernels a route runs on, with or without a residual (0 plain, 1 residual,
    /// 2 split-K residual): MLX compiles one pipeline per kernel, tile configuration and K
    /// alignment.
    fn kernel_of(route: Route, residual: bool) -> u8 {
        match (route, residual) {
            (Route::Split, _) => 2,
            (_, true) => 1,
            (_, false) => 0,
        }
    }

    /// The typed-decisions model's weight shapes (encoder and head): `(n, k, residual)`.
    const TYPED: [(i32, i32, bool); 5] =
        [(3072, 1024, false), (5248, 1024, false), (1024, 1024, true), (1024, 2624, true), (1024, 4096, true)];

    /// Over every row count up to 9,000, each weight shape uses no instantiation (route, tile
    /// configuration, K alignment) that its check rows miss; each instantiation is checked with
    /// a cut last row tile and, where its range has one, with every row tile full; and both
    /// sides of each change of MLX's route are checked where the kernel runs. Sizes around
    /// MLX's split-K limits and the table's.
    #[test]
    fn check_rows_reach_every_instantiation() {
        let sizes = [16, 100, 320, 768, 1000, 1024, 1152, 1200, 1365, 2304, 2624, 3000, 3072, 3500, 4096, 5248, 8192, 20000];
        let mut beyond = 0;
        for n in sizes {
            for k in sizes {
                for residual in [false, true] {
                    let rows = check_rows(n, k, residual);
                    beyond += rows.iter().filter(|m| !CHECK_ROWS.contains(m)).count();
                    let at = |m: i32| pick(m, n, k, residual);
                    let got: Vec<_> = rows.iter().filter_map(|&m| at(m)).collect();
                    assert_eq!(got.len(), rows.len(), "n {n}, k {k}: a check row where MLX's gemm runs");
                    for m in MIN_ROWS..=9000 {
                        let Some(p) = at(m) else { continue };
                        let ctx = format!("n {n}, k {k}, residual {residual}: {m} rows run {p:?}");
                        assert!(got.contains(&p), "{ctx}, unchecked");
                        if m > 3001 {
                            continue;
                        }
                        let checked = |f: &dyn Fn(i32) -> bool| rows.iter().any(|&r| at(r) == Some(p) && f(r));
                        if m % 16 != 0 {
                            assert!(checked(&|r| r % 16 != 0), "{ctx}, no cut tile checked");
                        }
                        if m % full_rows(p.1.bm) == 0 {
                            assert!(checked(&|r| r % full_rows(p.1.bm) == 0), "{ctx}, no full tiles checked");
                        }
                        let edge = route(m - 1, n, k) != route(m, n, k) || route(m, n, k) != route(m + 1, n, k);
                        assert!(!edge || rows.contains(&m), "{ctx}, at an edge of MLX's route, unchecked");
                    }
                }
            }
        }
        // K above 4,096 moves the regular gemm's first row count past 1,365 (MLX's own gemm
        // runs below it), so some of these shapes need a row beyond the list.
        assert!(beyond > 0);
        assert_eq!((full_rows(64), full_rows(96), full_rows(128)), (128, 384, 128));
        // The pipelines MLX compiles for the typed-decisions model: one per distinct (kernel,
        // tile configuration, K alignment).
        let mut pipelines: Vec<_> = Vec::new();
        for (n, k, residual) in TYPED {
            for m in MIN_ROWS..=3001 {
                if let Some((r, c, a)) = pick(m, n, k, residual) {
                    if !pipelines.contains(&(kernel_of(r, residual), c, a)) {
                        pipelines.push((kernel_of(r, residual), c, a));
                    }
                }
            }
        }
        assert_eq!(pipelines.len(), 10, "{pipelines:?}");
        // typed-decisions and english (ModernBERT-large), multilingual (mmBERT-base) encoders,
        // and the typed-decisions head: 31 gemms for typed-decisions (53 with the old list).
        let regular = vec![16, 37, 128, 1025, 1152];
        for (n, k, residual, want) in [
            (3072, 1024, false, regular.clone()),
            (5248, 1024, false, regular.clone()),
            (1024, 1024, true, regular.clone()),
            (1024, 2624, true, vec![16, 37, 128, 321, 384, 1024, 1025, 1152]),
            (1024, 4096, true, vec![16, 37, 128, 321, 384, 1365, 1366, 1408]),
            (2304, 768, false, regular.clone()),
            (768, 768, true, regular.clone()),
            (768, 1152, true, regular.clone()),
        ] {
            assert_eq!(check_rows(n, k, residual), want, "n {n}, k {k}");
        }
    }

    /// The architecture rule copies MLX's: the generation from the two characters before the
    /// last, 17 and up, 18 and up for phone GPUs, 0 for a name too short or without digits.
    #[test]
    fn arch_rule_follows_mlx() {
        assert!(arch_has_nax("applegpu_g17g"));
        assert!(arch_has_nax("applegpu_g17s"));
        assert!(arch_has_nax("applegpu_g18d"));
        assert!(!arch_has_nax("applegpu_g16g"));
        assert!(!arch_has_nax("applegpu_g15s"));
        assert!(!arch_has_nax("applegpu_g17p"));
        assert!(arch_has_nax("applegpu_g18p"));
        assert!(!arch_has_nax("g"));
        assert!(!arch_has_nax(""));
        assert!(!arch_has_nax("applegpu_gxxg"));
    }

    /// The copies in `kernels/mlx/` match the headers of the MLX build the runtime links, byte
    /// for byte, when that build is a prebuilt (`MLX_SYS_PREBUILT_DIR`, as `bench/env.sh`
    /// sets it). Skipped with a note otherwise.
    #[test]
    fn mlx_headers_are_verbatim() {
        let Some(dir) = std::env::var_os("MLX_SYS_PREBUILT_DIR") else {
            eprintln!("skipped: MLX_SYS_PREBUILT_DIR is not set");
            return;
        };
        let kernels = std::path::Path::new(&dir).join("include/mlx/backend/metal/kernels");
        if !kernels.is_dir() {
            eprintln!("skipped: {} has no MLX kernel headers", kernels.display());
            return;
        }
        for (path, text) in MLX_HEADERS {
            let upstream = std::fs::read_to_string(kernels.join(path)).unwrap_or_else(|e| panic!("{path}: {e}"));
            assert!(text == upstream, "kernels/mlx/{path} differs from the linked MLX's copy");
        }
    }

    /// The assembled header keeps every MLX line but the `#pragma once` and `#include "mlx/..."`
    /// ones, in order, and carries the attribution and the license.
    #[test]
    fn header_is_the_mlx_files_without_their_includes() {
        let h = header();
        assert!(h.contains("MLX 0.32.2") && h.contains("// Copyright © 2023 Apple Inc."));
        assert!(h.contains("// Permission is hereby granted, free of charge"));
        assert!(!h.lines().any(|l| l.trim_start().starts_with("#pragma once")));
        assert!(!h.lines().any(|l| l.trim_start().starts_with("#include \"")));
        let mut rest = h.as_str();
        for (path, text) in MLX_HEADERS {
            for line in text.lines() {
                let t = line.trim_start();
                if t.starts_with("#pragma once") || t.starts_with("#include \"mlx/") {
                    continue;
                }
                let at = rest.find(line).unwrap_or_else(|| panic!("{path}: line `{line}` missing or out of order"));
                rest = &rest[at + line.len()..];
            }
        }
    }

    /// Whether this machine passes the NAX gate; the GPU tests below skip with a note where it
    /// does not (MLX runs other gemms there, so there is nothing to compare).
    fn nax_machine() -> bool {
        eprintln!("nax gate: architecture {:?}, macOS {:?}, {:?}", gpu_architecture(), macos_version(), mlx_uses_nax());
        mlx_uses_nax().is_ok()
    }

    /// The load-time check passes for the encoder's and the head's shapes of the
    /// typed-decisions model, bit for bit, with a kernel at every check row. Uses the GPU.
    #[test]
    fn nax_matches_mlx() {
        if !nax_machine() {
            eprintln!("skipped: MLX does not use NAX here");
            return;
        }
        let nax = NaxGemm::build().unwrap_or_else(|e| panic!("{e}"));
        for (n, k, residual) in TYPED {
            let ran = nax.check(&[(n, k, residual)]).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(ran, check_rows(n, k, residual).len(), "n {n}, k {k}");
        }
        assert_eq!(nax.check(&TYPED).unwrap_or_else(|e| panic!("{e}")), 31);
    }

    /// A kernel that differs from MLX's gemm in one element fails the load check, alone or
    /// among passing shapes. Uses the GPU.
    #[test]
    fn check_rejects_a_kernel_that_differs() {
        if !nax_machine() {
            eprintln!("skipped: MLX does not use NAX here");
            return;
        }
        let mut k = NaxGemm::build().unwrap_or_else(|e| panic!("{e}"));
        // After every store of threadgroup 0, overwrite out[0] (a random sum, never exactly 7).
        let wrong = format!("{}\nthreadgroup_barrier(mem_flags::mem_device);\nif (t == 0) out[0] = half(7);\n", source(0));
        k.mm = MetalKernel::with_header("laya_nax_wrong", &["a", "w"], &["out"], &wrong, &header()).unwrap();
        let err = k.check(&[(1024, 1024, false)]).expect_err("a changed kernel must fail the check");
        let msg = err.to_string();
        eprintln!("check error as reported: {msg}");
        assert!(msg.contains("differs from MLX's matmul (16 x 1024 by 1024"), "{msg}");
        // After passing shapes, in a later batch: the first shape alone has 12.4M output
        // elements against a batch of 8.4M, and the plain one comes last.
        let shapes = [(5248, 1024, true), (1024, 4096, true), (5248, 1024, false)];
        let msg = k.check(&shapes).expect_err("a changed kernel must fail the check").to_string();
        assert!(msg.contains("differs from MLX's matmul (16 x 1024 by 5248"), "{msg}");
    }

    /// Every load runs the check, the shapes counted once each, and its log line gives the
    /// gemm count and the time. Uses the GPU.
    #[test]
    fn every_load_runs_the_check() {
        if !nax_machine() {
            eprintln!("skipped: MLX does not use NAX here");
            return;
        }
        let mm = [(3072, 1024), (5248, 1024), (3072, 1024)];
        let addmm = [(1024, 1024), (1024, 2624), (1024, 4096)];
        for load in ["first", "second"] {
            let (_, note) = NaxGemm::new(&mm, &addmm).unwrap_or_else(|e| panic!("{e}"));
            eprintln!("{load} load: {note}");
            assert!(note.starts_with("nax check ran: 31 gemms bit-identical to MLX's in ") && note.ends_with(" ms"), "{note}");
        }
    }
}
