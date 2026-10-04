//! Custom Metal kernels through mlx-c's `mlx_fast_metal_kernel` API, which mlx-rs 0.32 does
//! not wrap. A [`MetalKernel`] is built once from its Metal source; [`MetalKernel::apply`] adds
//! one launch to MLX's lazy graph and returns the output arrays, so on the request path it
//! costs what any other MLX op costs. MLX compiles the source into a pipeline at the first
//! evaluation of each distinct instantiation (template arguments plus the dtypes of the inputs
//! and outputs) and keeps it for the life of the process. Callers keep that set fixed and
//! small, and say so where they build the kernel.
//!
//! Every mlx-c call is checked and every handle is freed on every path. A failing call is an
//! [`Exception`], never a panic: mlx-c hands the message to mlx-rs's error handler, which
//! `new` installs before the first call (mlx-c's own default handler exits the process). The
//! handler's stash is private to mlx-rs, so an error raised inside `apply` names the kernel and
//! the failing step instead of quoting MLX. A kernel whose source does not compile fails at the
//! first `eval` of its outputs, through mlx-rs, with the Metal compiler's message.

use mlx_rs::error::Exception;
use mlx_rs::{Array, Dtype, Stream};
use std::ffi::CString;
use std::os::raw::c_char;

/// A template argument of a kernel. MLX writes `template <typename T, int N, bool B>` before
/// the generated signature, in the order given to [`MetalKernel::apply`], and instantiates it
/// with these values.
#[derive(Clone, Copy, Debug)]
pub enum TemplateArg {
    Dtype(Dtype),
    Int(i32),
    Bool(bool),
}

/// A custom kernel: its name, its input and output names and its Metal body. MLX generates the
/// signature around the body. The body sees each input as `const device T* name` (`constant`
/// when the input has fewer than 8 elements), `name_shape` and `name_strides` when it mentions
/// them, each output as `device T* name`, and the thread attributes it mentions
/// (`thread_position_in_grid` and the others MLX knows). Inputs that are not row-contiguous
/// are copied before the launch. Building a kernel does not touch the GPU.
pub struct MetalKernel {
    raw: mlx_sys::mlx_fast_metal_kernel,
    name: String,
}

// SAFETY: `raw` owns an immutable description of the kernel (the names, source and flags,
// which MLX's `metal_kernel` captures by value); `apply` only reads it, and MLX guards its
// pipeline cache with its own locks. Freeing it from another thread is a plain `delete`.
unsafe impl Send for MetalKernel {}
unsafe impl Sync for MetalKernel {}

impl Drop for MetalKernel {
    fn drop(&mut self) {
        // SAFETY: `raw` came from `mlx_fast_metal_kernel_new` and is freed here and nowhere else.
        unsafe { mlx_sys::mlx_fast_metal_kernel_free(self.raw) }
    }
}

impl MetalKernel {
    /// Build a kernel from its Metal body. `inputs` and `outputs` name the buffers in the order
    /// `apply` takes them. The body may use `metal::` functions and the vector types; MLX's own
    /// prelude is in scope, no header is added.
    pub fn new(name: &str, inputs: &[&str], outputs: &[&str], source: &str) -> Result<Self, Exception> {
        Self::with_header(name, inputs, outputs, source, "")
    }

    /// [`MetalKernel::new`] with `header` placed before the generated kernel: includes and
    /// helpers the body uses. MLX keeps a copy of the header with the kernel.
    pub fn with_header(name: &str, inputs: &[&str], outputs: &[&str], source: &str, header: &str) -> Result<Self, Exception> {
        // mlx-rs installs its error handler on its first checked call; before that, mlx-c's
        // default handler exits the process on any error. This checked call makes sure a
        // failure below is reported, not fatal.
        Stream::thread_local_or_default().get_index()?;
        let cname = cstring(name, name)?;
        let csource = cstring(source, name)?;
        let cheader = cstring(header, name)?;
        let ins = StringVec::new(inputs, name)?;
        let outs = StringVec::new(outputs, name)?;
        // SAFETY: every pointer is a live C string or vector for the duration of the call, and
        // MLX copies what it keeps.
        let raw = unsafe {
            mlx_sys::mlx_fast_metal_kernel_new(cname.as_ptr(), ins.0, outs.0, csource.as_ptr(), cheader.as_ptr(), true, false)
        };
        if raw.ctx.is_null() {
            return Err(failed(name, "mlx_fast_metal_kernel_new"));
        }
        Ok(Self { raw, name: name.to_string() })
    }

    /// Add one launch to the lazy graph: `inputs` in the order the kernel names them, one
    /// `(shape, dtype)` per output, the template arguments in the order the body declares them,
    /// and the grid and threadgroup sizes in threads. Returns the outputs in order. The graph
    /// holds its own references to the inputs, so the caller may drop them afterwards.
    pub fn apply(
        &self,
        inputs: &[&Array],
        outputs: &[(&[i32], Dtype)],
        template: &[(&str, TemplateArg)],
        grid: [i32; 3],
        threadgroup: [i32; 3],
    ) -> Result<Vec<Array>, Exception> {
        let cfg = Config::new(&self.name)?;
        for (shape, dtype) in outputs {
            // SAFETY: `shape` outlives the call and `size` is its length.
            let status = unsafe {
                mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(cfg.0, shape.as_ptr(), shape.len(), *dtype as u32)
            };
            self.check(status, "add_output_arg")?;
        }
        // SAFETY: `cfg` is a live config.
        let status = unsafe { mlx_sys::mlx_fast_metal_kernel_config_set_grid(cfg.0, grid[0], grid[1], grid[2]) };
        self.check(status, "set_grid")?;
        // SAFETY: as above.
        let status = unsafe {
            mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(cfg.0, threadgroup[0], threadgroup[1], threadgroup[2])
        };
        self.check(status, "set_thread_group")?;
        for (name, value) in template {
            let cname = cstring(name, &self.name)?;
            // SAFETY: `cname` is a live C string; MLX copies it.
            let status = unsafe {
                match value {
                    TemplateArg::Dtype(d) => {
                        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg.0, cname.as_ptr(), *d as u32)
                    }
                    TemplateArg::Int(i) => mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(cfg.0, cname.as_ptr(), *i),
                    TemplateArg::Bool(b) => mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_bool(cfg.0, cname.as_ptr(), *b),
                }
            };
            self.check(status, "add_template_arg")?;
        }
        let ptrs: Vec<mlx_sys::mlx_array> = inputs.iter().map(|a| a.as_ptr()).collect();
        let in_vec = ArrayVec::new(&ptrs, &self.name)?;
        let mut out_vec = ArrayVec::empty();
        let stream = Stream::thread_local_or_default();
        // SAFETY: every handle is live for the call. MLX fills `out_vec` only on success and
        // takes its own references to the inputs for the graph.
        let status = unsafe { mlx_sys::mlx_fast_metal_kernel_apply(&mut out_vec.0, self.raw, in_vec.0, cfg.0, stream.as_ptr()) };
        self.check(status, "mlx_fast_metal_kernel_apply")?;
        out_vec.into_arrays(&self.name)
    }

    fn check(&self, status: i32, what: &str) -> Result<(), Exception> {
        if status == 0 {
            Ok(())
        } else {
            Err(failed(&self.name, what))
        }
    }
}

fn failed(kernel: &str, what: &str) -> Exception {
    Exception::custom(format!("custom Metal kernel `{kernel}`: {what} failed (MLX reported the cause to mlx-rs's error handler)"))
}

fn cstring(s: &str, kernel: &str) -> Result<CString, Exception> {
    CString::new(s).map_err(|_| Exception::custom(format!("custom Metal kernel `{kernel}`: a name or the source contains a NUL byte")))
}

/// An `mlx_fast_metal_kernel_config`, freed when dropped.
struct Config(mlx_sys::mlx_fast_metal_kernel_config);

impl Config {
    fn new(kernel: &str) -> Result<Self, Exception> {
        // SAFETY: a plain allocation; a null `ctx` means it failed.
        let raw = unsafe { mlx_sys::mlx_fast_metal_kernel_config_new() };
        if raw.ctx.is_null() {
            return Err(failed(kernel, "mlx_fast_metal_kernel_config_new"));
        }
        Ok(Self(raw))
    }
}

impl Drop for Config {
    fn drop(&mut self) {
        // SAFETY: `0` came from `mlx_fast_metal_kernel_config_new` and is freed only here.
        unsafe { mlx_sys::mlx_fast_metal_kernel_config_free(self.0) }
    }
}

/// An `mlx_vector_string`, freed when dropped.
struct StringVec(mlx_sys::mlx_vector_string);

impl StringVec {
    fn new(items: &[&str], kernel: &str) -> Result<Self, Exception> {
        let owned: Vec<CString> = items.iter().map(|s| cstring(s, kernel)).collect::<Result<_, _>>()?;
        let mut ptrs: Vec<*const c_char> = owned.iter().map(|s| s.as_ptr()).collect();
        // SAFETY: `ptrs` holds `ptrs.len()` live C strings; MLX copies them.
        let raw = unsafe { mlx_sys::mlx_vector_string_new_data(ptrs.as_mut_ptr(), ptrs.len()) };
        if raw.ctx.is_null() {
            return Err(failed(kernel, "mlx_vector_string_new_data"));
        }
        Ok(Self(raw))
    }
}

impl Drop for StringVec {
    fn drop(&mut self) {
        // SAFETY: `0` came from `mlx_vector_string_new_data` and is freed only here. Freeing a
        // valid vector does not fail, so the status is not read.
        unsafe { mlx_sys::mlx_vector_string_free(self.0) };
    }
}

/// An `mlx_vector_array`, freed when dropped. The vector holds its own references to its
/// arrays; the Rust `Array`s it was built from stay owned by the caller.
struct ArrayVec(mlx_sys::mlx_vector_array);

impl ArrayVec {
    fn new(arrays: &[mlx_sys::mlx_array], kernel: &str) -> Result<Self, Exception> {
        // SAFETY: `arrays` holds `arrays.len()` live array handles; MLX takes references.
        let raw = unsafe { mlx_sys::mlx_vector_array_new_data(arrays.as_ptr(), arrays.len()) };
        if raw.ctx.is_null() {
            return Err(failed(kernel, "mlx_vector_array_new_data"));
        }
        Ok(Self(raw))
    }

    /// A vector with no storage yet: mlx-c allocates it when it fills the vector, and freeing
    /// it while still empty is a no-op.
    fn empty() -> Self {
        Self(mlx_sys::mlx_vector_array { ctx: std::ptr::null_mut() })
    }

    /// Take one owned `Array` per element. The vector itself is freed on return, on error too.
    fn into_arrays(self, kernel: &str) -> Result<Vec<Array>, Exception> {
        // SAFETY: `self.0` is a live vector (or empty, size 0); every handle produced by
        // `mlx_vector_array_get` is a fresh reference owned by the returned `Array`.
        unsafe {
            let n = mlx_sys::mlx_vector_array_size(self.0);
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let mut a = mlx_sys::mlx_array_new();
                let status = mlx_sys::mlx_vector_array_get(&mut a, self.0, i);
                if status != 0 {
                    mlx_sys::mlx_array_free(a);
                    return Err(failed(kernel, "mlx_vector_array_get"));
                }
                out.push(Array::from_ptr(a));
            }
            Ok(out)
        }
    }
}

impl Drop for ArrayVec {
    fn drop(&mut self) {
        // SAFETY: `0` is a vector from mlx-c or the empty handle; freed only here. Freeing a
        // valid or empty vector does not fail, so the status is not read.
        unsafe { mlx_sys::mlx_vector_array_free(self.0) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops;

    /// Exact equality: same shape and dtype, every element the same (f16 compared through f32,
    /// which is lossless).
    fn assert_same(got: &Array, want: &Array, what: &str) {
        assert_eq!(got.shape(), want.shape(), "{what}: shape");
        assert_eq!(got.dtype(), want.dtype(), "{what}: dtype");
        let g = got.as_dtype(Dtype::Float32).unwrap().contiguous().unwrap();
        let w = want.as_dtype(Dtype::Float32).unwrap().contiguous().unwrap();
        assert_eq!(g.as_slice::<f32>(), w.as_slice::<f32>(), "{what}: values");
    }

    fn ramp(n: i32, dtype: Dtype) -> Array {
        Array::from_iter((0..n).map(|i| i as f32 * 0.5 - 100.0), &[n]).as_dtype(dtype).unwrap()
    }

    /// `out[i] = a[i] + b[i]` against `ops::add`, in f32 and f16, for a length that is not a
    /// multiple of the threadgroup and for one so short that MLX passes the inputs in the
    /// `constant` address space.
    #[test]
    fn elementwise_add_matches_the_mlx_op() {
        let k = MetalKernel::new(
            "laya_test_add",
            &["a", "b"],
            &["out"],
            r#"
            const uint i = thread_position_in_grid.x;
            if (i >= uint(a_shape[0])) return;
            out[i] = a[i] + b[i];
            "#,
        )
        .unwrap();
        for dtype in [Dtype::Float32, Dtype::Float16] {
            for n in [1000, 5] {
                let a = ramp(n, dtype);
                let b = ops::multiply(ramp(n, dtype), Array::from_f32(-0.25)).unwrap().as_dtype(dtype).unwrap();
                let want = ops::add(&a, &b).unwrap();
                let got = k.apply(&[&a, &b], &[(&[n], dtype)], &[], [n, 1, 1], [256, 1, 1]).unwrap();
                assert_eq!(got.len(), 1);
                assert_same(&got[0], &want, &format!("add {dtype:?} n={n}"));
            }
        }
    }

    /// `[R, C] -> [C, R]`, `VEC` columns per thread with a partial last block, optionally
    /// negated, converted through `T`: the three template argument kinds, the shape read from
    /// `x_shape`, and a layout change checked against `transpose`.
    #[test]
    fn transpose_copy_matches_the_mlx_op() {
        let k = MetalKernel::new(
            "laya_test_transpose",
            &["x"],
            &["out"],
            r#"
            const int R = x_shape[0];
            const int C = x_shape[1];
            const int blocks = (C + VEC - 1) / VEC;
            const int g = int(thread_position_in_grid.x);
            const int r = g / blocks;
            const int c0 = (g % blocks) * VEC;
            if (r >= R) return;
            for (int j = 0; j < VEC; j++) {
                const int c = c0 + j;
                if (c < C) {
                    const T v = T(x[r * C + c]);
                    out[c * R + r] = NEG ? -v : v;
                }
            }
            "#,
        )
        .unwrap();
        let (r, c) = (7, 10);
        let x = Array::from_iter((0..r * c).map(|i| i as f32 * 1.5 - 20.0), &[r, c]);
        for (vec, neg) in [(4, false), (1, true), (4, true)] {
            let t = x.transpose().unwrap();
            let want = if neg { t.negative().unwrap() } else { t };
            let blocks = (c + vec - 1) / vec;
            let got = k
                .apply(
                    &[&x],
                    &[(&[c, r], Dtype::Float32)],
                    &[("T", TemplateArg::Dtype(Dtype::Float32)), ("VEC", TemplateArg::Int(vec)), ("NEG", TemplateArg::Bool(neg))],
                    [r * blocks, 1, 1],
                    [32, 1, 1],
                )
                .unwrap();
            assert_same(&got[0], &want, &format!("transpose vec={vec} neg={neg}"));
        }
    }

    /// A body that does not compile is an error from `eval`, with the compiler's message, and
    /// MLX keeps working afterwards.
    #[test]
    fn syntax_error_is_an_error_not_an_abort() {
        let k = MetalKernel::new("laya_test_broken", &["a"], &["out"], "out[0] = a[0] +;").unwrap();
        let a = Array::from_slice(&[1.0f32], &[1]);
        let out = k.apply(&[&a], &[(&[1], Dtype::Float32)], &[], [1, 1, 1], [1, 1, 1]).unwrap();
        let err = out[0].eval().expect_err("a kernel with a syntax error must not evaluate");
        eprintln!("compile error as reported: {}", err.what());
        assert!(err.what().contains("laya_test_broken") || err.what().contains("error"), "{}", err.what());
        let after = ops::add(&a, &a).unwrap();
        assert_eq!(after.as_slice::<f32>(), &[2.0]);
    }

    /// A launch MLX refuses (here: fewer inputs than the kernel names) is an error from `apply`.
    #[test]
    fn bad_launch_is_an_error_from_apply() {
        let k = MetalKernel::new("laya_test_arity", &["a", "b"], &["out"], "out[0] = a[0] + b[0];").unwrap();
        let a = Array::from_slice(&[1.0f32], &[1]);
        let err = k.apply(&[&a], &[(&[1], Dtype::Float32)], &[], [1, 1, 1], [1, 1, 1]).expect_err("one input for a two-input kernel");
        assert!(err.what().contains("laya_test_arity"), "{}", err.what());
        assert!(MetalKernel::new("bad\0name", &["a"], &["out"], "").is_err());
    }
}
