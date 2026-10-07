---
name: sys1rust-develop
description: Build sys1rust from source and change its Rust code (laya-core, laya-mlx with its Metal kernels, the sys1rust server, sys1-bench), run its unit, GPU and live test suites, check that answers stay the same, measure speed, and follow the repository's rules for commits, docs and releases. Use when modifying, testing, benchmarking, reviewing or releasing sys1rust itself, not when installing or calling it.
---

# Develop sys1rust

Read the code you change and the write-up behind it first. `results/SPEED.md` explains every engine setting, `results/SERVER.md` the server's design, and `bench/PLAN.md` the benchmark rules.

## Build

You need an Apple silicon Mac, Rust 1.89 or newer, CMake, the Xcode command line tools and Python 3.10 or newer. Python only fetches MLX, and `sys1rust` never runs it. From the repository root:

```sh
# A prebuilt MLX 0.32.2 (the Python wheel ships libmlx and its CMake files).
python3 -m venv .mlx && .mlx/bin/pip install mlx==0.32.2
export MLX_SYS_PREBUILT_DIR="$(.mlx/bin/python -c 'import mlx.core, os; print(os.path.dirname(mlx.core.__file__))')"

# The binaries go to runtime/target/release/.
cargo build --release --manifest-path runtime/Cargo.toml
```

Every later `cargo` command needs `MLX_SYS_PREBUILT_DIR` set the same way. The build gives `sys1rust`, `sys1-bench` and `sys1-probe`. The binary loads MLX from the venv by its absolute path, so keep `.mlx/` where it is. Release bundles build with the Rust version in `packaging/rust-toolchain-version`. For benchmark work, `source bench/env.sh` instead, as the README's "Build and run inside the benchmark setup" section says. It keeps the caches and the Cargo output under `bench/`.

## Where the code lives

- `runtime/crates/laya-core` is request parsing, question checks, tokenization, sequence layout, answer decoding and the action head. It has no GPU code. It reproduces the Python `laya` package step for step.
- `runtime/crates/laya-mlx` is the forward pass on MLX through mlx-rs. Its engine settings, the `--tuning` list, are the `Knobs` in `src/lib.rs`. The Metal kernels are in `src/kernels/`, `src/split_rope.rs` (`fuserope`, `band`) and `src/nax_gemm.rs` (`nax`).
- `runtime/crates/sys1rust` is the command. `cli.rs` holds the subcommands, `config.rs` the flags and model resolution, `http.rs` the router, `validate.rs` upstream's request checks, `worker.rs` the inference thread, `agent.rs` the warm-up, `models.rs` the pinned manifest and `download.rs` the downloader.
- `runtime/crates/sys1-bench` is the adapter for `bench/harness` and `sys1-probe`, which times 2 settings against each other in one process and with `--check` compares their answers.
- `runtime/vendor/mlx-sys` is mlx-sys 0.6.0 with a build that links a prebuilt MLX.
- `bench/models.lock.json` pins the model revisions, `bench/workloads/` holds the requests and `bench/reference/` the upstream fp32 answers.

## Test

Run the suites in this order, and stop at the first failure.

1. CPU tests, with no GPU and no model:
   ```sh
   cargo test --release --manifest-path runtime/Cargo.toml -p laya-core -p sys1rust
   ```
2. The whole workspace. This adds laya-mlx's kernel tests, which run on the GPU but need no model:
   ```sh
   cargo test --release --manifest-path runtime/Cargo.toml --workspace
   ```
3. The ignored suites, which load models on the GPU. Run each alone:
   ```sh
   (
   set -e
   for t in settings loading reference; do
     SYS1_TEST_ALL_CHECKPOINTS=1 cargo test --release --manifest-path runtime/Cargo.toml \
       -p laya-mlx --test $t -- --ignored --test-threads 1
   done
   cargo test --release --manifest-path runtime/Cargo.toml -p sys1rust --test live -- --ignored
   cargo test --release --manifest-path runtime/Cargo.toml -p sys1rust --test live_pull -- --ignored
   )
   ```
   - `settings` takes about 14 minutes with all 3 models. Its exact cases, including `fuserope` alone, `band`, `nax` and the loading settings, require identical answer JSON and raw logits and pooled outputs under `f32::to_bits`. The `dense_upto`, `headprune`, `unpad`, their combinations and boolean-mask cases require the same chosen answers and allow probability, answer-confidence and action-probability differences up to 0.001. `reference` checks the answers against `bench/reference`. `live` starts the real binary and requires each reply to equal an in-process prediction byte for byte. `live_pull` downloads typed-decisions (846 MB) into an empty cache, so it needs the network.
   - The laya-mlx suites load the snapshot that a model's `refs/main` names, or else its only snapshot, and `sys1rust pull` writes no `refs/main`. Point `HF_HUB_CACHE` at a cache that holds only the pinned snapshots, filled with `sys1rust pull <model>` for each of the 3 models. Step 3 of `packaging/RELEASE.md` sets one up.
   - Without `SYS1_TEST_ALL_CHECKPOINTS=1`, a model missing from the cache is skipped with a note.
   - The `settings` tests need MLX's NAX gemms, so they pass only on an M5-class GPU (generation 17 or later) with macOS 26.2 or later and the macOS 26 MLX build.
   - `laya-core`'s `fixtures_parity` and laya-mlx's `parity` and `bench` tests need fixtures that aren't in the repository. Skip them.

## Rules

- **One GPU job at a time.** A running `sys1rust serve`, the laya-mlx tests, the live tests, `sys1-probe` and the benchmarks all use the GPU. Run each with nothing else on the GPU, and stop any server you started before the next job. 2 jobs at once slow each other and make every timing worthless.
- **Format only your own lines.** The workspace isn't rustfmt-clean, so `cargo fmt` rewrites files you didn't touch. Never run it on the whole workspace. Check a new file with `rustfmt --edition 2021 --check <file>`, which also checks the modules a `lib.rs` or `main.rs` declares. In an existing file, keep your lines in rustfmt's style and leave the rest alone.
- **Answers must not change.** A new speed change must give bit-identical results to the current default. Require identical answer JSON and raw outputs under `f32::to_bits`, using an exact case in `tests/settings.rs`. Add such a test for a new setting, and make it fail if a kernel fell back to MLX's ops without saying so. Some existing work-reduction and boolean-mask cases use the 0.001 tolerance described above. Their passing tests do not prove bit identity. A change that is meant to change numbers must say how it was checked against `bench/reference`. The correctness workload agrees with upstream on 1,498 of 1,500 answers today, and 99% is the floor. `sys1-probe --ab SPEC_A SPEC_B --check` compares 2 settings' answers.
- **Kernels fall back.** A custom kernel that can't build or run on a Mac must be found at model load, with a line on stderr, and the MLX path it replaces stays behind a setting. It never fails in the middle of a request.
- **The server matches `laya serve`.** Status codes, `detail` strings, header names, response key order and request limits match upstream. Document an intentional difference in the code and in the README's "Differences from `laya serve`".
- **Speed claims need measurements.** Name the machine, the macOS version, the MLX build, the model, the request shapes, the number of runs and processes, and report medians. Compare against a baseline measured in the same session. `sys1-probe --ab` alternates the 2 settings in one process, because separate runs on one Mac vary by about 5%. Write the result up in `results/`, as `results/SPEED.md` does. 1 run proves nothing.
- **No Python at run time.** Code under `runtime/crates` must not call Python. Build time may use the MLX wheel.
- **Bound every cache keyed by request shape**, or document why its keys are few. A long-running server sees many lengths.
- **No machine paths.** Scripts and docs must not hard-code a home directory or another machine-specific absolute path. Resolve paths from the script or the repository root.
- **Releases follow `packaging/RELEASE.md`.** Pushing a `v*` tag publishes a release, so tag only with the maintainer's go-ahead.

## Writing

- Each Rust file starts with a `//!` header that says what the file does. Comments say why, and match the surrounding code's density and naming.
- In docs, use sentence case headings, active voice and digits for numbers. Write MB as 10^6 bytes. Use no em dashes, and no colon that joins 2 clauses. Every number must trace to the code or to a write-up in `results/`.
- A commit subject is an imperative sentence in sentence case with no type prefix and no final period, such as `Stop on a leftover draft release instead of deleting it`. The body says in prose what changed, why, and how it was checked. Make one logical change per commit and stage files by path.
