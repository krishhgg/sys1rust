# Synthesis: is the gap still there, and what to build

Date: 2026-09-29. Inputs: the base-M5 bake-off (`REPORT.md`, 227 runs), the 2026-09-28 research (`../research/summary.md`), and a live check of the main repos on GitHub today.

## The gaps from the research, checked against the bench

| gap (research, 2026-09-28) | status today | what the bench adds |
|---|---|---|
| 1. A model and runtime that encode the state once and answer every question against it | Still open. kime-v1 (split state and question towers, state cache) is milestone M3 on kime's roadmap, "weeks 9 to 14", and is not shipped. kime shipped 4 releases in the last 36 hours, all training and eval tooling. ikken has no checkpoint (last push 09-28). Laya upstream: not planned (issue #49). No new repo does it. | This is the biggest lever measured. See "Why the model is the lever" below. |
| 2. A same-machine benchmark across runtimes with an agreement gate | Built by us, not published. The research found no other. | The harness is in `bench/harness/`. The run data is in `raw/`, which is not in the repository. |
| 3. A native Apple engine that routes ANE and GPU, batches across requests, and beats MPS, MLX and Core ML | Still open. No Rust runtime has shown the compiled MLX speed without MLX's tails. The three Rust ports on mlx-rs were not in the bench. | The bench changes the recipe. See "What the bench says about the Rust + MLX/ANE plan" below. |
| 4. Laya numbers on a base M5 | The research found none. Ours are not published. | `REPORT.md` |

## What the bench says about the Rust + MLX/ANE plan

These parts of the original plan do not add speed on this laptop. They change how the runtime should use each piece, not whether to build it:

- **The ANE is a low-power lane, not a speed lane.** A short question takes the same time on the ANE as on MLX fp16 (12.4 ms for typed-decisions). From 256 tokens up, the GPU is 1.4 to 3.3x faster. The ANE uses 26% less energy on short typed-decisions questions (118 against 160 mJ) and none less on multilingual.
- **Cross-request batching.** Every server did the same requests/s with 4 clients as with 1 (0.93 to 1.03x). sys1's batching lost 23%. One request already fills this GPU.
- **Rust gives packaging, not speed.** A single binary with no Python is the reason to use Rust. kime and sys1 (Rust with their own Metal code) are 3 to 6x slower than MLX. jevalaya (Rust calling Python MLX) runs at the same speed as Python MLX, with the same spikes. The speed has to come from MLX used well, whatever language calls it.
- **Lower precision.** bf16 fails the agreement gate. Core ML 8-bit weights run at the same speed as fp16 and use 11 to 17 GB of RAM on the GPU.

## The runtime gap that is left: MLX speed with MPS steadiness

MLX wins the median but has a tail problem that nobody has fixed. p50 / p95 / p99 in ms on typed-decisions, interleaved shapes:

| runtime | 1 question, 128 tokens | 1 question, 512 tokens | 10 questions, 512 tokens |
|---|---|---|---|
| laya-mlx fp16 compiled (fastest p50) | 19 / 35 / 44 | 60 / 474 / 1695 | 453 / 2053 / 2284 |
| laya-apple HTTP server (MLX) | 27 / 228 / 799 | 62 / 312 / 1140 | 525 / 939 / 1058 |
| upstream Laya, PyTorch MPS fp16 | 45 / 48 / 48 | 88 / 92 / 197 | 714 / 780 / 803 |
| cbjev, PyTorch MPS fp16 (packed questions) | 28 / 40 / 41 | 72 / 75 / 76 | 198 / 205 / 207 |

The slowest 5% of MLX requests take 2 to 18x the median. On MPS they stay within 10% of it. For a decision model in front of every agent tool call, that tail matters more than the median.

The spikes grow when the input length keeps changing. With shapes grouped instead of interleaved, plain MLX fp16 drops from 18% to 11% of requests over 2x the median, and the compiled mode with padding to 16 tokens drops to 2.5%. The likely cause is MLX specializing kernels and compiled graphs per shape. The fix to try first is a small fixed set of length buckets and batch sizes, all warmed at startup. That is what Core ML does, and Core ML and MPS have almost no spikes. This is a hypothesis. The bench did not test it. The compiled mode's throughput drop after its first 30 seconds (6.8 down to 3.3 to 4.3 requests/s) is a separate, unexplained problem.

Update, 2026-09-29: the mlx-rs spike found one cause for both, and it is not kernel specialization. When lengths vary, MLX's buffer cache grows toward the size of RAM (23 GB here), and macOS swaps. Capping the cache at 512 MiB removes the spikes and the throughput drop, in Rust and in Python. See `SPIKE.md`.

Update, 2026-09-29: the Rust server `sys1d` now serves `/v1/systemone` with that engine. HTTP adds 0.3 ms per request, and over 5 minutes it meets the bar on all three shapes (p50 18.1 / 48.4 / 453 ms). See `SERVER.md`.

Update, 2026-09-29: three exact changes (dense local attention, pruning the last head layer, no computing on padding) make the Rust runtime 1.12x faster than Python laya-mlx at its best on the timing workload, with the same answers. int8 matmuls on the MLP are the next large step and pass the agreement gate only narrowly in simulation. See `SPEED.md`.

Update, 2026-09-29: after a review (one bug fixed, the unhelpful experiments removed), those three changes are the `sys1d` default (runtime 815ecd0). All three published Laya models now run in the Rust runtime with the upstream answers: typed-decisions, multilingual and english. The changes save 8.5 to 10% of the time on each. See `SPEED.md`, stage G.

Update, 2026-09-30: round 2 of the speed work ships one custom Metal kernel, `fuserope`, for the encoder's qkv split, RoPE and unpad expand. Its answers are bit-identical to the MLX ops on all three models, it takes 0.962 of the stage G default's time in a paired run, and the Rust runtime is now 1.17x over Python laya-mlx on the timing workload. It is the `sys1d` default. Int8 matmuls were measured with a real kernel and are not shipped: 95.1% agreement on multilingual, under the gate. See `SPEED.md`, round 2.

Update, 2026-10-04: round 3 makes four exact changes the `sys1d` default: banded local attention from 512 tokens, the projections on MLX's NAX gemm loop with a load-time check, `MLX_MAX_MB_PER_BUFFER=10` and three loading settings. All 4,500 correctness answers are identical to the round 2 default's. On the target shapes it takes 0.910 of round 2's time (s128_q1 14.7 ms, s512_q1 37.6 ms), and the Rust runtime is now 1.25x over Python laya-mlx on the timing workload. Resident memory at ready falls 111 to 337 MiB. NAX's bit check runs on every load, and starts take as long as round 2's on typed-decisions and english and 0.70 of its time on multilingual. The ANE, int8, an f32 residual stream and batching were measured and not shipped. See `SPEED.md`, round 3.

How much speed a runtime alone can still find (my estimate, not measured). laya-mlx's own FLOP counts put the 421M model at about 0.75 to 0.85 GFLOP per token. The compiled MLX path then runs at about 8 TFLOP/s for one question at 128 or 512 tokens, and about 10.6 TFLOP/s for ten questions at 512. A third-party test measured this chip's fp16 matmul peak at 15.3 to 16.6 TFLOP/s. So on 128 to 512 token inputs MLX already gets half to two thirds of the peak, and a better kernel can win at most 1.5 to 2x there. Short questions (about 80 tokens) run at about 4.8 TFLOP/s. They are limited by fixed per-call overhead, but laya-mlx's own kernel work there gained only 1.03 to 1.08x.

## Why the model is the lever

Every Laya runtime pays one full pass over the state per question. Ten questions at 512 tokens is about 6,000 tokens of compute. Packing all ten into one sequence is about 1,400 tokens, roughly 4x less.

The bench confirms this. cbjev packs the questions and, on the same PyTorch MPS fp16 setup as Laya, answers ten questions at 512 tokens in 198 ms against 714 ms (3.6x). Its p99 is 207 ms. It is also as accurate or better on held-out typed-decisions rows (78.7% against 76.9%, and 76.2% against 73.5%). No runtime trick came close. The best Laya runtime is 454 ms at p50 and 2 s at p95.

cbjev is GPL-3.0, and its state encoding depends on the question set, so it cannot cache the state across requests. An Apache-2.0 model that packs the questions, with the state encoded independently of them so the state can be cached, still does not exist. Kime is building toward the same design, but CUDA-first, with T4 and L4 as its speed targets. Its Apple path is 3x slower than MLX here.

Estimate, not measured: a packed model on the compiled MLX path would answer ten questions at 512 tokens in roughly 110 to 200 ms on this laptop, against 454 ms now. The low end is the compute count at ~10 TFLOP/s. The high end is cbjev's measured time on the slower MPS path. For one question, packing changes nothing.

## Building the Rust + MLX/ANE runtime

The goal stays the same: one Rust binary, no Python, serving `/v1/systemone`, with an MLX lane and an ANE lane. The bench sets the bar and says how each lane should work.

**The bar on this laptop (typed-decisions).** p50 of 19 / 60 / 454 ms for 1 question at 128 tokens, 1 at 512 and 10 at 512 (laya-mlx compiled fp16, Python), with a p95 within about 10% of p50 like PyTorch MPS. No runtime has both today.

**Who else is there in Rust.** Three Laya ports run on mlx-rs: tjameswilliams/laya-r-mlx, andyjusa/laya-mlx-rs and zerodegress/laya-rust. All have 0 or 1 stars, none has been pushed since 2026-09-23, and none has a base-M5 number or warmed buckets. Their own M4 numbers put Rust MLX 8 to 12% ahead of Python MLX, which is the Python overhead going away. jevalaya (Rust, ANE through objc2-core-ml, MLX through embedded Python) was in the bench and was no faster than Python MLX. Ollaya binds mlx-c directly but runs fp32 only, and on multilingual it was 3.4x slower than compiled MLX fp16.

**Design the bench points to.**

- MLX lane, used for every request by default:
  - fp16 weights
  - MLX's buffer cache capped at 512 MiB, which removes the spikes and the throughput drop (`SPIKE.md`)
  - superseded: this plan first called for compiled graphs and a fixed set of length buckets and batch sizes, all warmed at startup. With the cache capped, the spike found that 12 warmed buckets added nothing and made the model load take 7.1 s instead of about 0.2 s, and that compile gained 1 to 3% in Python (`SPIKE.md`, finding 3)
  - a per-shape choice between one pass per question and one padded batch
- ANE lane through objc2-core-ml, opt-in, for single questions up to 128 tokens when power matters more than speed. laya-apple's converter (Apache-2.0) already builds a typed-decisions ANE model that passed the agreement gate here.
- One inference thread and a request queue. Cross-request batching gained nothing here, so leave it out.
- The sequence layout comes from the model config, so a packed model (below) can be added later without a rewrite.
- Every change is gated on 99% agreement with upstream Laya's fp32 answers. `bench/harness/compare.py` checks it against the saved reference answers in `bench/reference/`.

**Risks to check first.**

- mlx-rs is v0.32.0 (2026-09-12), with two maintainers. Its `build.rs` sets no macOS deployment target, and Ollaya found that a target below 26.2 leaves out the M5 Neural Accelerator kernels. If the default build lacks them, the Rust lane will be slower than Python MLX. The fallback is to bind mlx-c directly, as Ollaya does.
- MLX builds from source. It does not build with Xcode 27 (MLX issue #4533), but this laptop has Xcode 26.4.1.
- The spike fix is still a guess. The first spike tests it. (Tested: the cause was MLX's buffer cache, see `SPIKE.md`.)

**Steps.**

1. **Spike (1 to 2 days).** Run the typed-decisions forward pass on mlx-rs in fp16 with compile and warmed buckets. Confirm the Neural Accelerator kernels are in the build, pass the agreement gate, then run the short, timing, tail and 5-minute stages. Decision point: does it reach the bar, p50 and p95? (Done, see `SPIKE.md`. With an fp16 bug fix and the capped buffer cache, and no compile or warmed buckets, the Rust runtime beat the bar for one question and came within 5% of it for ten questions at 512 tokens.)
2. **Server and ANE lane (days).** `/v1/systemone`, the queue, the ANE lane through objc2-core-ml, and a router rule based on this bench (ANE only for short single questions in low-power mode).
3. **Later, the model (weeks).** A packed or encode-once checkpoint (see "Why the model is the lever") multiplies what the runtime can do for multi-question requests. Start from OpenDecider-nano or laya-typed-decisions with ikken's block mask. The typed-decisions train split (1,200 states, Apache-2.0) is public, and kime's data converters list 2.2M questions with a license manifest. A small proof fits on this laptop. A full fine-tune needs a rented GPU.
4. **Optional: publish the benchmark.** Your call, because it is public.

The unknown that decides how much step 3 is worth is how many questions a real request carries. If most traffic is one short question, packing gains nothing, and steps 1 and 2 are the whole win.

## Sources for the live check (2026-09-29)

- kime commits and releases v0.1.3 to v0.1.5, and `spec/16-roadmap.md` M3: github.com/tamnd/kime
- ikken last push 2026-09-28: github.com/johnmofficial16-prog/ikken
- laya-mlx last push 2026-09-22, laya-apple 2026-09-27, cbjev 2026-09-24 (GPL-3.0), Laya upstream 2026-09-27
- GitHub search for Laya repos created since 2026-09-27: 20 results, mostly apps (agent guards, routers, triage), none about encode-once
- Base M5 fp16 matmul peak: research/report-c-apple-silicon.md, citing scratchy issue #118
- laya-mlx FLOP counts and kernel results: `raw/laya-mlx-docs/MATH_10X_RESEARCH.md`, `ENGINEERING_10X_RESEARCH.md`
