# Faster than Python MLX: results (2026-09-29)

`SERVER.md` found that Python laya-mlx, with MLX's buffer cache capped, runs as fast as the Rust runtime. Both call the same MLX kernels. This round looked for speed that Python laya-mlx does not have, without changing the answers.

## Short answer

- Three exact changes make the Rust runtime 1.12x faster than Python laya-mlx at its fastest (compiled, cache capped) on the timing workload. The runtime is faster on all 12 shapes, by 4 to 36%. Answers do not change: 1,498 of 1,500 agree with the reference, as before.
- On short single questions (69 to 92 tokens) the two tie. The Rust p50 was 11.6 to 12.2 ms and Python's 12.6 to 13.1 over three alternating runs.
- The three changes:
  - dense attention in the local layers
  - the last decision-head layer computed only where the scorer reads it
  - no computing on padding
- MLX's matmuls already run at 13 to 14 TFLOP/s on most of the model's shapes, about the published fp16 peak of this chip. Kernel settings, split-K and RoPE layout changes gained nothing.
- The one route to a much larger gain is int8 matmuls, which run about 2x fp16 on the M5 GPU. A simulation shows int8 on the MLP matmuls alone passes the 99% agreement gate, but narrowly (99.3%). It needs custom Metal kernels, because MLX has no int8 matmul.
- Round 2 (2026-09-30, below): one custom Metal kernel for the encoder's qkv split, RoPE and unpad expand. Bit-identical answers, 0.962 of the stage G default's time in a paired run, and 1.17x over Python laya-mlx in the two-round harness. It is the `sys1d` default. Int8 was measured with a real kernel and is not shipped: it fails the agreement gate on multilingual.
- Round 3 (2026-10-04, below): banded local attention from 512 tokens, the projections on MLX's NAX gemm loop, `MLX_MAX_MB_PER_BUFFER=10` and three loading settings. Answers are bit-identical. On the three target shapes the new default takes 0.910 of the round 2 default's time, process-paired: s128_q1 14.7 ms and s512_q1 37.6 ms. In the two-round harness it is 1.25x over Python laya-mlx. It is the `sys1d` default. `nax` checks its kernels on every load. Typed-decisions and english start as fast as with round 2 (217 against 216 ms on typed-decisions), and multilingual in 0.70 of round 2's time.

## Conditions

- AC power, load average 1.5 to 4.2. Runtime code is e758bfa plus the changes later committed as 0495800.
- Paired comparisons: `sys1-probe --ab` loads two engines in one process and alternates requests between them, so drift in the machine hits both. Separate runs on this laptop vary by about 5%, which is too much to see a 3% change.
- Harness comparisons against Python ran in two rounds in opposite order (stage E).

## Where the time goes

Cost per token is flat from 184 to 6,000 tokens (95 to 111 µs), so even one short question fills the GPU.

- **Matmuls.** At 10 rows of 568 tokens they take about 70% of the forward. Measured alone at the model's shapes, most run at 13 to 14 TFLOP/s. The exceptions are the two projections with a long inner dimension and 1,024 outputs (encoder MLP output, K 2624, and head FFN output, K 4096). They run at 6.7 to 10.7 TFLOP/s, because 1,024 outputs make few tiles for the GPU's 10 cores.
- **Everything else.** Local attention is 9%, RoPE with its copies 6%, GeGLU 6% and LayerNorms 5%.
- **Padding.** On multi-question requests every row pads to the longest. That is 29% of tokens at 10 questions over a 64-token state, 15% at 128, 11% at 256 and 6.4% at 512.

## What was tested

Paired A/B against the current default (`f16gelu,cache=512,wired=2048`), timing workload, 10 pairs per shape. Ratios below 1 are faster.

| change | setting | geo-mean time ratio | range over shapes | answers |
|---|---|---|---|---|
| dense local attention up to 1,024 tokens | `dense_upto=1024` | 0.968 | 0.91 to 1.01 | identical |
| last head layer only at scorer positions | `headprune` | 0.983 | 0.96 to 1.02 | max diff 0.0001 |
| no computing on padding | `unpad` | 0.953 | 0.79 to 1.02 | identical |
| all three | `dense_upto=1024,headprune,unpad` | **0.899** | 0.77 to 0.99 | max diff 0.0001, no answer changes |
| split rows into groups by length | `split=auto` | 0.970 (smoke) | | score and noul differ in the 4th decimal |
| one RoPE call for q and k | `rope1` (3 layouts) | 1.00 to 1.06 | | identical |
| split-K for the K 2624 and K 4096 projections | `splitk=2`, `splitk=4` | 1.00, 1.01 | | score differs in the 4th decimal |

- **Dense attention** wins at every length, including rows of 1,024 tokens (0.95 to 0.98). The chunked path was a leftover from laya-r-mlx. Python laya-mlx already uses dense masks, so this change only catches up with it.
- **Unpadding** keeps hidden states as one packed list of real tokens through the embeddings, LayerNorms, matmuls and GeGLU. It moves them into the padded layout only for attention. It pays most where padding is highest: 0.79 at 10 questions over a 64-token state, and about 1.0 at 512 tokens.
- **Head pruning.** The scorer reads only the CLS token and the option markers of the last head layer. That layer now runs its queries, output projection and FFN on those rows only. It saves about 2.8% of the FLOPs.
- **Split-K** was 1.3 to 2.2x faster on the two slow projections in a microbenchmark at 184 and 606 tokens, and slower from 2,048 up. In the model it gained nothing. The extra reshape, batched matmul and sum cost what the matmul saved.
- **MLX settings with no code change.** Raising or lowering the command-buffer limits (`MLX_MAX_OPS_PER_BUFFER`, `MLX_MAX_MB_PER_BUFFER`) and `MLX_METAL_FAST_SYNCH` stayed within noise. `MLX_METAL_GPU_ARCH=applegpu_g17s` selects the bigger chips' matmul tile. It was 3% faster at 606 tokens and 7% slower at 5,680.

## Against Python laya-mlx

Stage E, two rounds each, p50 ms. "Rust new" is `f16gelu,cache=512,wired=2048,dense_upto=1024,headprune,unpad`.

| shape | Python laya-mlx compiled, cache capped | Rust current default | Rust new | Python / Rust new |
|---|---|---|---|---|
| 1q, 64-token state | 21.0 | 18.9 | 19.1 | 1.10 |
| 4q, 64 | 62.1 | 58.3 | 50.8 | 1.22 |
| 10q, 64 | 144.0 | 132.5 | 105.6 | 1.36 |
| 1q, 128 | 18.9 | 17.3 | 17.6 | 1.08 |
| 4q, 128 | 68.1 | 61.1 | 61.1 | 1.12 |
| 10q, 128 | 164.6 | 150.2 | 142.0 | 1.16 |
| 1q, 256 | 28.5 | 28.7 | 26.8 | 1.06 |
| 4q, 256 | 103.6 | 105.3 | 98.9 | 1.05 |
| 10q, 256 | 258.2 | 268.1 | 235.2 | 1.10 |
| 1q, 512 | 49.4 | 48.2 | 45.6 | 1.08 |
| 4q, 512 | 173.2 | 174.3 | 166.1 | 1.04 |
| 10q, 512 | 453.2 | 453.0 | 422.1 | 1.07 |
| **geo-mean** | 83.7 | 80.3 | 74.9 | **1.12** |

- The mean request took 117 ms against Python's 130, so one client gets about 11% more requests per second.
- p95 at 10 questions over 512 tokens was 466 to 471 ms against Python's 496 to 518. No request in any run took over 2x its shape's median.
- The correctness run with the new settings agrees with the reference on 1,498 of 1,500 answers (99.9%), with gold accuracy 75.8%. Both match the current default.
- Short workload (40 requests, 15 passes, alternating runs): Rust new 11.64, 12.22 and 11.84 ms p50. Rust current default 11.82 and 11.68. Python 13.10 and 12.58. At about 80 tokens the forward is 11.1 to 11.4 ms of GPU time plus 0.2 ms to build the graph, in both.

## Int8 matmuls

MLX has no int8 × int8 matmul. Its quantized matmuls turn the weights back into fp16 first, so they cannot speed up matmuls that are limited by compute, as these are. Published measurements for the base M5 put int8 × int8 at 29.5 TOPS against 14.2 TFLOP/s for fp16.

To check accuracy before writing kernels, `fakeq_adapter.py` simulates int8 in Python laya-mlx. It rounds the weights to int8 per output channel and, for W8A8, the activations to int8 per token, then computes in fp16. Correctness workload, 1,500 answers:

| simulated | agreement | gold accuracy | gate (99%) |
|---|---|---|---|
| none (fp16) | 99.9% | 75.8% | pass |
| int8 weights, all 120 matmuls | 99.5% | 76.0% | pass |
| int8 weights and activations, all 120 matmuls | 98.5% | 75.9% | fail |
| int8 weights and activations, encoder MLP only (56 matmuls, 66% of matmul FLOPs) | 99.3% | 75.4% | pass |

Weight-only int8 passes but brings no speed on this GPU. The MLP-only W8A8 variant is the one that could pay off. If its matmuls ran at 1.8 to 2x, the forward would be about 1.2 to 1.3x faster. That estimate is not measured. It needs an int8 matmul kernel for the M5's matmul hardware, a kernel that turns activations into int8, and a check with more data than one correctness workload, given the thin margin.

## What the changes cost

- **Answers.** Nothing. The skipped work produced values nothing reads. Padding positions are masked out as keys and their outputs are dropped, and `score` reads the last head layer only at the CLS token and the option markers. Dense attention does more arithmetic than the chunked path, but in fewer, larger GPU calls, and its masked scores come out of the softmax as exact zeros. Head pruning moves probabilities by at most 0.0001, because matmuls with fewer rows can add in a different order in fp16. No answer changed.
- **Memory.** Nothing. On 1,024-token inputs (`long.jsonl`, battery), MLX peak memory was 1,671 MB with the new settings against 1,765 MB with the old ones, and the process footprint 2,379 MB against 2,458 MB. In stage E the footprint was 2,382 to 2,472 MB against 2,478 to 2,505 MB.
- **Speed on some shapes.** Unpadding alone was up to 2% slower where padding is low, because the copies in and out of the padded layout around attention cost more than the padding saved. With all three settings on, no shape was slower.
- **An assumption.** Head pruning is only correct while nothing reads the other positions of the last head layer. A future model or API feature that does, such as per-token outputs or embeddings, must turn it off. Since the review, the pruned output is its own type (`ScorerRows`), which only hands out the CLS and marker rows, so code that wants every position cannot read the wrong rows by mistake.

## Review, new default and the other Laya models (stage G)

A second agent reviewed the code (commit 0495800) line by line before the settings became the default.

- **One bug fixed.** The runtime kept one dense window mask per distinct input length and never freed them. With dense attention up to 1,024 tokens, a long-running server could collect about 700 MB of masks. It now keeps one mask for the longest length seen and uses its top-left corner for shorter inputs. A unit test checks that the corner equals a mask built for the shorter length.
- **Removed.** `split`, `rope1` and `splitk` gave no gain and are gone, about 600 lines. They stay in git history at 0495800.
- **New tests.** `crates/laya-mlx/tests/settings.rs` compares each setting, and all three together, against the old path on every cached Laya model. It covers:
  - 1, 2 and 20 options;
  - an input cut at the length limit;
  - heavy padding;
  - mixed answer types;
  - a request with no padding;
  - the smoke workload.
  All pass. The largest probability change is 0.0005 (multilingual, head pruning), with no answer changed.
- **Default.** `sys1d` now runs `f16gelu,cache=512,wired=2048,dense_upto=1024,headprune,unpad` (round 2 adds `fuserope`, round 3 adds `band=512,nax=all,directload,sharehead,parallel_load` and `MLX_MAX_MB_PER_BUFFER=10`, below). `sys1-bench` has a new variant, `mlx-fp16-lean`, with the same settings; `mlx-fp16-fast` keeps its old meaning. The `http-fp16-fast` bench variant uses the server default, so it now runs the new settings.

All three published Laya models now run in the Rust runtime. Before this round only typed-decisions had been tried. Stage G, on the reviewed build, battery power:

| model | agreement with upstream reference | paired time, new / old (geo-mean) | range over shapes |
|---|---|---|---|
| typed-decisions | 1,498 of 1,500 on correctness, in-process and over `sys1d` HTTP (as before) | 0.900 | 0.78 to 0.99 |
| multilingual (mmBERT, 322M) | 100% on correctness, smoke and short | 0.915 | 0.82 to 0.99 |
| english (base Laya) | 100% on smoke, short and cold (no correctness reference); against the old settings, 100% on correctness | 0.904 | 0.77 to 1.00 |

- The reviewed build gives byte-identical answers to the pre-review build on the typed-decisions correctness workload.
- In the paired runs, `--check` counted 3 score mismatches on multilingual. It counts any change in a score's value, and these differ by at most 0.0002. The harness rounds scores before comparing, and there multilingual agrees 100%.
- On multilingual, one question at 128 tokens takes 7.7 ms.

## Round 2: the split, RoPE and unpad expand as one Metal kernel (2026-09-30)

Round 1 left RoPE and its copies at 6% of the forward. In every encoder layer MLX ran `split_equal`, three reshape copies into `[n, H, len, hd]`, two `fast::rope` launches and, with `unpad`, the gather that expands the packed rows. A research round timed custom Metal kernels through the C API behind `mx.fast.metal_kernel` (mlx-rs 0.32 has no binding for it; `metal_kernels.rs` is a small wrapper over the vendored mlx-sys). One kernel ships, as the setting `fuserope`.

- **What it does.** One launch per layer reads the qkv rows and writes roped q, roped k and v in `[n, H, len, hd]`. With `unpad` it reads each padded position's row through the packing's index, so the expand costs nothing extra. It keeps MLX's rope arithmetic (exp2 of the log2 base, computed on the host as f32, `fast::cos` and `fast::sin` in float), so its output equals `fast::rope`'s bit for bit. One thread handles 4 pairs with 4-wide loads and stores, so the head dim must be a multiple of 8; all three models have 64.
- **Gain.** Paired A/B (`sys1-probe --ab`) against the stage G default on typed-decisions, 12 timing shapes, 5 pairs each after 2 warmups, AC power: geo-mean 0.962, range 0.928 to 1.005. Single questions gain nothing (0.994 to 1.005); 4 and 10 questions gain 4 to 7%. Paired runs on multilingual and english with 1 warmup and 2 pairs per shape: 0.948 and 0.964.
- **Answers.** On all three models, `fuserope` alone is byte-identical to the plain path on every state of the equivalence suite: `fuserope_matches_plain` in `tests/settings.rs` asserts the same answer JSON for every state and equal raw logits and pooled outputs. `sys1-probe --check` of the stage G default against the new one on typed-decisions, multilingual and english: largest probability difference 0.000000, 0 of 60 answers changed, on each. The correctness run with the new default agrees with the reference on 1,498 of 1,500 answers (99.9%), gold accuracy 75.8%, as before.
- **macOS 14 and 15.** MLX compiles the kernel from source on the user's machine. The same `sys1d` binary with `DYLD_LIBRARY_PATH` set to the macOS 14, 15 and 26 builds of MLX 0.32.2 compiled the kernel and passed its load check on each, and the new default gave the same answers as the old one on each build (0 choice flips, largest probability difference 0.0000 over the 240 timing requests). The 14 and 15 builds differ from 26 by 0.0056 with either default, as measured before this round: they have no kernels for this chip's matmul hardware and run about 3x slower. On the 26 build the sum of per-shape p50 went from 1,315 to 1,218 ms (0.93); on 14 and 15 from 4,030 to 3,962 ms (0.98), since the kernel's share is smaller where the matmuls are 3x slower.
- **Fallback.** The decision is made at load, never in a request. laya-mlx builds the kernel and runs it once on a small packed and an unpacked batch against the MLX ops. If the kernel does not compile, does not match, the backend is CPU, or the head dim is not a multiple of 8, it prints one line to stderr and the forward runs the MLX ops.
- **Tried and ruled out.**
  - A merge-side kernel (attention output back to the packed `[T, d]` rows for the output projection, with the pack gather fused): 1.02 to 1.05x slower. MLX's attention already writes its output in `[n, len, H, hd]` memory order, so the transpose it was meant to replace is a free view.
  - 1 pair per thread instead of 4: no measured difference. The 4-wide variant ships.
  - Int8 matmuls, measured with a real kernel on the MLP input projection (int8 weights and activations): 1.39x over Python on typed-decisions, but agreement with the reference fell to 99.6% there and to 95.1% on multilingual, under the 99% gate. Not shipped. The answers-unchanged rule holds for every model, and the simulation of round 1 (99.3% on typed-decisions) did not predict the multilingual result.

Stage E style, two rounds in opposite order, p50 ms. "Old default" is the stage G default, "round 2 default" adds `fuserope`.

| shape | Python laya-mlx compiled, cache capped | Rust old default | Rust round 2 default | Python / Rust round 2 |
|---|---|---|---|---|
| 1q, 64-token state | 22.1 | 18.5 | 18.4 | 1.20 |
| 4q, 64 | 59.6 | 48.7 | 46.1 | 1.29 |
| 10q, 64 | 138.4 | 101.1 | 93.4 | 1.48 |
| 1q, 128 | 18.9 | 16.7 | 16.6 | 1.14 |
| 4q, 128 | 62.1 | 58.3 | 55.6 | 1.12 |
| 10q, 128 | 157.5 | 135.5 | 126.3 | 1.25 |
| 1q, 256 | 27.2 | 25.1 | 24.8 | 1.09 |
| 4q, 256 | 96.7 | 94.2 | 88.9 | 1.09 |
| 10q, 256 | 245.4 | 224.2 | 211.7 | 1.16 |
| 1q, 512 | 46.3 | 43.9 | 44.1 | 1.05 |
| 4q, 512 | 165.3 | 158.7 | 150.1 | 1.10 |
| 10q, 512 | 421.8 | 402.5 | 379.8 | 1.11 |
| **geo-mean** | 80.1 | 71.5 | 68.6 | **1.17** (range 1.05 to 1.48) |

The old default measured 1.12x over Python in the same session (range 1.03 to 1.37), as in stage E.

## Round 3: banded attention, NAX gemms, a buffer limit and faster loading (2026-10-04)

Research round 2 (2026-10-01 to 10-03) found four exact changes. Round 3 ships them as the `sys1d` default. The settings string adds `band=512,nax=all,directload,sharehead,parallel_load` to the round 2 default, and `sys1d` sets `MLX_MAX_MB_PER_BUFFER=10`. Each change gives the same raw logits, pooled outputs and answers as the round 2 default, bit for bit, on all three models. The research gains come from the round 2 reports. Every other number comes from this round's runs of the committed code, on typed-decisions unless a model is named. Every timing run used AC power. Three correctness processes ran on battery: multilingual with the round 3 default and english with both defaults. Their answers and peak resident memory are below, and no timing comes from them.

### The changes

- **`band=512`.** From a padded length of 512 tokens up, the local layers attend by chunks of 64 queries against the 192 keys each chunk's window can reach. Before, they used a dense len x len window mask. The `fuserope` kernel writes q, k and v straight into the chunk layout. In research it gave 0.967 at s512_q1 (19 of 20 rows faster) and 0.991 over the 12 shapes. Below 512 tokens it runs the same code as before. From 256 tokens it was slower (s256_q4 1.014, s256_q10 1.026), so the threshold is 512.
- **`nax=all`.** The encoder's 4 projections and the decision head's 2 residual products run on MLX's own gemm loop for the M5's matmul units (NAX), with launches this runtime chooses. The regular gemm uses 64 x 64 tiles up to 1,024 rows and runs row tiles that share a weight tile back to back. wo2's split K runs in one launch instead of two, with 64 x 64 tiles up to 320 rows and 64 x 128 above. Each output element is the same sum in the same order as in MLX's gemm, so the bits are the same. The loop comes from 5 MLX 0.32.2 headers, copied verbatim with MLX's MIT license. In research it gave 0.956 over the 12 shapes and 0.89 to 0.95 up to 774 tokens.
- **The `nax` guards.** Two checks run at every load. A device gate copies MLX's own NAX condition: macOS 26.2 or later and a GPU architecture generation of 17 or more. A bit check runs each kernel against MLX's matmul or addmm at every weight shape of the model. Its row counts reach every kernel variant the shape can use, each with a cut row tile and with every row tile full, and both sides of each change of MLX's route: 31 gemms on typed-decisions and english, 23 on multilingual. If either check fails, `laya-mlx` prints the reason and the model runs MLX's gemms. If they pass, it prints one line with the check's gemm count and time. Nothing is decided during a request, and nothing is written to disk.
- **`MLX_MAX_MB_PER_BUFFER=10`.** MLX commits a command buffer to the GPU once the inputs added to it pass this limit. The default is 40 on this M5 and 50 on Max and Ultra chips. MLX adds element counts, not bytes, so 10 is a nominal value, not 10 MB. In research, on top of the other two changes, it gave s512_q1 0.931 and s512_q10 0.981. At s128_q1 its two rounds disagreed (1.009 and 0.938).
- **The loading settings.** `directload` copies each f16 tensor into MLX straight from the file mapping, without a temporary copy. `sharehead` makes the pruned head layer's q and k|v projections views of its full projection, which saves 12.0 MiB of MLX memory (6.8 MiB on multilingual). `parallel_load` loads the tokenizer on a second thread while the first thread opens the weights and builds the backend. In research, without `nax`, the three made a model ready in 0.673 of the time (0.610 to 0.754 per model) with 127 to 388 MiB less resident memory.

### Setting the buffer limit in-process

MLX reads `MLX_MAX_MB_PER_BUFFER` once, when the first GPU operation creates the Metal device. `sys1d` sets it to 10 first thing in `main`, before it starts a thread or calls MLX, unless the user has set it. It logs one line when it does. The `mlx-fp16-lean` variant of `sys1-bench` does the same and records the value in its meta line. A library injected for this check printed the device's limit at exit: 10 with the variable unset, 25 with the user's 25, and 40 for the round 2 `sys1d`.

A paired check through `sys1d` HTTP at s512_q1 compared three processes of the new `sys1d`: A set 10 itself, B was started with 10, and C was started with 40. Each process ran 20 rows, with 2 warm-ups and 5 requests per row, over 4 rounds in rotating order. A/B was 1.000 over all rounds (per round 1.017, 0.989, 1.002 and 0.994), so the in-process value acts like one set before the start. A/C was 0.959, with all 20 rows faster in rounds 2 to 4. All 12 processes gave the same answers. Load average was 6 to 8 in round 1, from other work on the laptop.

### Gain

In one process: `sys1-probe --ab --all-rows --check`, the round 2 default against the round 3 settings, all 240 timing rows, 2 warm-ups and 5 pairs per row. Both sides share one process, so both run with MLX's default buffer limit, and this table leaves out the gain from `MLX_MAX_MB_PER_BUFFER=10`. Medians over each shape's 20 rows, in ms:

| shape | round 2 default | round 3 settings | ratio |
|---|---|---|---|
| s64_q1 | 19.67 | 19.25 | 0.979 |
| s64_q4 | 51.28 | 47.27 | 0.922 |
| s64_q10 | 96.53 | 93.66 | 0.970 |
| s128_q1 | 18.62 | 17.78 | 0.955 |
| s128_q4 | 62.60 | 58.81 | 0.939 |
| s128_q10 | 129.83 | 128.67 | 0.991 |
| s256_q1 | 27.47 | 24.78 | 0.902 |
| s256_q4 | 98.41 | 95.99 | 0.975 |
| s256_q10 | 216.49 | 214.91 | 0.993 |
| s512_q1 | 48.41 | 43.48 | 0.898 |
| s512_q4 | 166.29 | 164.12 | 0.987 |
| s512_q10 | 392.29 | 385.05 | 0.982 |
| **geo-mean** | | | **0.957** (range 0.898 to 0.993) |

229 of the 240 rows were faster, and the geo-mean over rows is 0.962. Load average was 8.9 at the start. On the first row of each shape, with the same method, multilingual gave 0.956 (0.900 to 1.044) and english 0.971 (0.918 to 1.015).

Process-paired, with the buffer limit (opus2's `procab`): A ran the round 2 default with MLX's default limit, and B the round 3 default with the limit at 10. Two rounds, A B then B A, 20 rows per shape, AC at the start and end of every process. Medians over each shape's rows, both rounds, in ms:

| shape | round 2 default | round 3 default | ratio | rows faster | round 3 rows under the target |
|---|---|---|---|---|---|
| s128_q1 | 16.51 | 14.73 | 0.892 | 20 of 20 | 16 of 20 under 15 ms |
| s512_q1 | 43.44 | 37.61 | 0.866 | 20 of 20 | 19 of 20 under 40 ms |
| s512_q10 | 390.62 | 381.07 | 0.976 | 20 of 20 | |

The geo-mean over the three shapes is 0.910. By round, s128_q1 gave 0.896 and 0.890, s512_q1 0.867 and 0.863, and s512_q10 0.979 and 0.969. MLX's peak memory in these processes fell from 1,635 to 1,281 MiB.

Against Python laya-mlx, stage E style: two rounds in opposite order, warm-up 12, 2 repeats, p50 in ms. "Round 2 default" is the old default here. The round 3 side ran the `mlx-fp16-lean` variant, which sets the buffer limit as `sys1d` does.

| shape | Python laya-mlx compiled, cache capped | Rust round 2 default | Rust round 3 default | Python / Rust round 3 |
|---|---|---|---|---|
| 1q, 64-token state | 20.9 | 18.8 | 17.5 | 1.19 |
| 4q, 64 | 60.8 | 47.1 | 41.4 | 1.47 |
| 10q, 64 | 140.2 | 94.9 | 90.2 | 1.55 |
| 1q, 128 | 19.0 | 17.1 | 15.5 | 1.23 |
| 4q, 128 | 62.8 | 56.3 | 50.7 | 1.24 |
| 10q, 128 | 159.5 | 128.3 | 125.2 | 1.27 |
| 1q, 256 | 27.2 | 25.4 | 22.2 | 1.23 |
| 4q, 256 | 97.9 | 89.4 | 86.2 | 1.14 |
| 10q, 256 | 250.0 | 214.5 | 209.1 | 1.20 |
| 1q, 512 | 47.6 | 44.9 | 37.9 | 1.26 |
| 4q, 512 | 168.3 | 152.9 | 148.3 | 1.13 |
| 10q, 512 | 430.2 | 386.5 | 378.6 | 1.14 |
| **geo-mean** | 80.8 | 69.8 | 64.8 | **1.25** (range 1.13 to 1.55) |

The round 2 default measured 1.16x over Python in the same session (range 1.06 to 1.48), and the round 3 default took 0.93 of its time (1.077x, range 1.02 to 1.18). The geo-means of each round agree within 0.1 ms for all three. In these runs the peak resident memory was 991 to 993 MiB for Python, 1,814 MiB for the round 2 default and 1,687 to 1,688 MiB for round 3.

### Answers

- **Equivalence tests.** `stack_matches_plain` in `tests/settings.rs` runs the round 3 settings against the round 2 default on every state of the suite: 40 states on typed-decisions and multilingual, 37 on english. It runs them with `band=1` too, so every state takes the band path. Every answer JSON is byte-identical, and the raw logits and pooled outputs compare equal with `f32::to_bits`. Other tests run `band=1` and `band=512` alone and on the round 2 default (with states of 511, 512 and 513 tokens and two-row batches padded to 512 and 600), `nax=all` against the plain path, and each loading setting alone. A test fails if a kernel fell back without saying so (`Backend::active_kernels`). `tests/loading.rs` checks that a missing, empty or cut weights file is an error naming the file, with each loading setting. With `SYS1_TEST_ALL_CHECKPOINTS=1`, a test fails unless all three models load and run. All 16 tests in `tests/settings.rs`, the one in `tests/loading.rs` and the two in `tests/reference.rs` pass that way.
- **Paired run.** The in-process A/B above checked all 1,200 answers: largest probability difference 0, no answer changed.
- **Correctness workload.** 300 requests and 1,500 answers per model, round 2 default against round 3 default (`sys1-bench` in the harness, round 3 with the buffer limit at 10). All 4,500 answers are identical JSON, with 0 choice changes and a largest difference of 0. Against the upstream reference, both agree on 1,498 of 1,500 on typed-decisions (99.9%, the same 2 near-ties as before) and on 100% of multilingual. Gold accuracy is 75.8%, 38.3% and 38.2% for both. english has no reference.

### macOS 14 and 15

Same method as round 2: the same `sys1d` binaries with `DYLD_LIBRARY_PATH` set to the macOS 14, 15 and 26 builds of MLX 0.32.2. Each process warmed up on 2 requests per shape, then answered the 240 timing requests once. The times are one unpaired pass per process.

| MLX build | `nax` at load | printed reason | answers, round 3 against round 2 on the same build | sum of per-shape p50, round 2 / round 3 (ms) |
|---|---|---|---|---|
| macOS 26 | on | none | 240 of 240 identical | 1,387 / 1,347 |
| macOS 15 | off | the kernel differs from MLX's matmul (16 x 1024 by 3072, max diff 0.0009765625) | 240 of 240 identical | 3,978 / 3,891 |
| macOS 14 | off | the same | 240 of 240 identical | 3,997 / 3,891 |
| macOS 26, `MLX_METAL_GPU_ARCH=applegpu_g16s` | off | MLX does not use NAX on this GPU (architecture `applegpu_g16s`) | 240 of 240 identical | 4,019 / 3,966 |

- The device gate reads the running macOS (26.2 here) and the GPU, so on this Mac it passes with every build. The macOS 14 and 15 builds of MLX have no NAX kernels, so their matmuls give other bits. The bit check finds this on the first weight shape, and the load falls back. Both builds differ from the 26 build by up to 0.0056 in probability, with either default, as in round 2.
- The architecture override makes MLX itself skip NAX, and the device gate falls back with its own reason.
- The printed reasons come from the startup round's build. The first round 3 build printed max diff 0.001953125: the check's random inputs changed since.
- No real macOS 14 or 15 machine was tested. There the gate should fall back before the bit check, with "MLX uses NAX from macOS 26.2". That is read from the code, not measured.

### Startup and memory

`sys1d` round 2 against round 3: fresh processes, five rounds per model with the order rotating, medians. "Ready" is the time from spawn to the ready line. The round 3 build still had code that saved a passed check, since removed. It had no saved record, so every round 3 load ran the `nax` check, as every load does now. Those loads also built a record key, looked for a record and saved one after the check. The round 3 code does none of these. macOS's shader cache held the kernels in every run (below). Load average 4.0 to 4.1.

| model | run | ready, ms | against round 2 | load_ms | warm-up, ms | resident memory at ready, MiB |
|---|---|---|---|---|---|---|
| typed-decisions | round 2 | 216.3 | | 169 | 42 | 1,010 |
| | round 3 | 216.7 | 1.00 | 173 | 39 | 899 |
| multilingual | round 2 | 417.4 | | 382 | 30 | 1,536 |
| | round 3 | 293.8 | 0.70 | 269 | 26 | 1,200 |
| english | round 2 | 220.7 | | 172 | 42 | 1,012 |
| | round 3 | 216.8 | 0.98 | 173 | 39 | 900 |

- Resident memory at ready falls by 111 to 337 MiB.
- The same run also started round 3 without `nax`, for reference: 179 ms on typed-decisions and english and 294 ms on multilingual. So the loading settings alone make typed-decisions ready in 0.83 of round 2's time here (0.75 in the first session; research measured 0.673), and `nax` adds about 38 ms to a start on typed-decisions and english.
- Three such runs were made, all on builds with the saved-pass code. The table is the third. The first gave every ratio within 0.02 of these. The second spread up to 2x within a line, likely from other work on the laptop, so it is left out.
- The check takes 40 to 43 ms in `sys1d` on typed-decisions and english and 17 to 30 ms on multilingual, as the load line reports. On multilingual `parallel_load` parses the tokenizer (about 280 ms, research round 2) next to the weight load and the `nax` work, which hides the check there.
- The first round 3 build started typed-decisions in 321 ms against 240 for round 2, in another session, so `nax` added about 120 ms. A probe of each step in fresh processes on typed-decisions found where it went (medians of 5 or 6, load average 7.7 to 8.8):
  - the device gate, the kernel header and the 3 kernel objects: 0.2 ms
  - the 10 pipelines, compiled by their first launches: 9 ms in all
  - the check, 105 ms: 53 gemms, each with its own random activations and its own wait for the GPU. Its MLX gemms took 29 ms and its kernel gemms 28 ms. The same comparisons on ready inputs, in one MLX graph, took 46 ms.
- Three changes cut the check without narrowing it. It runs 31 gemms instead of 53, at row counts chosen per weight shape. The old counts never filled every row tile of MLX's 128-row tile, so MLX's `align_M` variant was not compared, and they did not take both sides of MLX's split-K limit at 1,365 and 1,366 rows for K 4,096. The new counts do both. The gemms share their inputs: one weight and one activation per K and one residual per N, each gemm taking the leading rows. They wait for the GPU once for the inputs and once per 8M output elements, 4 times in all on typed-decisions. In a fresh process the check went from 105 to 47 ms (median of 5, range 46 to 52), with MLX's peak memory at 145 MB during it.
- macOS caches compiled Metal shaders by executable path and source. A new path or new kernel source makes the first start compile the 10 pipelines, about 1.7 s (0.12 to 0.23 s each). That cost predates this change. After it, macOS's cache serves them in about 5 ms.
- The buffer limit does not change startup: 238.1 ms at 10 and 239.7 ms at 40, with the round 2 settings.
- Peak resident memory in the correctness runs fell from 1,815 to 1,687 MiB on typed-decisions, 2,156 to 1,767 on multilingual and 1,820 to 1,688 on english. The multilingual round 3 process and both english processes ran on battery.

### Tried in research round 2 and not shipped

- **Int8 matmuls.** Every scheme tested fails the quality gate. On the MLP input projection, typed-decisions flips 6 answers against fp32 where the default flips 2, and moves one probability by 0.041 against a cap of 0.01. Multilingual agrees with the default on 95.1% of answers. SmoothQuant, 32 fp16 outlier channels, both together, the first 13 layers only and the other projections also fail, with largest shifts of 0.014 to 0.066. A per-layer choice by sensitivity and per-group activation scales were not tried. The M5's matmul units have no fp8 type in the macOS 26.2 or 26.4 SDK.
- **The ANE.** It is not faster for one request. Its typed-decisions encoder pass for one 151-token row took 14.6 to 15.1 ms, while the GPU answered the whole request in 14.3 to 14.8 ms. Past about 160 tokens it falls further behind. Next to the GPU it could add encoder capacity, 1.71 to 1.89 times the items per second of the GPU alone, but that is an estimate, not measured answers per second. It would need a Core ML binding and one compiled model per length bucket, each about 750 MB and 4 to 10 s to compile. Its quality was checked on typed-decisions only, where its mean drift is 3.4 times the default's.
- **f32 residual stream (`f32res`).** It costs time: 1.142 of the default's (1.056 to 1.207). It cuts the mean drift from fp32 by 16 to 23% and removes typed-decisions' 2 near-tie flips. But it changes answers: multilingual gains a flip against fp32 and moves a probability by 0.0115, and english moves by up to 0.0372. The `addnorm` kernel only helps this path, so neither ships.
- **Batching across requests in `sys1d`.** It adds throughput only: 19 to 26% more answers per second at s128_q1 with 8 clients, 13% with 4, and no gain at 1 or 2 clients or at s512_q10. It does not shorten a single request. An answer's probabilities depend on which requests share its forward. On english the largest shift was 0.0048, over the 0.0044 bar. An s512_q1 tail at 1 client after batched traffic has no known cause.
- **Exact ideas that lost.** `nax` with big tiles picked by an independent-op sweep took 1.029 of the time in the forward. A custom Metal 4 `matmul2d` gemm took 1.03 to 1.20 per op. `band` from 256 tokens was slower (above). A contiguous weight copy (`wcopy=t`) gave 0.988 to 1.014 per op, with no consistent sign. Caching or trimming the `nax` kernel source is deprioritized, not ruled out.

## What this means

- The lead over Python laya-mlx comes from doing less work, not from Rust. Python could copy head pruning (its research notes already list it) and, with more work, unpadding. Today no released runtime does either.
- The settings are reviewed, tested on all three Laya models, and the `sys1d` default.
- Closed after review: MLX used to compile the GeGLU step once per input shape, and with `unpad` the shape is the request's total token count, so each new count paid a trace and added a cache entry. The GeGLU trace is now shapeless (one trace for every shape; the split happens outside it). Paired A/B against the per-shape trace on the timing workload with the default settings: 1.001 on typed-decisions, 1.013 and 1.004 (sides swapped) on multilingual, identical answers.

## Files

- **Code.** `runtime/crates/laya-mlx/src/lib.rs` has the settings `dense_upto`, `headprune` and `unpad`. The experiments that did not help are at commit 0495800. `crates/laya-mlx/tests/settings.rs` has the equivalence tests. `runtime/crates/sys1-bench/src/bin/sys1-probe.rs` has `--ab SPEC_A SPEC_B` and `--check`.
- **Round 2.** `runtime/crates/laya-mlx/src/metal_kernels.rs` is the wrapper for MLX custom Metal kernels, `src/split_rope.rs` and `src/kernels/split_rope.metal` are the `fuserope` kernel with its load-time check and its tests against the MLX ops, and `tests/settings.rs` runs `fuserope` alone and with the three settings. The merge and int8 kernels were not merged.
- **Round 3.** In `runtime/crates/laya-mlx/`, `src/split_rope.rs` and `src/kernels/split_rope.metal` gain the `band` layout. `src/nax_gemm.rs` and `src/kernels/nax_gemm.metal` are the `nax` kernels with the device gate and the load check, and `src/kernels/mlx/` holds the MLX headers with their license. The loading settings are in `laya-core/src/agent.rs`, `backend.rs` and `weights.rs` and in `laya-mlx/src/lib.rs`. `tests/settings.rs` and `tests/loading.rs` have the equivalence and error tests. `sys1d/src/main.rs` sets the buffer limit through `laya_mlx::set_mlx_env_defaults`. `sys1-probe` has `--all-rows` and `--rows-out` for per-row paired timing.
- **Stage G.** `raw/speed/review/` holds the multilingual and english runs before the review (stage F) and after it (stage G), the paired speed log `stageG-ab.log`, and `stageF.sh`. Stage F ran the `sys1-bench` in the default target directory, not the build passed to it as `BIN_DIR`, because the sys1rust adapter re-sourced `bench/env.sh` and reset `CARGO_TARGET_DIR` (since fixed). Its results record `code_version` 0495800, which names the source tree, not the binary that ran; stage G ran on the reviewed build and is unaffected.
- **Results.** `raw/speed/bench/results/` holds stage E (timing, correctness and short) and the three alternating short runs. `raw/speed/fakeq/` holds the int8 simulation outputs.
- **Scripts.** `raw/speed/scripts/` has `stageE.sh` and its log, `gemm_shapes.py` (matmul speed at the model's shapes), `fakeq_adapter.py`, `long.jsonl` (1,024-token rows), and the per-op profile taken on battery (`ops-battery.txt`).
