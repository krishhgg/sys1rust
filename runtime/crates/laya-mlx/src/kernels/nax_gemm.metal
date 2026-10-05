// `nax=all`: the encoder's gemms on MLX's own NAX gemm loop, in another tile shape and order.
//
// MLX 0.32.2's NAX gemm gives each simdgroup an SM x SN block of the output, which it
// computes alone, straight from device memory, in K steps of 32 (`mlx::steel::gemm_loop`,
// in `mlx/steel/gemm/gemm_nax.h`). The threadgroup tile (BM x BN, WM x WN simdgroups) and the order the
// tiles launch in decide which blocks run side by side and so how often a weight tile comes
// from DRAM, but not the arithmetic: each output element is the same sum in the same order.
// This body calls that loop with a tile and an order of its own:
//
//   order: bands of G row tiles, row tiles fastest inside a band, so the row tiles that read
//   the same weight tile run back to back (MLX runs the column tiles of a row first)
//
// and finishes like MLX does, chosen by MODE (a `#define` the Rust side puts before this):
//
//   MODE 0  out = a @ w^T                          (MLX `matmul`)
//   MODE 1  out = a @ w^T + c                      (MLX `addmm`: `acc += float(c)`, then half)
//   MODE 2  out = half((0 + p0 + p1) * 1 + half(1) * c), p0 over K in [0, PART) and p1 over
//           [PART, K): MLX's NAX split-K addmm (`gemm_splitk_nax` and the accumulate kernel)
//           in one launch, without the fp32 scratch buffer
//
// Template arguments: BM, BN, WM, WN (tile and simdgroup grid; SM = BM / WM and SN = BN / WN
// are multiples of 16), BK (K block, as MLX's loop takes it), ALIGN_K (K, or for MODE 2 the
// second range, is a multiple of BK), G (band height in row tiles), PART (MODE 2).
// Inputs: a [M, K], w [N, K] (the weight as stored), c [M, N] (MODE 1 and 2), all half.
// Output: out [M, N] (half). Grid: one threadgroup of 32 * WM * WN threads per tile.

using namespace mlx::steel;
const int M = a_shape[0];
const int K = a_shape[1];
const int N = w_shape[0];
const int tiles_m = (M + BM - 1) / BM;
const int tiles_n = (N + BN - 1) / BN;
const int t = threadgroup_position_in_grid.x;
const int band = t / (G * tiles_n);
const int first = band * G;
const int rows_in_band = min(tiles_m - first, G);
const int r = t % (G * tiles_n);
const int rt = first + r % rows_in_band;
const int ct = r / rows_in_band;
constexpr short SM = BM / WM;
constexpr short SN = BN / WN;
constexpr short SK = 32;
constexpr short TM = SM / 16;
constexpr short TN = SN / 16;
const short sg = short(simdgroup_index_in_threadgroup);
const int row0 = rt * BM + SM * (sg / WN);
const int col0 = ct * BN + SN * (sg % WN);
// Blocks past the edge get a size of 0 or less: they still run the loop's barriers, as in
// MLX, and store nothing.
const short sgp_sm = short(min(int(SM), M - row0));
const short sgp_sn = short(min(int(SN), N - col0));
const device half* ap = a + size_t(row0) * K;
const device half* wp = w + size_t(col0) * K;
device half* dp = out + size_t(row0) * N + col0;
using Tile = NAXTile<float, TM, TN>;
using CFrag = typename Tile::NAXFrag_t;
using cfrag_t = typename CFrag::template dtype_frag_t<half>;
#if MODE != 0
const device half* cp = c + size_t(row0) * N + col0;
#endif

dispatch_bool(sgp_sm == SM, [&](auto am) {
  dispatch_bool(sgp_sn == SN, [&](auto an) {
#if MODE == 2
    Tile D = gemm_loop<half, SM, SN, SK, BK, false, true, am.value, an.value, (PART % BK) == 0, float>(
        ap, wp, K, K, PART, PART / BK, sgp_sm, sgp_sn);
    Tile D1 = gemm_loop<half, SM, SN, SK, BK, false, true, am.value, an.value, ALIGN_K, float>(
        ap + PART, wp + PART, K, K, K - PART, (K - PART) / BK, sgp_sm, sgp_sn);
#else
    Tile D = gemm_loop<half, SM, SN, SK, BK, false, true, am.value, an.value, ALIGN_K, float>(
        ap, wp, K, K, K, K / BK, sgp_sm, sgp_sn);
#endif
    if ((am.value || sgp_sm > 0) && (an.value || sgp_sn > 0)) {
#if MODE != 0
      const_for_loop<0, TM, 1>([&](auto mm) {
        const_for_loop<0, TN, 1>([&](auto nn) {
          auto m = mm * Int<CFrag::kFragRows>{};
          auto n = nn * Int<CFrag::kFragCols>{};
          cfrag_t ce;
          if constexpr (am.value && an.value) {
            CFrag::load(ce, cp, N, Int<1>{}, m, n);
          } else {
            CFrag::load_safe(ce, cp, N, Int<1>{}, sgp_sm, sgp_sn, m, n);
          }
          thread auto& de = D.template frag_at<mm, nn>();
#if MODE == 2
          thread auto& de1 = D1.template frag_at<mm, nn>();
#endif
          STEEL_PRAGMA_UNROLL
          for (short i = 0; i < Tile::kElemsPerFrag; i++) {
#if MODE == 2
            float acc = 0.0f;
            acc += de[i];
            acc += de1[i];
            de[i] = acc * 1.0f + (static_cast<half>(1.0f) * ce[i]);
#else
            de[i] += static_cast<float>(ce[i]);
#endif
          }
        });
      });
#endif
      if constexpr (am.value && an.value) {
        D.store(dp, N);
      } else {
        D.store_safe(dp, N, short2(sgp_sn, sgp_sm));
      }
    }
  });
});
