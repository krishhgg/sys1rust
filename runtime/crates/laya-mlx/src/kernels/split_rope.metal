// Body of the `sys1_split_rope` kernel; MLX writes the signature around it (see
// metal_kernels.rs) with the template arguments `T` (the element type, half or float), `H`
// (heads), `D` (head dim), `PACKED`, `BAND` (0, 1 or 2, the output layout) and `S` (the local
// window, used by the banded layouts only).
//
// Inputs: qkv [R, 3*H*D], the encoder's qkv projection rows, where R is n*len or, with
// PACKED, the packed token count; unpack [n*len] u32, the packed index of every padded
// position (a 1-element placeholder without PACKED, never read); dims [n, len, nc] i32 (nc is
// the chunk count, read by the banded layouts only); lbase [1] f32, log2 of the rope base.
//
// BAND 0: the outputs are q and k rotated, v copied, each [n, H, len, D]. BAND 1: the chunked
// layout of the banded local attention: q [n*nc, H, S, D] where chunk c of row b holds
// positions c*S .. c*S+S-1 (zero past len), and k, v [n*nc, H, 3S, D] where slot j of chunk c
// holds position (c-1)*S + j (zero outside 0..len); the attention mask hides the zero slots.
// BAND 2: the same positions written once, in the padded position layout the chunk windows
// are views of: q [n, H, nc*S, D] (position p at row p, zero past len) and k, v
// [n, H, (nc+2)*S, D] (position p at row p + S, zero outside 0..len), so that chunk c's
// window of 3S keys starts at row c*S.
//
// One thread per (b, p, h) and 4 pairs (i, i + D/2) with i = 4m, so D must be a multiple of
// 8: then every vec<T, 4> load and store is inside its row and aligned (4 elements from a
// row start that is itself a multiple of 4 elements). The grid is exactly the work (with
// BAND, one thread per position of the padded chunk windows, see below), and the row guard
// covers any thread past it. The angle is MLX's rope arithmetic (exp2 of the
// log2 base, fast cos and sin, in float), so the result matches `fast::rope` bit for bit.
using V = metal::vec<T, 4>;
const int LEN = dims[1];
const int W = qkv_shape[1];
constexpr int HALF = D / 2;
constexpr int PAIRS = HALF / 4;
const int g = int(thread_position_in_grid.x);
const int i = (g % PAIRS) * 4;
if (BAND == 0) {
    const int h = (g / PAIRS) % H;
    const int p = (g / PAIRS / H) % LEN;
    const int b = g / PAIRS / H / LEN;
    if (b >= dims[0]) return;
    const int slot = b * LEN + p;
    const int srow = PACKED ? int(unpack[slot]) : slot;
    const size_t src = (size_t)srow * W + h * D + i;
    const size_t dst = (((size_t)b * H + h) * LEN + p) * D + i;
    float c[4];
    float s[4];
    #pragma clang loop unroll(full)
    for (int j = 0; j < 4; j++) {
        const float inv_freq = metal::exp2(-(float(i + j) / float(HALF)) * lbase[0]);
        const float theta = float(p) * inv_freq;
        c[j] = metal::fast::cos(theta);
        s[j] = metal::fast::sin(theta);
    }
    {
        const V x1 = *(const device V*)(qkv + src);
        const V x2 = *(const device V*)(qkv + src + HALF);
        V r1, r2;
        #pragma clang loop unroll(full)
        for (int j = 0; j < 4; j++) {
            r1[j] = T(float(x1[j]) * c[j] - float(x2[j]) * s[j]);
            r2[j] = T(float(x1[j]) * s[j] + float(x2[j]) * c[j]);
        }
        *(device V*)(q + dst) = r1;
        *(device V*)(q + dst + HALF) = r2;
    }
    {
        const V x1 = *(const device V*)(qkv + src + H * D);
        const V x2 = *(const device V*)(qkv + src + H * D + HALF);
        V r1, r2;
        #pragma clang loop unroll(full)
        for (int j = 0; j < 4; j++) {
            r1[j] = T(float(x1[j]) * c[j] - float(x2[j]) * s[j]);
            r2[j] = T(float(x1[j]) * s[j] + float(x2[j]) * c[j]);
        }
        *(device V*)(k + dst) = r1;
        *(device V*)(k + dst + HALF) = r2;
    }
    *(device V*)(v + dst) = *(const device V*)(qkv + src + 2 * H * D);
    *(device V*)(v + dst + HALF) = *(const device V*)(qkv + src + 2 * H * D + HALF);
} else {
    // One thread per (b, h, 4 pairs) and virtual position kp in -S .. (nc + 1) * S: the
    // positions any chunk window can hold. A real position (0 <= kp < len) is rotated once
    // and written, with BAND 1, to the 1 to 3 chunk windows that hold it (slot kp - (c - 1) * S
    // of chunk c for c in kp / S - 1 .. kp / S + 1, inside 0 .. nc) or, with BAND 2, to row
    // kp + S of the padded layout; a virtual one writes zeros to the same slots. Positions
    // below nc * S also write their query slot, zero past len.
    const int NC = dims[2];
    constexpr int KS = 3 * S;
    const int NP = (NC + 2) * S;
    const int h = (g / PAIRS) % H;
    const int kp = (g / PAIRS / H) % NP - S;
    const int b = g / PAIRS / H / NP;
    if (b >= dims[0]) return;
    const bool real = kp >= 0 && kp < LEN;
    const V zero = V(T(0));
    V kr1 = zero, kr2 = zero, v1 = zero, v2 = zero, qr1 = zero, qr2 = zero;
    if (real) {
        const int slot = b * LEN + kp;
        const int srow = PACKED ? int(unpack[slot]) : slot;
        const size_t src = (size_t)srow * W + h * D + i;
        float c[4];
        float s[4];
        #pragma clang loop unroll(full)
        for (int j = 0; j < 4; j++) {
            const float inv_freq = metal::exp2(-(float(i + j) / float(HALF)) * lbase[0]);
            const float theta = float(kp) * inv_freq;
            c[j] = metal::fast::cos(theta);
            s[j] = metal::fast::sin(theta);
        }
        {
            const V x1 = *(const device V*)(qkv + src);
            const V x2 = *(const device V*)(qkv + src + HALF);
            #pragma clang loop unroll(full)
            for (int j = 0; j < 4; j++) {
                qr1[j] = T(float(x1[j]) * c[j] - float(x2[j]) * s[j]);
                qr2[j] = T(float(x1[j]) * s[j] + float(x2[j]) * c[j]);
            }
        }
        {
            const V x1 = *(const device V*)(qkv + src + H * D);
            const V x2 = *(const device V*)(qkv + src + H * D + HALF);
            #pragma clang loop unroll(full)
            for (int j = 0; j < 4; j++) {
                kr1[j] = T(float(x1[j]) * c[j] - float(x2[j]) * s[j]);
                kr2[j] = T(float(x1[j]) * s[j] + float(x2[j]) * c[j]);
            }
        }
        v1 = *(const device V*)(qkv + src + 2 * H * D);
        v2 = *(const device V*)(qkv + src + 2 * H * D + HALF);
    }
    if (BAND == 1) {
        // Floor division of kp by S (kp >= -S, so kp + S >= 0).
        const int cb = (kp + S) / S - 1;
        #pragma clang loop unroll(full)
        for (int dc = -1; dc <= 1; dc++) {
            const int c = cb + dc;
            if (c < 0 || c >= NC) continue;
            const int j = kp - (c - 1) * S;
            const size_t dst = ((((size_t)b * NC + c) * H + h) * KS + j) * D + i;
            *(device V*)(k + dst) = kr1;
            *(device V*)(k + dst + HALF) = kr2;
            *(device V*)(v + dst) = v1;
            *(device V*)(v + dst + HALF) = v2;
        }
        if (kp >= 0 && kp < NC * S) {
            const int c = kp / S;
            const int j = kp - c * S;
            const size_t dst = ((((size_t)b * NC + c) * H + h) * S + j) * D + i;
            *(device V*)(q + dst) = qr1;
            *(device V*)(q + dst + HALF) = qr2;
        }
    } else {
        const size_t dst = (((size_t)b * H + h) * NP + (kp + S)) * D + i;
        *(device V*)(k + dst) = kr1;
        *(device V*)(k + dst + HALF) = kr2;
        *(device V*)(v + dst) = v1;
        *(device V*)(v + dst + HALF) = v2;
        if (kp >= 0 && kp < NC * S) {
            const size_t dst = (((size_t)b * H + h) * NC * S + kp) * D + i;
            *(device V*)(q + dst) = qr1;
            *(device V*)(q + dst + HALF) = qr2;
        }
    }
}
