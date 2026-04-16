// Bloodscope compute shader (WGSL).
//
// Ports paint_blood_uv.py + paint_blood_circle_coverage.py to the GPU.
// One thread = one seed. Full accuracy (no subsampling): each thread samples
// ALL ~9477 scope-lens pixels and counts the bloody ones for both wear
// patterns simultaneously (1 gather per pixel via a combined bitmap).

// -- Source Engine CUniformRandomStream constants ----------------------------
const IA:   i32 = 16807;
const IM:   i32 = 2147483647;
const IQ:   i32 = 127773;
const IR:   i32 = 2836;
const NDIV: i32 = 67108864;                  // 1 + (IM-1)/32
const AM:   f32 = 4.6566128730773925e-10;    // 1.0 / IM
const RNMX: f32 = 0.9999998807907104;        // 1 - 1.2e-7

// -- Texture / scope geometry (match paint_blood_circle_coverage.py) ---------
const W:      u32 = 2048u;
const H:      u32 = 2048u;
const W_F:    f32 = 2048.0;
const H_F:    f32 = 2048.0;
const W_MASK: u32 = 2047u;
const H_MASK: u32 = 2047u;
const BIG:    f32 = 1024.0;                  // floor-avoidance bias
const DEG2RAD: f32 = 0.017453292519943295;

// -- Bindings ---------------------------------------------------------------
struct Params {
    start_lo:    u32,    // low 32 bits of first seed in this dispatch
    start_hi:    u32,    // high 32 bits of first seed in this dispatch
    num_seeds:   u32,    // number of seeds in this dispatch
    num_offsets: u32,    // length of `offsets`
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> offsets: array<vec2<f32>>;
// Pattern is packed 4 bytes per u32. pattern[i / 4] byte (i & 3):
//   bit 0 = FT bloody, bit 1 = Buckets bloody.
@group(0) @binding(2) var<storage, read> pattern: array<u32>;
// One u32 per seed: low byte = FT pct, next byte = Buckets pct.
@group(0) @binding(3) var<storage, read_write> output: array<u32>;

// -- RNG state held in function scope ----------------------------------------
struct Rng {
    idum: i32,
    iy:   i32,
    iv:   array<i32, 32>,
};

fn rng_init(seed: i32) -> Rng {
    var r: Rng;
    var idum: i32;
    if (seed < 0) { idum = seed; } else { idum = -seed; }
    if (-idum < 1) { idum = 1; } else { idum = -idum; }
    // Numerical Recipes warmup: 8 + NTAB iterations, last 32 fill iv[]
    for (var j: i32 = 39; j >= 0; j = j - 1) {
        let k = idum / IQ;
        idum = IA * (idum - k * IQ) - IR * k;
        if (idum < 0) { idum = idum + IM; }
        if (j < 32) { r.iv[j] = idum; }
    }
    r.idum = idum;
    r.iy   = r.iv[0];
    return r;
}

fn rng_gen(r: ptr<function, Rng>) -> i32 {
    let k = (*r).idum / IQ;
    (*r).idum = IA * ((*r).idum - k * IQ) - IR * k;
    if ((*r).idum < 0) { (*r).idum = (*r).idum + IM; }
    var j = u32((*r).iy / NDIV);
    if (j >= 32u) { j = j & 31u; }
    (*r).iy    = (*r).iv[j];
    (*r).iv[j] = (*r).idum;
    return (*r).iy;
}

fn rng_float(r: ptr<function, Rng>, lo: f32, hi: f32) -> f32 {
    let raw = f32(rng_gen(r));
    var fl  = AM * raw;
    if (fl > RNMX) { fl = RNMX; }
    return fl * (hi - lo) + lo;
}

// Odd-indexed bits of a 64-bit seed (split into two u32 halves) go to
// seedlo (blood stream[1]). Output bit i = input bit (2i+1) for i=0..31.
fn get_seed_lo_u64(lo: u32, hi: u32) -> u32 {
    var out: u32 = 0u;
    // Output bits 0..15 <- odd bits of `lo` (positions 1,3,...,31)
    for (var i: u32 = 0u; i < 16u; i = i + 1u) {
        out = out | (((lo >> (2u * i + 1u)) & 1u) << i);
    }
    // Output bits 16..31 <- odd bits of `hi` (positions 1,3,...,31)
    for (var i: u32 = 0u; i < 16u; i = i + 1u) {
        out = out | (((hi >> (2u * i + 1u)) & 1u) << (16u + i));
    }
    return out;
}

@compute @workgroup_size(64)
fn cs_bloodscope(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    // Flatten 2D dispatch (Wx, Wy) with workgroup_size=(64,1,1) into 1D tid.
    let tid = gid.y * (num_wg.x * 64u) + gid.x;
    if (tid >= params.num_seeds) { return; }

    // 64-bit seed = (start_hi, start_lo) + tid, carrying into hi
    let sum_lo = params.start_lo + tid;
    let carry  = select(0u, 1u, sum_lo < params.start_lo);
    let sum_hi = params.start_hi + carry;

    // 1. RNG pipeline (stream[1] only; stream[0] isn't used by blood stage)
    let lo = get_seed_lo_u64(sum_lo, sum_hi);
    var rng = rng_init(bitcast<i32>(lo));

    // 7 pre-calls consumed by preceding texture-A stage
    for (var i: u32 = 0u; i < 7u; i = i + 1u) {
        _ = rng_float(&rng, 0.0, 1.0);
    }
    let tu    = rng_float(&rng, 0.0, 1.0);
    let tv    = rng_float(&rng, 0.0, 1.0);
    let rot   = rng_float(&rng, 0.0, 360.0);
    let scale = rng_float(&rng, 0.4, 0.5);

    // 2. Factor transform once per seed
    let rad = rot * DEG2RAD;
    let c   = cos(rad);
    let s   = sin(rad);
    let mc  = scale * c;
    let ms  = scale * s;
    let tpx = mc * tu - ms * tv;
    let tpy = ms * tu + mc * tv;

    // 3. Sample the scope disc
    var hit_ft: u32 = 0u;
    var hit_bk: u32 = 0u;
    let n = params.num_offsets;
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let off = offsets[i];
        let usrc = mc * off.x - ms * off.y + tpx + BIG;
        let vsrc = ms * off.x + mc * off.y + tpy + BIG;
        let sx = u32(usrc * W_F) & W_MASK;
        let sy = u32(vsrc * H_F) & H_MASK;
        let idx = sy * W + sx;
        // unpack byte `idx & 3` from u32 `idx / 4`
        let word = pattern[idx >> 2u];
        let b    = (word >> ((idx & 3u) * 8u)) & 0xFFu;
        hit_ft = hit_ft + (b & 1u);
        hit_bk = hit_bk + ((b >> 1u) & 1u);
    }

    // 4. Round to integer percent and pack
    let half_n = n / 2u;
    var ft_pct = (hit_ft * 100u + half_n) / n;
    var bk_pct = (hit_bk * 100u + half_n) / n;
    if (ft_pct > 100u) { ft_pct = 100u; }
    if (bk_pct > 100u) { bk_pct = 100u; }

    output[tid] = ft_pct | (bk_pct << 8u);
}
