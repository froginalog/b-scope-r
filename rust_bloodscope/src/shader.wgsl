// Bloodscope compute shader (WGSL).
//
// One thread = one seed. The combined-config shader computes ALL FOUR group
// bloodscopes per seed in a single pass:
//
//   Group A FT  (paint_blood          @ scale 0.4-0.5)  -- = Group C
//   Group A BS  (paint_blood_buckets  @ scale 0.4-0.5)
//   Group B     (paint_blood          @ scale 0.6-0.7)
//   Group D     (paint_blood          @ scale 0.2-0.3)
//
// Group C's bloodscope is identical to Group A FT (same texture + same scale)
// so we don't compute it separately. Output is one u32 per seed packing
// {ft_a, bs_a, b, d} into 4 bytes (low to high).
//
// The trick: the `scale_uv` parameter in TF2's compositor consumes ONE
// RandomFloat call regardless of range. So u, v, rotation, raw_scale01 are
// the same across all configs; only the scale-range remap differs. We capture
// raw_scale01 once and remap it three times.

const IA:   i32 = 16807;
const IM:   i32 = 2147483647;
const IQ:   i32 = 127773;
const IR:   i32 = 2836;
const NDIV: i32 = 67108864;
const AM:   f32 = 4.6566128730773925e-10;
const RNMX: f32 = 0.9999998807907104;

const W:      u32 = 2048u;
const H:      u32 = 2048u;
const W_F:    f32 = 2048.0;
const H_F:    f32 = 2048.0;
const W_MASK: u32 = 2047u;
const H_MASK: u32 = 2047u;
const BIG:    f32 = 1024.0;
const DEG2RAD: f32 = 0.017453292519943295;

struct Params {
    start_lo:    u32,
    start_hi:    u32,
    num_seeds:   u32,
    num_offsets: u32,
    // Three independent scale ranges. With multi-config mode, the shader
    // remaps the same raw RNG output to all three. Single-config mode just
    // uses scale_a (and ignores b/d).
    scale_a_min: f32,
    scale_a_max: f32,
    scale_b_min: f32,
    scale_b_max: f32,
    scale_d_min: f32,
    scale_d_max: f32,
    // multi_config=1 means the shader samples 3 positions per scope pixel
    // and writes 4 percentages per output u32 (ft_a | bs_a<<8 | b<<16 | d<<24).
    // multi_config=0 means single-config: only scale_a is used and we write
    // 2 percentages (ft | bs<<8) like before.
    multi_config: u32,
    _pad:         u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> offsets: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> pattern: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;

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

fn get_seed_lo_u64(lo: u32, hi: u32) -> u32 {
    var out: u32 = 0u;
    for (var i: u32 = 0u; i < 16u; i = i + 1u) {
        out = out | (((lo >> (2u * i + 1u)) & 1u) << i);
    }
    for (var i: u32 = 0u; i < 16u; i = i + 1u) {
        out = out | (((hi >> (2u * i + 1u)) & 1u) << (16u + i));
    }
    return out;
}

fn pct(hits: u32, n: u32) -> u32 {
    let p = (hits * 100u + n / 2u) / n;
    if (p > 100u) { return 100u; }
    return p;
}

@compute @workgroup_size(64)
fn cs_bloodscope(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let tid = gid.y * (num_wg.x * 64u) + gid.x;
    if (tid >= params.num_seeds) { return; }

    let sum_lo = params.start_lo + tid;
    let carry  = select(0u, 1u, sum_lo < params.start_lo);
    let sum_hi = params.start_hi + carry;

    let lo = get_seed_lo_u64(sum_lo, sum_hi);
    var rng = rng_init(bitcast<i32>(lo));

    for (var i: u32 = 0u; i < 7u; i = i + 1u) {
        _ = rng_float(&rng, 0.0, 1.0);
    }
    let tu          = rng_float(&rng, 0.0, 1.0);
    let tv          = rng_float(&rng, 0.0, 1.0);
    let rot         = rng_float(&rng, 0.0, 360.0);
    // Capture the raw [0,1] RNG output for scale BEFORE remapping. We then
    // linearly remap it to each of the three configs' scale ranges. This is
    // mathematically equivalent to running the compositor 3 times with
    // different scale_uv values but the same seed.
    let raw_scale01 = rng_float(&rng, 0.0, 1.0);
    let scale_a = mix(params.scale_a_min, params.scale_a_max, raw_scale01);
    let scale_b = mix(params.scale_b_min, params.scale_b_max, raw_scale01);
    let scale_d = mix(params.scale_d_min, params.scale_d_max, raw_scale01);

    let rad = rot * DEG2RAD;
    let c   = cos(rad);
    let s   = sin(rad);

    let mc_a  = scale_a * c; let ms_a = scale_a * s;
    let tpx_a = mc_a * tu - ms_a * tv;
    let tpy_a = ms_a * tu + mc_a * tv;

    var hit_a_ft: u32 = 0u;
    var hit_a_bs: u32 = 0u;
    var hit_b:    u32 = 0u;
    var hit_d:    u32 = 0u;
    let n = params.num_offsets;

    let multi = params.multi_config != 0u;
    var mc_b: f32 = 0.0; var ms_b: f32 = 0.0;
    var tpx_b: f32 = 0.0; var tpy_b: f32 = 0.0;
    var mc_d: f32 = 0.0; var ms_d: f32 = 0.0;
    var tpx_d: f32 = 0.0; var tpy_d: f32 = 0.0;
    if (multi) {
        mc_b  = scale_b * c; ms_b = scale_b * s;
        tpx_b = mc_b * tu - ms_b * tv;
        tpy_b = ms_b * tu + mc_b * tv;
        mc_d  = scale_d * c; ms_d = scale_d * s;
        tpx_d = mc_d * tu - ms_d * tv;
        tpy_d = ms_d * tu + mc_d * tv;
    }

    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let off = offsets[i];
        // Scale A position (Group A FT/BS, Group C)
        let usrc_a = mc_a * off.x - ms_a * off.y + tpx_a + BIG;
        let vsrc_a = ms_a * off.x + mc_a * off.y + tpy_a + BIG;
        let sx_a = u32(usrc_a * W_F) & W_MASK;
        let sy_a = u32(vsrc_a * H_F) & H_MASK;
        let idx_a = sy_a * W + sx_a;
        let word_a = pattern[idx_a >> 2u];
        let b_a = (word_a >> ((idx_a & 3u) * 8u)) & 0xFFu;
        hit_a_ft = hit_a_ft + (b_a & 1u);
        hit_a_bs = hit_a_bs + ((b_a >> 1u) & 1u);

        if (multi) {
            // Scale B position (Group B = paint_blood @ 0.6-0.7)
            let usrc_b = mc_b * off.x - ms_b * off.y + tpx_b + BIG;
            let vsrc_b = ms_b * off.x + mc_b * off.y + tpy_b + BIG;
            let sx_b = u32(usrc_b * W_F) & W_MASK;
            let sy_b = u32(vsrc_b * H_F) & H_MASK;
            let idx_b = sy_b * W + sx_b;
            let word_b = pattern[idx_b >> 2u];
            let b_b = (word_b >> ((idx_b & 3u) * 8u)) & 0xFFu;
            hit_b = hit_b + (b_b & 1u);

            // Scale D position (Group D = paint_blood @ 0.2-0.3)
            let usrc_d = mc_d * off.x - ms_d * off.y + tpx_d + BIG;
            let vsrc_d = ms_d * off.x + mc_d * off.y + tpy_d + BIG;
            let sx_d = u32(usrc_d * W_F) & W_MASK;
            let sy_d = u32(vsrc_d * H_F) & H_MASK;
            let idx_d = sy_d * W + sx_d;
            let word_d = pattern[idx_d >> 2u];
            let b_d = (word_d >> ((idx_d & 3u) * 8u)) & 0xFFu;
            hit_d = hit_d + (b_d & 1u);
        }
    }

    output[tid] = pct(hit_a_ft, n)
        | (pct(hit_a_bs, n) << 8u)
        | (pct(hit_b,    n) << 16u)
        | (pct(hit_d,    n) << 24u);
}
