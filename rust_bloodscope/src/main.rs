//! Exhaustive bloodscope coverage enumerator.
//!
//! For each seed in a given range, computes the paint_blood UV transform
//! (ported exactly from paint_blood_uv.py, ground-truth verified) and
//! measures the percentage of pixels inside the sniper rifle scope-lens
//! disc that are covered by splatter.
//!
//! Fast paths:
//!   * only stream[1] RNG is computed (blood stage lives on seedlo)
//!   * only the ~9477 pixels inside the scope disc are sampled -- NOT the
//!     full 2048x2048 composite
//!   * the blood pattern is precomputed once into a 4MB u8 bitmap
//!   * rayon fans out across physical cores
//!
//! Output is a binary file: seed `s` maps to byte `s - start` = pct (0..=100).

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use image::GenericImageView;
use rayon::prelude::*;

// -- Texture + scope geometry (must match paint_blood_circle_coverage.py) ----
const W: u32 = 2048;
const H: u32 = 2048;
const W_F: f32 = W as f32;
const H_F: f32 = H as f32;

const SCOPE_CX: i32 = 1003;
const SCOPE_CY: i32 = 155;
const SCOPE_R:  i32 = 55;

// Darkness threshold for "bloody": 1 - min(R,G,B)/255 >= 0.20
//   => min(R,G,B) <= 204
const BLOODY_MIN_RGB_MAX: u8 = 204;

// -- Source Engine CUniformRandomStream (Park-Miller + Bays-Durham) ----------
const NTAB: usize = 32;
const IA:   i32 = 16_807;
const IM:   i32 = 2_147_483_647;
const IQ:   i32 = 127_773;
const IR:   i32 = 2_836;
const NDIV: i32 = 1 + (IM - 1) / NTAB as i32; // 67_108_864
const AM:   f32 = 1.0 / IM as f32;
const RNMX: f32 = 1.0 - 1.2e-7;

#[derive(Clone)]
struct Rng {
    idum: i32,
    iy:   i32,
    iv:   [i32; NTAB],
}

impl Rng {
    #[inline]
    fn seeded(seed: i32) -> Self {
        // Matches vstdlib/random.cpp SetSeed + first-call warmup in ran1.
        let mut idum: i32 = if seed < 0 { seed } else { seed.wrapping_neg() };
        idum = if idum.wrapping_neg() < 1 { 1 } else { idum.wrapping_neg() };

        let mut iv = [0i32; NTAB];
        for j in (0..NTAB + 8).rev() {
            let k = idum / IQ;
            idum = IA.wrapping_mul(idum - k * IQ) - IR.wrapping_mul(k);
            if idum < 0 {
                idum = idum.wrapping_add(IM);
            }
            if j < NTAB {
                iv[j] = idum;
            }
        }
        let iy = iv[0];
        Rng { idum, iy, iv }
    }

    #[inline(always)]
    fn gen(&mut self) -> i32 {
        let k = self.idum / IQ;
        self.idum = IA.wrapping_mul(self.idum - k * IQ) - IR.wrapping_mul(k);
        if self.idum < 0 {
            self.idum = self.idum.wrapping_add(IM);
        }
        let mut j = (self.iy / NDIV) as usize;
        if j >= NTAB {
            j &= NTAB - 1;
        }
        self.iy = self.iv[j];
        self.iv[j] = self.idum;
        self.iy
    }

    #[inline(always)]
    fn random_float(&mut self, lo: f32, hi: f32) -> f32 {
        let raw = self.gen() as f32;
        let mut fl = AM * raw;
        if fl > RNMX {
            fl = RNMX;
        }
        fl * (hi - lo) + lo
    }
}

// -- GetSeed (ctexturecompositor.cpp:1652) bit deinterleave ------------------
#[inline(always)]
fn get_seed_lo(n: u64) -> u32 {
    // Odd-indexed bits of n -> lo.
    let mut lo: u32 = 0;
    for i in 0..32 {
        lo |= ((n & (1u64 << (2 * i + 1))) >> (i + 1)) as u32;
    }
    lo
}

/// (u, v, rotation, scale) for `paint_blood` stage given a paint_kit_seed.
#[inline]
fn compute_blood_uv(paint_kit_seed: u64) -> (f32, f32, f32, f32) {
    let lo = get_seed_lo(paint_kit_seed) as i32;
    let mut rng = Rng::seeded(lo);

    // 7 pre-calls consumed by texture-A stage on same stream
    for _ in 0..7 {
        rng.random_float(0.0, 1.0);
    }
    let u     = rng.random_float(0.0, 1.0);
    let v     = rng.random_float(0.0, 1.0);
    let rot   = rng.random_float(0.0, 360.0);
    let scale = rng.random_float(0.4, 0.5);
    (u, v, rot, scale)
}

// -- Scope pixel offsets in UV space -----------------------------------------
/// Offsets (in UV [0,1)) of every pixel inside the scope lens disc.
fn scope_offsets_uv() -> Vec<(f32, f32)> {
    let mut offsets = Vec::with_capacity(9477);
    let r2 = SCOPE_R * SCOPE_R;
    for y in (SCOPE_CY - SCOPE_R)..=(SCOPE_CY + SCOPE_R) {
        for x in (SCOPE_CX - SCOPE_R)..=(SCOPE_CX + SCOPE_R) {
            let dx = x - SCOPE_CX;
            let dy = y - SCOPE_CY;
            if dx * dx + dy * dy <= r2 {
                offsets.push((x as f32 / W_F, y as f32 / H_F));
            }
        }
    }
    offsets
}

// -- Pattern loading ---------------------------------------------------------
/// Load a PNG pattern and return a 2048*2048 byte bitmap:
///   1 = bloody (darkness >= 0.20), 0 otherwise.
fn load_pattern(path: &std::path::Path) -> Vec<u8> {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("failed to open {:?}: {}", path, e));
    let (iw, ih) = img.dimensions();
    if iw != W || ih != H {
        panic!("pattern {:?} is {}x{}, expected {}x{}", path, iw, ih, W, H);
    }
    let rgba = img.into_rgba8();
    let mut out = vec![0u8; (W * H) as usize];
    for (i, px) in rgba.pixels().enumerate() {
        let m = px[0].min(px[1]).min(px[2]);
        // bloody iff min(R,G,B) <= 204  (i.e. darkness >= 0.20)
        out[i] = (m <= BLOODY_MIN_RGB_MAX) as u8;
    }
    out
}

// -- Core per-seed coverage --------------------------------------------------
/// Compute R*S*T matrix coefficients once per seed (6 f32s).
#[inline(always)]
fn setup(seed: u64) -> (f32, f32, f32, f32) {
    let (tu, tv, rot, scale) = compute_blood_uv(seed);
    let rad = rot.to_radians();
    let (c, s) = (rad.cos(), rad.sin());
    let mc = scale * c;
    let ms = scale * s;
    let tpx = mc * tu - ms * tv;
    let tpy = ms * tu + mc * tv;
    (mc, ms, tpx, tpy)
}

// W and H are 2^11 so % is a bitwise AND.
const W_MASK: u32 = W - 1;
const H_MASK: u32 = H - 1;
// Add-BIG trick: usrc+BIG is always positive, and BIG*W is a multiple of W,
// so the bottom log2(W) bits of `(usrc+BIG)*W` as u32 equal the bits we
// want from usrc*W mod W. Avoids the floor() on x86 w/o SSE4.1.
const BIG: f32 = 1024.0;

/// Coverage for ONE pattern (u8 per pixel, 0 or 1).
#[inline]
fn coverage_pct_one(seed: u64, pattern: &[u8], offsets_uv: &[(f32, f32)]) -> u8 {
    let (mc, ms, tpx, tpy) = setup(seed);

    let mut hit: u32 = 0;
    for &(ou, ov) in offsets_uv {
        let usrc = mc * ou - ms * ov + tpx + BIG;
        let vsrc = ms * ou + mc * ov + tpy + BIG;
        let sx = ((usrc * W_F) as u32) & W_MASK;
        let sy = ((vsrc * H_F) as u32) & H_MASK;
        unsafe {
            hit += *pattern.get_unchecked((sy * W + sx) as usize) as u32;
        }
    }

    let n = offsets_uv.len() as u32;
    (((hit as u64 * 100 + (n as u64) / 2) / n as u64) as u8).min(100)
}

/// Coverage for BOTH patterns in a single scan.
///
/// `combined[i]` packs two bits: bit0 = FT bloody, bit1 = Buckets bloody.
#[inline]
fn coverage_pct_both(
    seed: u64,
    combined: &[u8],
    offsets_uv: &[(f32, f32)],
) -> (u8, u8) {
    let (mc, ms, tpx, tpy) = setup(seed);

    let mut hit_ft: u32 = 0;
    let mut hit_bk: u32 = 0;
    for &(ou, ov) in offsets_uv {
        let usrc = mc * ou - ms * ov + tpx + BIG;
        let vsrc = ms * ou + mc * ov + tpy + BIG;
        let sx = ((usrc * W_F) as u32) & W_MASK;
        let sy = ((vsrc * H_F) as u32) & H_MASK;
        let b = unsafe { *combined.get_unchecked((sy * W + sx) as usize) } as u32;
        hit_ft += b & 1;
        hit_bk += (b >> 1) & 1;
    }

    let n = offsets_uv.len() as u64;
    let half = n / 2;
    let ft_pct = ((hit_ft as u64 * 100 + half) / n) as u8;
    let bk_pct = ((hit_bk as u64 * 100 + half) / n) as u8;
    (ft_pct.min(100), bk_pct.min(100))
}

/// Combine two single-bit pattern maps into bit0=FT, bit1=Buckets.
fn combine_patterns(ft: &[u8], bk: &[u8]) -> Vec<u8> {
    assert_eq!(ft.len(), bk.len());
    let mut out = vec![0u8; ft.len()];
    for i in 0..ft.len() {
        out[i] = (ft[i] & 1) | ((bk[i] & 1) << 1);
    }
    out
}

// -- RNG self-test (must match paint_blood_uv.py exactly) --------------------
fn self_test() -> bool {
    // RNG sanity: set_seed(72) -> first random_float(0,1) == 0.5430998
    let mut r = Rng::seeded(72);
    let f = r.random_float(0.0, 1.0);
    let rng_ok = (f - 0.5430998).abs() < 1e-5;
    eprintln!(
        "RNG sanity: set_seed(72) -> {:.7}  [{}]",
        f,
        if rng_ok { "OK" } else { "FAIL" }
    );

    // Ground truth seeds 53 and 72
    let check = |seed: u64, eu: f32, ev: f32, erot: f32, escale: f32| -> bool {
        let (u, v, rot, sc) = compute_blood_uv(seed);
        let ok = (u - eu).abs() < 2e-3
            && (v - ev).abs() < 2e-3
            && (rot - erot).abs() < 5e-2
            && (sc - escale).abs() < 2e-3;
        eprintln!(
            "seed {:12}:  u={:.3}  v={:.3}  rot={:.3}  scale={:.3}  [{}]",
            seed, u, v, rot, sc, if ok { "OK" } else { "FAIL" }
        );
        ok
    };
    let a = check(53, 0.606, 0.670, 193.978, 0.429);
    let b = check(72, 0.662, 0.767, 21.384, 0.498);
    // seed 4466973305: user-provided; coverage should be ~100
    let (u, v, rot, sc) = compute_blood_uv(4_466_973_305);
    eprintln!(
        "seed 4466973305:  u={:.3}  v={:.3}  rot={:.3}  scale={:.3}",
        u, v, rot, sc
    );
    rng_ok && a && b
}

// -- CLI ---------------------------------------------------------------------
#[derive(Copy, Clone, PartialEq, Eq, Debug, ValueEnum)]
enum Wear {
    /// Field-Tested (uses paint_blood.png)
    Ft,
    /// Well-Worn / Battle-Scarred (uses paint_blood_buckets.png)
    Buckets,
    /// Both patterns (writes two output files)
    Both,
}

#[derive(Parser)]
#[command(
    name = "bloodscope",
    about = "Enumerate bloodscope coverage % for every paint_kit_seed in a range."
)]
struct Args {
    /// First seed (inclusive). Default 0.
    #[arg(long, default_value_t = 0u64)]
    start: u64,

    /// One past the last seed (exclusive). Default 2^32.
    #[arg(long, default_value_t = 1u64 << 32)]
    end: u64,

    /// Which wear pattern(s) to run.
    #[arg(long, value_enum, default_value_t = Wear::Both)]
    wear: Wear,

    /// Path to the assets/ directory (containing paint_blood*.png).
    #[arg(long)]
    assets: Option<PathBuf>,

    /// Output directory.
    #[arg(long, default_value = "bloodscope_out")]
    out: PathBuf,

    /// Also write the full per-seed binary dumps (2 x 4GB for the default
    /// range). Off by default -- most users only want the CSV of hits.
    #[arg(long)]
    full_binary: bool,

    /// Write CSV of any seed whose coverage on either pattern is >= this.
    /// Set to 101 to disable the CSV.
    #[arg(long, default_value_t = 90u8)]
    min_pct: u8,

    /// Use every Nth scope pixel (1 = exact, 4 = ~4x faster, ~1-2% noisier).
    #[arg(long, default_value_t = 1usize)]
    subsample: usize,

    /// Chunk size per parallel batch (also the streaming-write granularity).
    #[arg(long, default_value_t = 1u64 << 20)]
    chunk: u64,

    /// Only run the self-test and exit.
    #[arg(long)]
    self_test: bool,
}

fn default_assets_dir() -> PathBuf {
    // Assume binary is run from the repo or from rust_bloodscope/
    let mut p = std::env::current_dir().unwrap();
    loop {
        if p.join("assets").is_dir() {
            return p.join("assets");
        }
        if !p.pop() {
            break;
        }
    }
    PathBuf::from("../assets")
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();

    // 1. Self-test against ground truth
    let ok = self_test();
    if !ok {
        eprintln!("\nSelf-test FAILED. Aborting.");
        std::process::exit(2);
    }
    eprintln!("Self-test OK.\n");
    if args.self_test {
        return Ok(());
    }

    // 2. Load patterns
    let assets = args.assets.unwrap_or_else(default_assets_dir);
    let ft_path      = assets.join("paint_blood.png");
    let buckets_path = assets.join("paint_blood_buckets.png");

    let wants_ft      = matches!(args.wear, Wear::Ft | Wear::Both);
    let wants_buckets = matches!(args.wear, Wear::Buckets | Wear::Both);

    eprintln!("Loading patterns from {:?}", assets);
    let pat_ft      = if wants_ft { load_pattern(&ft_path) } else { Vec::new() };
    let pat_buckets = if wants_buckets { load_pattern(&buckets_path) } else { Vec::new() };
    let bloody_ft      = pat_ft.iter().map(|&b| b as u64).sum::<u64>();
    let bloody_buckets = pat_buckets.iter().map(|&b| b as u64).sum::<u64>();
    if wants_ft {
        eprintln!(
            "  paint_blood.png          -> {} bloody px ({:.3}%)",
            bloody_ft,
            100.0 * bloody_ft as f64 / (W * H) as f64
        );
    }
    if wants_buckets {
        eprintln!(
            "  paint_blood_buckets.png  -> {} bloody px ({:.3}%)",
            bloody_buckets,
            100.0 * bloody_buckets as f64 / (W * H) as f64
        );
    }

    // 3. Precompute scope offsets (optionally subsampled)
    let mut offsets = scope_offsets_uv();
    if args.subsample > 1 {
        offsets = offsets.into_iter().step_by(args.subsample).collect();
    }
    eprintln!("Scope disc: center=({},{})  r={}  {} pixels (subsample={})",
        SCOPE_CX, SCOPE_CY, SCOPE_R, offsets.len(), args.subsample);

    // 4. Open output files
    std::fs::create_dir_all(&args.out)?;
    let mut ft_writer = if wants_ft && args.full_binary {
        Some(BufWriter::with_capacity(
            1 << 20,
            File::create(args.out.join("bloodscope_field_tested.u8"))?,
        ))
    } else { None };
    let mut bk_writer = if wants_buckets && args.full_binary {
        Some(BufWriter::with_capacity(
            1 << 20,
            File::create(args.out.join("bloodscope_buckets.u8"))?,
        ))
    } else { None };
    let mut csv_writer = if args.min_pct <= 100 {
        let mut w = BufWriter::with_capacity(
            1 << 20,
            File::create(args.out.join("bloodscopes.csv"))?,
        );
        writeln!(w, "seed,field_tested_pct,well_worn_battle_scarred_pct")?;
        Some(w)
    } else { None };

    let total = args.end - args.start;
    eprintln!("Enumerating {} seeds [{}, {}) on {} threads\n",
        total, args.start, args.end, rayon::current_num_threads());

    // Combined (bit0=FT, bit1=Buckets) pattern for the dual path -- halves
    // pattern lookups when both wears are requested.
    let combined: Vec<u8> = if wants_ft && wants_buckets {
        combine_patterns(&pat_ft, &pat_buckets)
    } else {
        Vec::new()
    };

    let t0 = Instant::now();
    let mut done: u64 = 0;
    let chunk = args.chunk.min(total);
    let mut next_report = Instant::now() + std::time::Duration::from_secs(2);

    let mut cur = args.start;
    while cur < args.end {
        let hi = (cur + chunk).min(args.end);
        let n  = (hi - cur) as usize;

        // Produce chunks in parallel
        let (ft_buf, bk_buf): (Vec<u8>, Vec<u8>) = if wants_ft && wants_buckets {
            (0..n as u64)
                .into_par_iter()
                .map(|i| coverage_pct_both(cur + i, &combined, &offsets))
                .unzip()
        } else if wants_ft {
            let v: Vec<u8> = (0..n as u64).into_par_iter()
                .map(|i| coverage_pct_one(cur + i, &pat_ft, &offsets))
                .collect();
            (v, Vec::new())
        } else {
            let v: Vec<u8> = (0..n as u64).into_par_iter()
                .map(|i| coverage_pct_one(cur + i, &pat_buckets, &offsets))
                .collect();
            (Vec::new(), v)
        };

        if let Some(w) = ft_writer.as_mut() { w.write_all(&ft_buf)?; }
        if let Some(w) = bk_writer.as_mut() { w.write_all(&bk_buf)?; }

        // Write CSV rows for any hit >= min_pct
        if let Some(w) = csv_writer.as_mut() {
            let min = args.min_pct;
            for i in 0..n {
                let ft_v = if wants_ft { ft_buf[i] } else { 0 };
                let bk_v = if wants_buckets { bk_buf[i] } else { 0 };
                if ft_v >= min || bk_v >= min {
                    writeln!(w, "{},{},{}", cur + i as u64, ft_v, bk_v)?;
                }
            }
        }

        done += n as u64;
        cur   = hi;

        if Instant::now() >= next_report {
            let elapsed = t0.elapsed().as_secs_f64();
            let rate    = done as f64 / elapsed;
            let eta     = (total - done) as f64 / rate;
            eprintln!(
                "  {:>12}/{} seeds  ({:5.2}%)  {:>9.0} seed/s  elapsed {:.0}s  ETA {:.0}s",
                done, total, 100.0 * done as f64 / total as f64,
                rate, elapsed, eta
            );
            next_report = Instant::now() + std::time::Duration::from_secs(5);
        }
    }

    if let Some(w) = ft_writer.as_mut()  { w.flush()?; }
    if let Some(w) = bk_writer.as_mut()  { w.flush()?; }
    if let Some(w) = csv_writer.as_mut() { w.flush()?; }

    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!("\nDone. {} seeds in {:.1}s  ({:.0} seed/s)",
        total, elapsed, total as f64 / elapsed);
    eprintln!("Output dir: {:?}", args.out);
    if args.full_binary && wants_ft {
        eprintln!(
            "  {:?}  ({} bytes)",
            args.out.join("bloodscope_field_tested.u8"), total
        );
    }
    if args.full_binary && wants_buckets {
        eprintln!(
            "  {:?}  ({} bytes)",
            args.out.join("bloodscope_buckets.u8"), total
        );
    }
    if args.min_pct <= 100 {
        eprintln!("  {:?}  (CSV of seeds with coverage >= {}%)",
            args.out.join("bloodscopes.csv"), args.min_pct);
    }
    Ok(())
}
