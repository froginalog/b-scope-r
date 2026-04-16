# bloodscope (Rust)

Exhaustive enumeration of bloodscope coverage % for every TF2 `paint_kit_seed`.

Ports the verified UV pipeline from `paint_blood_uv.py` and the scope-lens
disc from `paint_blood_circle_coverage.py` into Rust, then fans out over
all seeds with `rayon`. Only the ~9477 pixels inside the scope disc are
sampled per seed, not the full 2048² composite.

## Build

```bash
cd rust_bloodscope
cargo build --release
```

Requires a recent Rust toolchain. `.cargo/config.toml` sets
`target-cpu=native` so the build is specialized to the host's SIMD
capabilities.

## Run

```bash
# Self-test (RNG + ground-truth against paint_blood_uv.py)
./target/release/bloodscope --self-test

# All 2^32 seeds, both wear patterns, CSV of everything >= 90% coverage
./target/release/bloodscope --assets ../assets

# A range, just Field-Tested, full binary dump (byte per seed)
./target/release/bloodscope \
    --start 0 --end 1000000 --wear ft --full-binary --assets ../assets

# Quick scan (4x faster, ~1-2% noisier)
./target/release/bloodscope --assets ../assets --subsample 4
```

## Output

In `--out` directory (default `bloodscope_out/`):

- `bloodscopes.csv` — `seed,field_tested_pct,well_worn_battle_scarred_pct`
  for every seed whose coverage on EITHER pattern is ≥ `--min-pct`
  (default 90). Always written; pass `--min-pct 101` to skip.
- `bloodscope_field_tested.u8` — only with `--full-binary`. One byte per
  seed (0..=100). File size = `end - start` bytes. For default range =
  exactly 4 GiB.
- `bloodscope_buckets.u8` — likewise for `paint_blood_buckets.png`.

## Benchmarks

Measured on the dev machine (16-thread CPU + RTX 3090):

| Config                  | Seeds/s    | 2^32 ETA   |
|-------------------------|------------|------------|
| CPU `--subsample 1`     | ~260,000   | ~4.5 hours |
| CPU `--subsample 4`     | ~970,000   | ~73 min    |
| CPU `--subsample 16`    | ~3.7M      | ~19 min    |
| **GPU `--gpu`**         | **~20M**   | **3m 35s** |

CPU subsampling uses every Nth scope pixel (~√(N/9477) noise in the %).
GPU mode is full accuracy and still ~75x faster than the fastest CPU mode.

A measured full-space run: 4,294,967,296 seeds in 215 s on an RTX 3090.
~49M seeds (1.14% of u32) had coverage >= 90% on at least one wear.

## GPU mode

```bash
./target/release/bloodscope --gpu --assets ../assets
```

Portable: uses wgpu, runs on Vulkan / DirectX 12 / Metal (NVIDIA, AMD,
Intel, Apple). The shader is in `src/shader.wgsl`. One GPU thread per
seed; each thread runs the full RNG + all 9477 scope-pixel gathers.

Tuning:
- `--gpu-batch N` — threads per compute submit (default 2^22 = 4M).
  Larger batches reduce CPU↔GPU overhead but risk Windows' 2-second TDR
  driver timeout on very long kernels. 4M gives ~200 ms per dispatch on
  an RTX 3090 -- well under the limit.

## How it stays fast

- **Only stream[1] is computed.** `paint_blood` lives on the blood RNG
  stream (seedlo); stream[0] never affects the result, so we skip it.
- **9477 scope pixels vs 4.2M composite pixels** — 440× less work per seed.
- **Factored transform.** `R*S*T*uv + R*S*t` is combined into a single
  affine with 2 f32 mults and 2 f32 adds per pixel (+ one gather lookup).
- **Power-of-two modulo** — `sy % H` becomes `sy & 2047`.
- **Add-BIG trick** replaces `x.floor()` with a bias that survives the
  `&MASK` collapse, avoiding an SSE4.1 `roundss` dependency.
- **Combined pattern bitmap.** For `--wear both`, `paint_blood.png` and
  `paint_blood_buckets.png` are merged into one byte-per-pixel texture
  (bit0 = FT, bit1 = Buckets), halving pattern lookups.
- **`get_unchecked`** on the pattern array skips bounds checks in the
  innermost loop.
- **rayon fan-out** over chunks of seeds; each chunk is collected in
  seed order before streaming to disk so the output file is always
  seed-indexed.
