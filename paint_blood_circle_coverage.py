#!/usr/bin/env python3
"""
Measure blood coverage (percentage of bloody pixels) inside the SNIPER RIFLE
SCOPE LENS GLASS on the composite, for a given paint_kit_seed / wear.

Scope lens circle is fixed at (cx=1003, cy=155, r=55) in 2048x2048 texture
pixel coordinates. This was derived by:

  1. Using p_sniperrifle_groups.vtf to find the scope-group UV island (the
     only blue-group region, roughly x=[931,1070], y=[51,226]).
  2. Within that island, filtering by the distinctive blue-grey lens glass
     color (B>60, nearly-equal RGB, mid brightness) to isolate just the
     round lens surface excluding its surrounding scope housing/ring.
  3. Taking the centroid and a radius that fits the detected glass disc.

See assets/scope_circle_reference.png (full texture with circle outlined)
and assets/scope_circle_reference_zoom.png (close-up of the lens).

This lets you track "bloodscopes" -- percentage of the scope lens pixels
that are covered by splatter for a given seed/wear.

Reuses the verified UV transform from paint_blood_composite.py. A pixel is
"bloody" iff darkness = 1 - min(R,G,B)/255 >= threshold (default 0.20).

Usage:
    python paint_blood_circle_coverage.py <seed> [seed ...]
        [--wear N | --all-wears] [--threshold F] [--preview]

Examples:
    python paint_blood_circle_coverage.py 53 72 4466973305 --all-wears --preview
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Reuse compositor pipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paint_blood_uv import compute_blood_uv
from paint_blood_composite import (
    SNIPER_RIFLE_TEX,
    WEAR_BLOOD_TEXTURE,
    WEAR_NAMES,
    OUTPUT_DIR,
    load_vtf_as_pil,
    apply_uv_transform,
    composite_blood_on_base,
)

# -- Fixed scope lens circle ---------------------------------------------------
# Derived from c_sniperrifle.vtf + p_sniperrifle_groups.vtf (see module docstring)
SCOPE_CX     = 1003
SCOPE_CY     = 155
SCOPE_RADIUS = 55


_scope_mask_cache = None


def get_scope_mask(shape=(2048, 2048)) -> np.ndarray:
    """Boolean mask of the scope lens glass disc on the 2048x2048 texture."""
    global _scope_mask_cache
    if _scope_mask_cache is not None and _scope_mask_cache.shape == shape:
        return _scope_mask_cache
    H, W = shape
    yi, xi = np.mgrid[0:H, 0:W]
    mask = (xi - SCOPE_CX) ** 2 + (yi - SCOPE_CY) ** 2 <= SCOPE_RADIUS ** 2
    _scope_mask_cache = mask
    return mask


def _scope_outline(mask: np.ndarray) -> np.ndarray:
    """1-pixel-thick outline around the scope mask (boolean array)."""
    # Dilate by 1 and XOR with original
    pad = np.pad(mask, 1, constant_values=False)
    dilated = (pad[:-2,  1:-1] | pad[2:,  1:-1] |
               pad[1:-1, :-2] | pad[1:-1, 2:] | mask)
    return dilated & ~mask


# -- Fast path ----------------------------------------------------------------
# Sample only the 9,477 scope-disc pixels (not all 4.2M), and do both wears in
# a single gather by packing FT-bloody into bit 0 and BS/WW-bloody into bit 1
# of a combined 2048×2048 u8 bitmap. Same RGB-darkness rule as the slow path,
# but ~500x faster -- typically <1 ms per seed once caches are warm.

_FAST_CACHE: dict = {}


def _fast_init(threshold: float):
    """Precompute scope pixel UV coords + combined threshold bitmap (cached)."""
    key = round(threshold, 6)
    cached = _FAST_CACHE.get(key)
    if cached is not None:
        return cached

    # (1) Scope pixels in UV space
    mask = get_scope_mask()                      # (2048, 2048) bool
    ys, xs = np.where(mask)
    H, W = mask.shape
    scope_u = (xs.astype(np.float32) / W)
    scope_v = (ys.astype(np.float32) / H)

    # (2) Combined bitmap (bit 0 = FT bloody, bit 1 = Buckets bloody).
    ft_img = load_vtf_as_pil(WEAR_BLOOD_TEXTURE[3], mode="RGBA")
    bk_img = load_vtf_as_pil(WEAR_BLOOD_TEXTURE[5], mode="RGBA")
    ft_rgb = np.asarray(ft_img)[..., :3]
    bk_rgb = np.asarray(bk_img)[..., :3]
    # darkness = 1 - min(R,G,B)/255 >= threshold
    #        <=> min(R,G,B) <= round(255 * (1 - threshold))
    cutoff = int(round(255 * (1.0 - threshold)))
    ft_bloody = (ft_rgb.min(axis=-1) <= cutoff).astype(np.uint8)
    bk_bloody = (bk_rgb.min(axis=-1) <= cutoff).astype(np.uint8)
    combined  = (ft_bloody | (bk_bloody << 1)).astype(np.uint8)

    state = {
        "scope_u":  scope_u,
        "scope_v":  scope_v,
        "n_scope":  int(len(scope_u)),
        "combined": combined,
        "W":        int(W),
        "H":        int(H),
    }
    _FAST_CACHE[key] = state
    return state


def measure_coverage_fast(seed: int,
                          threshold: float = 0.20,
                          scale_range: tuple[float, float] = (0.4, 0.5),
                          ) -> tuple[float, float]:
    """Fast bloodscope coverage for BOTH wears of a seed.

    Returns (field_tested_pct, battle_scarred_pct). BS and WW share the
    same result (see docstring of paint_blood_circle_coverage.py).

    scale_range is the paintkit's `scale_uv` range. Harvest uses
    (0.4, 0.5); Night Owl, Purple Range etc. use (0.6, 0.7). Pass the
    value from BLOOD_PAINT_INDICES for the listing's paint_index.

    Roughly 500x faster than measure_coverage(); designed for sweeping
    hundreds of seeds (the market scanner). Matches measure_coverage()
    bit-for-bit at the default threshold 0.20 / scale_range (0.4, 0.5).
    """
    st = _fast_init(threshold)
    scope_u  = st["scope_u"]
    scope_v  = st["scope_v"]
    combined = st["combined"]
    n        = st["n_scope"]
    W        = st["W"]
    H        = st["H"]

    r = compute_blood_uv(seed, scale_range=scale_range)
    tu    = round(r["u"],        3)
    tv    = round(r["v"],        3)
    rot   = round(r["rotation"], 3)
    scale = round(r["scale"],    3)

    # Factored affine: pattern_uv = (R·S)·output_uv + (R·S)·t  = M·uv + tp
    rad  = np.float32(np.radians(rot))
    c    = np.cos(rad)
    s    = np.sin(rad)
    mc   = np.float32(scale) * c
    ms   = np.float32(scale) * s
    tpx  = mc * np.float32(tu) - ms * np.float32(tv)
    tpy  = ms * np.float32(tu) + mc * np.float32(tv)

    # Add-BIG trick (see rust_bloodscope/src/shader.wgsl): shifts usrc/vsrc
    # into the strictly-positive range, and because BIG*W is a multiple of W
    # the bottom log2(W) bits survive the AND-mask unchanged -- we get the
    # same pixel index as a floor-mod would, without actually doing floor().
    BIG = np.float32(1024.0)

    u_src = mc * scope_u - ms * scope_v + tpx + BIG
    v_src = ms * scope_u + mc * scope_v + tpy + BIG

    sx = (u_src * np.float32(W)).astype(np.int64) & (W - 1)
    sy = (v_src * np.float32(H)).astype(np.int64) & (H - 1)

    bits    = combined[sy, sx]
    ft_hits = int((bits & 1).sum())
    bk_hits = int(((bits >> 1) & 1).sum())
    return (100.0 * ft_hits / n, 100.0 * bk_hits / n)


# -- Coverage ------------------------------------------------------------------
def measure_coverage(seed: int, wear: int,
                     threshold: float = 0.20) -> dict:
    """
    Compute percentage of bloody pixels inside the scope-lens mask.

    Bloody := darkness = 1 - min(R, G, B)/255 >= threshold.
    """
    r = compute_blood_uv(seed)
    u     = round(r["u"],        3)
    v     = round(r["v"],        3)
    rot   = round(r["rotation"], 3)
    scale = round(r["scale"],    3)

    blood_path = WEAR_BLOOD_TEXTURE[wear]
    blood_img  = load_vtf_as_pil(blood_path, mode="RGBA")
    base_img   = load_vtf_as_pil(SNIPER_RIFLE_TEX, mode="RGB")
    W, H       = base_img.size

    warped = apply_uv_transform(blood_img, u, v, rot, scale, out_size=(W, H))
    warped_arr = np.asarray(warped)
    rgb = warped_arr[..., :3].astype(np.float32) / 255.0
    darkness = 1.0 - rgb.min(axis=-1)

    mask = get_scope_mask((H, W))

    total_pixels  = int(mask.sum())
    bloody_pixels = int((mask & (darkness >= threshold)).sum())
    pct           = 100.0 * bloody_pixels / total_pixels
    mean_dark     = float(darkness[mask].mean())

    return dict(
        seed            = seed,
        wear            = wear,
        wear_name       = WEAR_NAMES[wear],
        blood_texture   = blood_path,
        u               = u,
        v               = v,
        rotation        = rot,
        scale           = scale,
        threshold       = threshold,
        total_pixels    = total_pixels,
        bloody_pixels   = bloody_pixels,
        coverage_pct    = pct,
        mean_darkness   = mean_dark,
        _base           = base_img,
        _warped         = warped,
        _mask           = mask,
    )


# -- Preview -------------------------------------------------------------------
def save_preview(result: dict, out_dir: str) -> str:
    """Save a composite PNG with the scope-lens outline drawn on top."""
    base   = result["_base"]
    warped = result["_warped"]
    mask   = result["_mask"]
    comp   = composite_blood_on_base(base, warped).convert("RGB")

    # Overlay scope-lens outline in yellow
    arr = np.asarray(comp).copy()
    outline = _scope_outline(mask)
    arr[outline] = (255, 220, 0)
    comp = Image.fromarray(arr)

    draw = ImageDraw.Draw(comp)
    label = (f"seed {result['seed']}  wear {result['wear']} "
             f"{result['wear_name']}  bloodscope {result['coverage_pct']:.2f}%")
    draw.rectangle([10, 10, 10 + 10 * len(label) + 20, 50], fill=(0, 0, 0))
    draw.text((22, 18), label, fill=(255, 220, 0))

    os.makedirs(out_dir, exist_ok=True)
    wear_slug = result["wear_name"].replace(" ", "_").replace("-", "")
    fname = f"bloodscope_seed{result['seed']}_wear{result['wear']}_{wear_slug}.png"
    path = os.path.join(out_dir, fname)
    comp.save(path)
    return path


# -- CLI -----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Measure bloodscope % (blood coverage on sniper rifle "
                    "scope lens) for paint_blood war paints.")
    p.add_argument("seeds", nargs="+", type=int,
                   help="One or more paint_kit_seed values.")
    p.add_argument("--wear", type=int, default=3, choices=[3, 4, 5],
                   help="3=Field-Tested, 4=Well-Worn, 5=Battle-Scarred "
                        "(default: 3).")
    p.add_argument("--all-wears", action="store_true",
                   help="Run for all three wears.")
    p.add_argument("--threshold", type=float, default=0.20,
                   help="Darkness threshold, where darkness = 1 - "
                        "min(R,G,B)/255 (default 0.20).")
    p.add_argument("--preview", action="store_true",
                   help="Save a composite PNG with the scope lens outlined.")
    p.add_argument("--out-dir", default=OUTPUT_DIR,
                   help="Output directory for preview PNGs.")

    args = p.parse_args()
    wears = (3, 4, 5) if args.all_wears else (args.wear,)

    mask_px = int(get_scope_mask().sum())
    print(f"Scope lens disc: center=({SCOPE_CX},{SCOPE_CY})  "
          f"radius={SCOPE_RADIUS}  ({mask_px} px)")
    print(f"Threshold: darkness >= {args.threshold:.3f}")
    print(f"{'seed':>12s}  {'wear':>4s}  {'name':14s}  "
          f"{'bloody':>10s} / {'total':>8s}   {'cov %':>7s}  "
          f"{'meanDark':>8s}")

    for seed in args.seeds:
        for wear in wears:
            res = measure_coverage(seed, wear, threshold=args.threshold)
            print(f"{seed:12d}  {wear:4d}  {res['wear_name']:14s}  "
                  f"{res['bloody_pixels']:10d} / {res['total_pixels']:8d}   "
                  f"{res['coverage_pct']:7.3f}  "
                  f"{res['mean_darkness']:8.4f}")
            if args.preview:
                path = save_preview(res, args.out_dir)
                print(f"              preview -> {path}")


if __name__ == "__main__":
    main()
