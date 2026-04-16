#!/usr/bin/env python3
"""
TF2 paint_blood -> sniper rifle TEXTURE compositor.

Given a paint_kit_seed, computes the UV transform (via paint_blood_uv.py),
loads the appropriate blood splatter texture for the wear level, applies the
transform, and composites it on top of the sniper rifle texture.

Wear mapping (from paintkits_master.txt, harvest_sniperrifle_boneyard):
    Field-Tested   (wear 3) -> patterns/paint_blood.vtf
    Well-Worn      (wear 4) -> patterns/paint_blood_buckets.vtf
    Battle Scarred (wear 5) -> patterns/paint_blood_buckets.vtf

Textures ship with the repo under ./assets/ as PNGs extracted from the
TF2 VPK -- no TF2 install required to run this script.

Outputs are written to ./output/ as PNG files.

Usage:
    python paint_blood_composite.py <seed>
    python paint_blood_composite.py 53
    python paint_blood_composite.py 72
"""

import os
import sys

import numpy as np
from PIL import Image

# -- Import the verified RNG pipeline -----------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paint_blood_uv import compute_blood_uv


# -- Config --------------------------------------------------------------------
REPO_DIR   = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(REPO_DIR, "assets")
OUTPUT_DIR = os.path.join(REPO_DIR, "output")

# Bundled textures (extracted from the TF2 VPK at repo build time).
SNIPER_RIFLE_TEX = os.path.join(ASSETS_DIR, "c_sniperrifle.png")

# Wear levels: 1=Factory New, 2=Minimal Wear, 3=Field-Tested,
#              4=Well-Worn, 5=Battle Scarred
WEAR_BLOOD_TEXTURE = {
    3: os.path.join(ASSETS_DIR, "paint_blood.png"),          # Field-Tested
    4: os.path.join(ASSETS_DIR, "paint_blood_buckets.png"),  # Well-Worn
    5: os.path.join(ASSETS_DIR, "paint_blood_buckets.png"),  # Battle Scarred
}

WEAR_NAMES = {
    3: "Field-Tested",
    4: "Well-Worn",
    5: "Battle Scarred",
}


# -- Texture loading -----------------------------------------------------------
_tex_cache = {}


def load_vtf_as_pil(path: str, mode: str = "RGBA") -> Image.Image:
    """Load a bundled texture PNG from ./assets/ and return a PIL Image.

    Note: historically these were loaded from the TF2 VPK as .vtf files.
    They have been pre-extracted to PNG so this repo can run standalone.
    The function name is kept for backwards compatibility.

    Note 2: many TF2 weapon textures use the alpha channel for other purposes
    (masking, phong exponent, etc.) and are NOT meant to be alpha-composited
    as-is. For display/composite-over-base use mode='RGB'.
    """
    key = (path, mode)
    if key in _tex_cache:
        return _tex_cache[key]
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Texture not found: {path}\n"
            f"Expected it under {ASSETS_DIR}. Re-extract with "
            f"tools/extract_assets.py if needed."
        )
    img = Image.open(path).convert(mode)
    _tex_cache[key] = img
    return img


# -- UV transform --------------------------------------------------------------
def apply_uv_transform(blood_img: Image.Image,
                       u: float, v: float, rot_deg: float, scale: float,
                       out_size=(2048, 2048)) -> Image.Image:
    """
    Apply the paint_blood UV transform to produce the pattern texture for
    compositing onto the base weapon texture.

    FULL DERIVATION FROM TF2 SOURCE:
    --------------------------------

    (1) Matrix construction (ctexturecompositor.cpp:599-605):

        MatrixBuildRotateZ( m, rotation );        // m = R
        m = m.Scale( (scaleUV, scaleUV, 1) );     // m = R * S   (post-multiply)
        MatrixTranslate( m, (tu, tv, 0) );        // m = R * S * T  (post-multiply)
        // Copy W into Z because this is a texture matrix:
        m[0][2] = m[0][3];
        m[1][2] = m[1][3];
        m[2][2] = 1;

        - MatrixTranslate does MatrixMultiply(m, T, out)=m*T (vmatrix.h:330)
        - VMatrix::Scale post-multiplies by diag(s) (vmatrix.cpp:442)

    (2) Shader sampling (compositor_vs20.fxc:35):

        texCoord = mul( float3( baseUV, 1 ), (float3x2)M )

        With HLSL column-major packing (default), the VMatrix's row r becomes
        HLSL column r. After the copy-W-into-Z step, m[i][2] holds the
        translated_xy. Expanding mul(vec, float3x2):

          texCoord.x = m[0][0]*u + m[0][1]*v + m[0][2]
          texCoord.y = m[1][0]*u + m[1][1]*v + m[1][2]

        Substituting R*S*T:
          m[0][0] =  cos*scale,  m[0][1] = -sin*scale,  m[0][2] = scale*(cos*tu - sin*tv)
          m[1][0] =  sin*scale,  m[1][1] =  cos*scale,  m[1][2] = scale*(sin*tu + cos*tv)

        Therefore the effective forward sample map is:

          pattern_uv = Rotate( rot, scale * (output_uv + (tu, tv)) )

        Rotation and scale happen about the ORIGIN (0,0), NOT the texture center.

    (3) This is the FORWARD transform: for each output pixel we compute where
        to sample the pattern. The pattern sampler uses WRAP mode so the
        pattern tiles across the output.

    With scale=[0.4, 0.5] the output UV range [0,1] maps to a pattern range of
    size 'scale', i.e. ~half the pattern is visible (the pattern appears 2x
    larger on the output).
    """
    W, H = out_size
    sW, sH = blood_img.size
    src = np.asarray(blood_img)  # (sH, sW, 4) RGBA

    # Output pixel -> normalized UV in [0, 1)
    yi, xi = np.mgrid[0:H, 0:W].astype(np.float32)
    u_out = xi / W
    v_out = yi / H

    rad = np.radians(rot_deg)
    c_r, s_r = np.cos(rad), np.sin(rad)

    # Source-derived forward transform (rotation/scale about UV origin).
    # pattern_uv = Rotate( scale * (output_uv + (tu, tv)) )
    u_t = u_out + u
    v_t = v_out + v
    u_s = u_t * scale
    v_s = v_t * scale
    u_src = u_s * c_r - v_s * s_r
    v_src = u_s * s_r + v_s * c_r

    # Wrap to [0, 1) - TF2 uses WRAP address mode for the pattern sampler,
    # so the pattern tiles across the output.
    u_src = np.mod(u_src, 1.0)
    v_src = np.mod(v_src, 1.0)

    # Sample with nearest (we're at 2048x2048, it's fine)
    sx = (u_src * sW).astype(np.int32) % sW
    sy = (v_src * sH).astype(np.int32) % sH

    out = src[sy, sx]
    return Image.fromarray(out, mode="RGBA")


# -- Compositor ----------------------------------------------------------------
def composite_blood_on_base(base_rgb: Image.Image,
                            blood_rgba: Image.Image) -> Image.Image:
    """
    Composite blood pattern onto the sniper rifle base texture using a
    multiply blend (matches TF2's combine_multiply operation).

    The blood VTF has red splatter on a near-white RGB with alpha showing the
    splatter mask. We use alpha as the blend weight: where alpha is 0 the base
    is unchanged; where alpha is high we tint the rifle with the blood color
    (multiply). This matches the in-engine look for paintkits using paint_blood
    (combine_multiply in paintkits_master.txt, see compositor_ps2x.fxc).
    """
    base = np.asarray(base_rgb.convert("RGB")).astype(np.float32) / 255.0

    bld_rgba = np.asarray(blood_rgba.convert("RGBA")).astype(np.float32) / 255.0
    bld_rgb  = bld_rgba[..., :3]
    bld_a    = bld_rgba[..., 3:4]  # shape (H,W,1)

    # Multiply result: base * blood_rgb. Then lerp to base by (1 - alpha) so
    # fully-transparent blood pixels show the base unchanged.
    multiplied = base * bld_rgb
    out_rgb    = base * (1.0 - bld_a) + multiplied * bld_a

    out = np.clip(out_rgb * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


# -- Main ----------------------------------------------------------------------
def render_for_wear(seed: int, wear: int, out_dir: str) -> str:
    r = compute_blood_uv(seed)
    # Round to 3 decimals (per user requirement)
    u     = round(r["u"],        3)
    v     = round(r["v"],        3)
    rot   = round(r["rotation"], 3)
    scale = round(r["scale"],    3)

    wear_name = WEAR_NAMES[wear]
    blood_path = WEAR_BLOOD_TEXTURE[wear]

    print(f"  [wear={wear} {wear_name:14s}] texture: {blood_path}")
    print(f"    u={u:.3f}  v={v:.3f}  rotation={rot:.3f}  scale={scale:.3f}")

    blood_img = load_vtf_as_pil(blood_path, mode="RGBA")
    base_img  = load_vtf_as_pil(SNIPER_RIFLE_TEX, mode="RGB")

    # Match base size (2048x2048)
    out_size = base_img.size

    blood_warped = apply_uv_transform(blood_img, u, v, rot, scale,
                                      out_size=out_size)

    composite = composite_blood_on_base(base_img, blood_warped)

    os.makedirs(out_dir, exist_ok=True)
    wear_slug = wear_name.replace(' ', '_').replace('-', '')
    fname = f"sniperrifle_blood_seed{seed}_wear{wear}_{wear_slug}.png"
    path  = os.path.join(out_dir, fname)
    composite.save(path)
    print(f"    wrote {path}")
    return path


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nNo seed given -- running ground truth demo with seeds 53, 72,"
              " and 4466973305.")
        seeds = [53, 72, 4466973305]
    else:
        seeds = [int(a) for a in sys.argv[1:]]

    for seed in seeds:
        r = compute_blood_uv(seed)
        print(f"\n=== seed {seed} ===")
        print(f"  seed_hi = {r['seed_hi']},  seed_lo = {r['seed_lo']}")
        print(f"  u={round(r['u'],3):.3f}  v={round(r['v'],3):.3f}  "
              f"rotation={round(r['rotation'],3):.3f}  "
              f"scale={round(r['scale'],3):.3f}")
        for wear in (3, 4, 5):  # Field-Tested, Well-Worn, Battle Scarred
            render_for_wear(seed, wear, OUTPUT_DIR)

    print("\nDone. Outputs in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
