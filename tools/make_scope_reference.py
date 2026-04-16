#!/usr/bin/env python3
"""
Generate assets/scope_circle_reference.png -- a static illustration of the
circular region used by paint_blood_circle_coverage.py to measure blood
coverage on the sniper rifle scope lens.

The circle center/radius must exactly match the SCOPE_CX/CY/RADIUS constants
in paint_blood_circle_coverage.py.

Usage:
    python tools/make_scope_reference.py
"""

import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paint_blood_composite import SNIPER_RIFLE_TEX, ASSETS_DIR
from paint_blood_circle_coverage import SCOPE_CX, SCOPE_CY, SCOPE_RADIUS


def main():
    base = Image.open(SNIPER_RIFLE_TEX).convert("RGB")
    W, H = base.size

    # 1) Full-texture view with the circle outlined in yellow
    full = base.copy()
    draw = ImageDraw.Draw(full)
    draw.ellipse(
        [SCOPE_CX - SCOPE_RADIUS, SCOPE_CY - SCOPE_RADIUS,
         SCOPE_CX + SCOPE_RADIUS, SCOPE_CY + SCOPE_RADIUS],
        outline=(255, 220, 0), width=5,
    )
    label = (f"scope lens ROI: center=({SCOPE_CX}, {SCOPE_CY})  "
             f"radius={SCOPE_RADIUS}  ({W}x{H} texture)")
    draw.rectangle([20, 20, 20 + 11 * len(label) + 20, 60], fill=(0, 0, 0))
    draw.text((32, 30), label, fill=(255, 220, 0))
    full_path = os.path.join(ASSETS_DIR, "scope_circle_reference.png")
    full.save(full_path, optimize=True)
    print(f"  {full_path}  ({W}x{H})")

    # 2) Zoomed crop centered on the lens, for a clearer illustration
    pad = SCOPE_RADIUS * 2
    x0 = max(SCOPE_CX - pad, 0)
    y0 = max(SCOPE_CY - pad, 0)
    x1 = min(SCOPE_CX + pad, W)
    y1 = min(SCOPE_CY + pad, H)
    crop = base.crop((x0, y0, x1, y1))
    draw = ImageDraw.Draw(crop)
    draw.ellipse(
        [SCOPE_CX - x0 - SCOPE_RADIUS, SCOPE_CY - y0 - SCOPE_RADIUS,
         SCOPE_CX - x0 + SCOPE_RADIUS, SCOPE_CY - y0 + SCOPE_RADIUS],
        outline=(255, 220, 0), width=3,
    )
    crop_path = os.path.join(ASSETS_DIR, "scope_circle_reference_zoom.png")
    crop.save(crop_path, optimize=True)
    print(f"  {crop_path}  ({crop.size[0]}x{crop.size[1]})")


if __name__ == "__main__":
    main()
