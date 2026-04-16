#!/usr/bin/env python3
"""
Re-extract the bundled texture PNGs in ../assets/ from a TF2 install.

The repo ships with the PNGs pre-extracted so no TF2 install is required
at runtime. This script is here purely for reproducibility / asset
refresh. Requires:

    pip install vpk srctools

Usage:
    python tools/extract_assets.py
    python tools/extract_assets.py --tf2 "D:/Games/Team Fortress 2"
"""

import argparse
import io
import os
import sys

DEFAULT_TF2 = r"C:\Program Files (x86)\Steam\steamapps\common\Team Fortress 2"

SOURCES = {
    "materials/models/weapons/c_models/c_sniperrifle/c_sniperrifle.vtf":
        "c_sniperrifle.png",
    "materials/patterns/paint_blood.vtf":
        "paint_blood.png",
    "materials/patterns/paint_blood_buckets.vtf":
        "paint_blood_buckets.png",
    # Groups map is optional -- only used by tools/make_scope_reference.py
    # to visualise the scope region; the main pipeline doesn't load it.
    "materials/models/weapons/c_models/c_sniperrifle/p_sniperrifle_groups.vtf":
        "p_sniperrifle_groups.png",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tf2", default=DEFAULT_TF2,
                   help="Path to Team Fortress 2 install directory.")
    args = p.parse_args()

    try:
        import vpk
        from srctools.vtf import VTF
    except ImportError as e:
        print("Missing dependency:", e, file=sys.stderr)
        print("Install with: pip install vpk srctools", file=sys.stderr)
        sys.exit(2)

    vpk_path = os.path.join(args.tf2, "tf", "tf2_textures_dir.vpk")
    if not os.path.isfile(vpk_path):
        print(f"Could not find {vpk_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "assets")
    out_dir = os.path.normpath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    pak = vpk.open(vpk_path)
    for vpath, fname in SOURCES.items():
        data = pak.get_file(vpath).read()
        vtf  = VTF.read(io.BytesIO(data))
        frame = vtf.get()
        frame.load()
        img = frame.to_PIL().convert("RGBA")
        out = os.path.join(out_dir, fname)
        img.save(out, optimize=True)
        print(f"  {vpath}\n    -> {out}  ({img.size[0]}x{img.size[1]})")

    print("\nDone. Assets written to:", out_dir)


if __name__ == "__main__":
    main()
