#!/usr/bin/env python3
"""
Walk every paintkit in _paintkits_master.txt and classify it by which
"blood compositor convention" the paintkit uses, then emit a flat text
file at paintkit_groups.txt.

Group A:  scale_uv = 0.4-0.5,  WW/BS wear switches to paint_blood_buckets.
          (Tough Break / Halloween / Pyroland / Smissmas-era collections.)
Group B:  scale_uv = 0.6-0.7,  paint_blood at all wears (no texture swap).
          (Original Gun Mettle 2015 collections.)
Other:    anything that doesn't match either convention.

The classification mirrors the two configs the Rust enumerator runs under
`--all-configs`. If a paintkit lands in "Other" it'll need bespoke
--scale-min/--scale-max/--bs-pattern flags rather than --all-configs.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO   = Path(__file__).resolve().parent.parent
MASTER = REPO / "_paintkits_master.txt"
OUT    = REPO / "paintkit_groups.txt"


# -- minimal KeyValues parser ------------------------------------------------
_TOK = re.compile(r'"((?:[^"\\]|\\.)*)"|([{}])|//[^\n]*', re.S)


def _tokens(text: str):
    for m in _TOK.finditer(text):
        s = m.group(0)
        if s.startswith("//"):
            continue
        if s in ("{", "}"):
            yield s
        else:
            yield m.group(1)


def parse_kv(text: str):
    toks = list(_tokens(text))
    i = [0]

    def parse_block():
        out = []
        while i[0] < len(toks):
            t = toks[i[0]]
            if t == "}":
                i[0] += 1
                return out
            key = t
            i[0] += 1
            if i[0] >= len(toks):
                return out
            nxt = toks[i[0]]
            if nxt == "{":
                i[0] += 1
                val = parse_block()
            elif nxt == "}":
                return out
            else:
                val = nxt
                i[0] += 1
            out.append((key, val))
        return out

    return parse_block()


def _child(block, key, default=None):
    if not isinstance(block, list):
        return default
    for k, v in block:
        if k == key:
            return v
    return default


# -- variable substitution (for community paintkits using $[var] templating) -
_VAR_REF_RE = re.compile(r'\$\[([A-Za-z_][A-Za-z0-9_]*)\]')


def _collect_vars(block, out=None):
    if out is None:
        out = {}
    if not isinstance(block, list):
        return out
    for k, v in block:
        if isinstance(k, str) and k.startswith("$") and isinstance(v, str):
            out[k[1:]] = v
        elif isinstance(v, list):
            _collect_vars(v, out)
    return out


def _resolve(value, vars_, depth=0):
    if not isinstance(value, str) or depth > 8:
        return value
    def repl(m):
        n = m.group(1)
        return _resolve(vars_[n], vars_, depth + 1) if n in vars_ else m.group(0)
    return _VAR_REF_RE.sub(repl, value)


def _rchild(block, key, vars_, default=None):
    v = _child(block, key, default)
    return _resolve(v, vars_) if isinstance(v, str) else v


# -- paintkit walker ---------------------------------------------------------
def _iter_top_level_paintkits(text):
    m = re.search(r'"item_paintkit_definitions"\s*\{', text)
    i = m.end() if m else 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == '"':
            j = text.index('"', i + 1)
            name = text[i + 1: j]
            k = j + 1
            while k < n and text[k] in " \t\r\n":
                k += 1
            if k >= n or text[k] != "{":
                i = j + 1
                continue
            body_start = k + 1
            d, p = 1, body_start
            while p < n and d > 0:
                ch = text[p]
                if ch == "{":
                    d += 1
                elif ch == "}":
                    d -= 1
                p += 1
            yield name, text[body_start:p - 1]
            i = p
            continue
        if c == "}":
            return
        i += 1


BLOOD_TEXTURES = ("patterns/paint_blood", "patterns/paint_blood_buckets")


def _adjust_offset_visible(tl_body, vars_):
    off = _rchild(tl_body, "adjust_offset", vars_)
    if not isinstance(off, str):
        return False
    for tok in off.split():
        try:
            if float(tok) != 0.0:
                return True
        except ValueError:
            continue
    return False


def _scale_range(tl_body, vars_):
    sc = _rchild(tl_body, "scale_uv", vars_)
    if not isinstance(sc, str):
        return None
    parts = sc.split()
    try:
        if len(parts) == 1:
            v = float(parts[0])
            return (v, v)
        if len(parts) == 2:
            return (float(parts[0]), float(parts[1]))
    except ValueError:
        return None
    return None


def _find_visible_blood(node, inside_select, vars_):
    """Return (scale_min, scale_max, texture_basename) of the FIRST visible
    paint_blood texture_lookup outside any `select`."""
    if not isinstance(node, list):
        return None
    for k, v in node:
        if k == "select":
            continue
        if k == "texture_lookup" and not inside_select:
            tex = _rchild(v, "texture", vars_, "") or ""
            if tex in BLOOD_TEXTURES and _adjust_offset_visible(v, vars_):
                rng = _scale_range(v, vars_)
                if rng is not None:
                    return (rng[0], rng[1], tex.rsplit("/", 1)[-1])
        if isinstance(v, list):
            got = _find_visible_blood(v, inside_select, vars_)
            if got is not None:
                return got
    return None


def per_wear_blood(body_text):
    parsed = parse_kv(body_text)
    paintkit_vars = _collect_vars(parsed)
    out = {}
    for wear in (3, 4, 5):
        wblock = _child(parsed, f"wear_level_{wear}")
        if not wblock:
            continue
        wv = dict(paintkit_vars)
        wv.update(_collect_vars(wblock))
        info = _find_visible_blood(wblock, inside_select=False, vars_=wv)
        if info:
            out[wear] = info
    return out


# -- main --------------------------------------------------------------------
WEAPONS = {
    "sniperrifle", "knife", "rocketlauncher", "grenadelauncher",
    "stickybomblauncher", "medigun", "flamethrower", "minigun",
    "scattergun", "pistol", "revolver", "shotgun", "smg",
    "wrench", "glove", "axe",
}


def main():
    text = MASTER.read_text(encoding="utf-8", errors="replace")
    rows = []  # (group, name, scale_str, bs_tex, weapon)
    skipped_no_blood = 0
    for name, body in _iter_top_level_paintkits(text):
        info = per_wear_blood(body)
        if 3 not in info:
            skipped_no_blood += 1
            continue
        smin, smax, ft_tex = info[3]
        bs_info = info.get(5) or info.get(4)
        bs_tex = bs_info[2] if bs_info else ft_tex

        weapon = "?"
        for tok in name.split("_"):
            if tok in WEAPONS:
                weapon = tok
                break

        # Finer grouping derived from what actually appears in master:
        #   A:  scale 0.4-0.5, BS = paint_blood_buckets   (modern collections)
        #   B:  scale 0.6-0.7, BS = paint_blood           (Gun Mettle sniper rifles)
        #   C:  scale 0.4-0.5, BS = paint_blood           (Gun Mettle non-pistol non-sniper)
        #   D:  scale 0.2-0.3, BS = paint_blood           (Gun Mettle pistols)
        #   OTHER: anything else
        sm_eq = lambda a, b: abs(a - b) < 1e-3
        if sm_eq(smin, 0.4) and sm_eq(smax, 0.5) and bs_tex == "paint_blood_buckets":
            group = "A"
        elif sm_eq(smin, 0.6) and sm_eq(smax, 0.7) and bs_tex == "paint_blood":
            group = "B"
        elif sm_eq(smin, 0.4) and sm_eq(smax, 0.5) and bs_tex == "paint_blood":
            group = "C"
        elif sm_eq(smin, 0.2) and sm_eq(smax, 0.3) and bs_tex == "paint_blood":
            group = "D"
        else:
            group = "OTHER"
        rows.append((group, name, f"{smin:.2f}-{smax:.2f}", bs_tex, weapon))

    counts = {"A": 0, "B": 0, "C": 0, "D": 0, "OTHER": 0}
    for r in rows:
        counts[r[0]] += 1

    lines = [
        "# TF2 War Paint paintkit -> group assignment",
        "# Auto-generated from _paintkits_master.txt by tools/gen_paintkit_groups.py",
        "#",
        "# Group A:  scale=0.4-0.5  BS=paint_blood_buckets",
        "#           Post-Gun-Mettle collections (Halloween / Tough Break /",
        "#           Pyroland / Smissmas / Mannversary / Workshop), the modern default.",
        "#",
        "# Group B:  scale=0.6-0.7  BS=paint_blood",
        "#           Gun Mettle 2015 sniper rifles only.",
        "#",
        "# Group C:  scale=0.4-0.5  BS=paint_blood",
        "#           Gun Mettle 2015 non-sniper non-pistol weapons.",
        "#",
        "# Group D:  scale=0.2-0.3  BS=paint_blood",
        "#           Gun Mettle 2015 pistols.",
        "#",
        "# Groups B/C/D never switch to paint_blood_buckets at WW/BS -- the",
        "# original Gun Mettle compositor stays on paint_blood at every wear.",
        "# Group A swaps to paint_blood_buckets at WW and BS for the gorier",
        "# wear-progression effect introduced in later updates.",
        "#",
        "# Other:    anything that doesn't match the above.",
        "",
        f"# Total paintkits with visible FT blood: {len(rows)}",
        f"# Group A:   {counts['A']}",
        f"# Group B:   {counts['B']}",
        f"# Group C:   {counts['C']}",
        f"# Group D:   {counts['D']}",
        f"# Other:     {counts['OTHER']}",
        f"# Skipped ({skipped_no_blood} paintkits with no visible FT blood)",
        "",
    ]
    headers = {
        "A":     f"=== Group A  ({counts['A']:>3} paintkits)  scale=0.4-0.5  BS=paint_blood_buckets  (post-Gun-Mettle) ===",
        "B":     f"=== Group B  ({counts['B']:>3} paintkits)  scale=0.6-0.7  BS=paint_blood          (Gun Mettle sniper rifles) ===",
        "C":     f"=== Group C  ({counts['C']:>3} paintkits)  scale=0.4-0.5  BS=paint_blood          (Gun Mettle non-pistol non-sniper) ===",
        "D":     f"=== Group D  ({counts['D']:>3} paintkits)  scale=0.2-0.3  BS=paint_blood          (Gun Mettle pistols) ===",
        "OTHER": f"=== Other    ({counts['OTHER']:>3} paintkits) -- non-standard config ===",
    }
    for grp in ("A", "B", "C", "D", "OTHER"):
        lines.append(headers[grp])
        sub = sorted(r for r in rows if r[0] == grp)
        for _, name, sc, bs, wpn in sub:
            if grp == "OTHER":
                lines.append(f"  {name:55s}  scale={sc:9s}  bs={bs:22s}  weapon={wpn}")
            else:
                lines.append(f"  {name:55s}  weapon={wpn}")
        lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"  Group A: {counts['A']}")
    print(f"  Group B: {counts['B']}")
    print(f"  Other:   {counts['OTHER']}")


if __name__ == "__main__":
    main()
