#!/usr/bin/env python3
"""
Generate paint_blood_paintkits.py -- the set of TF2 paintkit indices whose
compositor VISIBLY renders paint_blood on the weapon body (and so onto the
Sniper Rifle scope lens) at Field-Tested or worse wear.

Filter (tighter than a plain substring match on `paint_blood`):

  A paintkit block in _paintkits_master.txt contains wear_level_{1..5}
  children. Each wear's compositor is a nested tree of combine_multiply,
  combine_lerp, select, and texture_lookup nodes. A paintkit counts as
  "visibly bloody at wear N" iff:

    1) A texture_lookup's `texture` is `patterns/paint_blood` or
       `patterns/paint_blood_buckets`.
    2) That texture_lookup has `adjust_offset` != 0 (`"0"` or `"0 0"`
       fades the layer to transparent, so it renders nothing).
    3) It is NOT nested inside a `select` ancestor. `select` blocks
       restrict the layer to a subset of group IDs (wooden stock, metal
       body parts, etc.) and never include the scope lens.

Wear 1 (Factory New) and Wear 2 (Minimal Wear) are skipped -- the shipped
paintkits set adjust_offset=0 on their blood layer, so blood isn't visible
at those wears in-game anyway. Wears 3/4/5 are the ones the player cares
about and the ones this tool measures.

Output: paint_blood_paintkits.py with
    BLOOD_PAINT_INDICES: dict[int, tuple[str, frozenset[int]]]
where the frozenset lists the wears (3, 4, 5) that qualify.

Usage:
    python tools/gen_blood_paintkits.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MASTER = REPO / "_paintkits_master.txt"
ITEMS_GAME = Path(
    r"C:\Program Files (x86)\Steam\steamapps\common"
    r"\Team Fortress 2\tf\scripts\items\items_game.txt"
)
OUT_FILE = REPO / "paint_blood_paintkits.py"

BLOOD_TEXTURES = ("patterns/paint_blood", "patterns/paint_blood_buckets")


# -----------------------------------------------------------------------------
# Minimal KeyValues parser
# -----------------------------------------------------------------------------
_TOK_RE = re.compile(r'"((?:[^"\\]|\\.)*)"|([{}])|//[^\n]*', re.S)


def _tokens(text: str):
    for m in _TOK_RE.finditer(text):
        whole = m.group(0)
        if whole.startswith("//"):
            continue
        if whole in ("{", "}"):
            yield whole
        else:
            yield m.group(1)


def parse_kv(text: str):
    """Parse a KeyValues fragment. Returns list of (key, value); value is
    either a string or another list."""
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


# -----------------------------------------------------------------------------
# Paintkits master: walk each wear's compositor tree
# -----------------------------------------------------------------------------
def _iter_top_level_paintkits(text: str):
    """Yield (name, body_text) for every paintkit inside the outer
    item_paintkit_definitions block."""
    m = re.search(r'"item_paintkit_definitions"\s*\{', text)
    if not m:
        # File might not have the wrapper -- treat the whole text as the
        # map of paintkits directly.
        i = 0
    else:
        i = m.end()
    n = len(text)

    while i < n:
        c = text[i]
        if c == '"':
            j = text.index('"', i + 1)
            name = text[i + 1:j]
            k = j + 1
            while k < n and text[k] in " \t\r\n":
                k += 1
            if k >= n or text[k] != "{":
                i = j + 1
                continue
            body_start = k + 1
            d = 1
            p = body_start
            while p < n and d > 0:
                ch = text[p]
                if ch == "{":
                    d += 1
                elif ch == "}":
                    d -= 1
                p += 1
            body_end = p - 1
            yield name, text[body_start:body_end]
            i = p
            continue
        if c == "}":
            # Closing the wrapper
            return
        i += 1


def _is_blood_texture(tl_body) -> bool:
    t = _child(tl_body, "texture", "") or ""
    return t in BLOOD_TEXTURES


def _adjust_offset_visible(tl_body) -> bool:
    off = _child(tl_body, "adjust_offset")
    if off is None:
        # Default is 0 -- blood invisible
        return False
    if not isinstance(off, str):
        return False
    # Can be "N" or "MIN MAX". Visible if any token parses as non-zero.
    for tok in off.split():
        try:
            if float(tok) != 0.0:
                return True
        except ValueError:
            continue
    return False


def _harvest_compatible_core(tl_body) -> bool:
    """Same translate/rotation ranges as Harvest. Scale is handled
    separately (returned per-paintkit so compute_blood_uv can be called
    with the right range)."""
    checks = {
        "translate_u": "0.0 1.0",
        "translate_v": "0.0 1.0",
        "rotation":    "0 360",
    }
    for key, want in checks.items():
        got = _child(tl_body, key)
        if not isinstance(got, str):
            return False
        if got.split() != want.split():
            return False
    # scale_uv must be present and parseable as 1 or 2 floats
    sc = _child(tl_body, "scale_uv")
    if not isinstance(sc, str):
        return False
    parts = sc.split()
    if len(parts) not in (1, 2):
        return False
    try:
        for p in parts:
            float(p)
    except ValueError:
        return False
    return True


def _scale_range(tl_body) -> tuple[float, float] | None:
    sc = _child(tl_body, "scale_uv")
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


def _walk_visible_blood_with_scale(
    node, inside_select: bool,
) -> tuple[float, float, str] | None:
    """Recursively walk the KV tree; return (scale_min, scale_max,
    texture_suffix) of the FIRST visible paint_blood texture_lookup
    (outside any `select`), where texture_suffix is 'paint_blood' or
    'paint_blood_buckets'. None if no visible blood layer."""
    if not isinstance(node, list):
        return None
    for k, v in node:
        if k == "select":
            continue
        if k == "texture_lookup" and not inside_select:
            if (_is_blood_texture(v)
                    and _adjust_offset_visible(v)
                    and _harvest_compatible_core(v)):
                rng = _scale_range(v)
                if rng is None:
                    continue
                tex = _child(v, "texture", "") or ""
                suffix = tex.rsplit("/", 1)[-1]
                return (rng[0], rng[1], suffix)
        if isinstance(v, list):
            got = _walk_visible_blood_with_scale(v, inside_select)
            if got is not None:
                return got
    return None


def paintkit_bloody_wears(body_text: str) -> dict[int, tuple[float, float, str]]:
    """Return {wear: (scale_min, scale_max, texture_suffix)} for each wear
    that has a visible, UV-compatible paint_blood layer."""
    parsed = parse_kv(body_text)
    out: dict[int, tuple[float, float, str]] = {}
    for wear in (3, 4, 5):
        wblock = _child(parsed, f"wear_level_{wear}")
        if not wblock:
            continue
        info = _walk_visible_blood_with_scale(wblock, inside_select=False)
        if info is not None:
            out[wear] = info
    return out


# -----------------------------------------------------------------------------
# items_game.txt: name -> paint_index
# -----------------------------------------------------------------------------
# Iterate each numeric item-definition block inside the "items" section
# of items_game.txt. Each block looks like:
#     "15000"
#     {
#         "name" "concealedkiller_sniperrifle_nightowl"
#         ...
#         "static_attrs"
#         {
#             "paintkit_proto_def_index" "14"
#         }
#     }
# Parsing per-block (rather than regex-greedy across the whole file) avoids
# a gnarly bug where `"name" "X"` at the top of some collection block gets
# paired with a `"paintkit_proto_def_index"` a million chars later, which
# the substring-based `"name" in gap` filter then incorrectly rejects.
_ITEM_DEF_OPEN_RE = re.compile(r'^\t*"(\d+)"\s*$', re.M)


def _extract_block(text: str, brace_open_pos: int) -> tuple[int, int]:
    """Given the position of a '{' in `text`, return (body_start, body_end)
    where the body is the text strictly between this brace and its match."""
    assert text[brace_open_pos] == "{"
    d = 1
    p = brace_open_pos + 1
    n = len(text)
    while p < n and d > 0:
        c = text[p]
        if c == "{":
            d += 1
        elif c == "}":
            d -= 1
        p += 1
    return (brace_open_pos + 1, p - 1)


def load_items_game_maps() -> tuple[dict[str, int], dict[int, int]]:
    """Walk each numeric item_definition block and extract:
      - name -> paintkit_proto_def_index
      - def_index -> paintkit_proto_def_index
    Second map is needed because PRE-BUILT painted items (def_index in
    15000+ range, e.g. 15000 = Night Owl Sniper Rifle) carry their
    paintkit_proto_def_index as a `static_attrs` child. Their inspect
    responses DON'T include attribute 834 at runtime, so we have to
    look the paint_index up by def_index.
    """
    if not ITEMS_GAME.is_file():
        print(f"ERROR: items_game.txt not found at {ITEMS_GAME}",
              file=sys.stderr)
        sys.exit(2)
    text = ITEMS_GAME.read_text(encoding="utf-8", errors="replace")

    name_to_idx: dict[str, int] = {}
    def_to_idx:  dict[int, int] = {}
    rows = 0
    for m in _ITEM_DEF_OPEN_RE.finditer(text):
        def_index = int(m.group(1))
        p = m.end()
        while p < len(text) and text[p] in " \t\r\n":
            p += 1
        if p >= len(text) or text[p] != "{":
            continue
        body_start, body_end = _extract_block(text, p)
        body = text[body_start:body_end]
        nm = re.search(r'"name"\s*"([A-Za-z0-9_\-]+)"', body)
        idm = re.search(r'"paintkit_proto_def_index"\s*"(\d+)"', body)
        if not nm or not idm:
            continue
        name = nm.group(1)
        paint_idx = int(idm.group(1))
        rows += 1
        name_to_idx.setdefault(name, paint_idx)
        def_to_idx[def_index] = paint_idx

    print(f"[items_game] {rows} item-def blocks w/ paintkit_proto_def_index, "
          f"{len(name_to_idx)} unique paintkit names, "
          f"{len(def_to_idx)} def_index->paint_index entries")
    return name_to_idx, def_to_idx


# Backwards-compatible alias for callers that only wanted the name map.
def load_name_to_index() -> dict[str, int]:
    name_map, _ = load_items_game_maps()
    return name_map


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print(f"reading {MASTER}")
    master_text = MASTER.read_text(encoding="utf-8", errors="replace")

    bloody_by_name: dict[str, dict[int, tuple[float, float, str]]] = {}
    total = 0
    for name, body in _iter_top_level_paintkits(master_text):
        total += 1
        info = paintkit_bloody_wears(body)
        if info:
            bloody_by_name[name] = info
    print(f"[master] scanned {total} paintkits, "
          f"{len(bloody_by_name)} have a visible UV-compatible blood layer at wear 3/4/5")

    print(f"\nreading {ITEMS_GAME}")
    name_to_idx, def_to_idx = load_items_game_maps()

    # Intersect
    idx_to_info: dict[int, tuple[str, dict[int, tuple[float, float, str]]]] = {}
    for name, per_wear in sorted(bloody_by_name.items()):
        idx = name_to_idx.get(name)
        if idx is None:
            continue
        if idx not in idx_to_info:
            idx_to_info[idx] = (name, dict(per_wear))
        else:
            existing_name, existing = idx_to_info[idx]
            # Union the wear info: multiple weapon item-defs share an index;
            # per-wear data should be identical across them for a given paintkit
            # so we just merge, preferring any wear that resolves.
            for w, v in per_wear.items():
                existing.setdefault(w, v)

    print(f"\n[final] {len(idx_to_info)} paint_indices qualify as blood paintkits")

    expectations = [
        (64,  True,  "harvest_*_boneyard family"),
        (306, False, "Bamboo Brushed (NOT blood)"),
        (7,   False, "Purple Range (wood paintkit)"),
        (14,  True,  "Night Owl -- scale 0.6-0.7"),
    ]
    for idx, expect_present, note in expectations:
        present = idx in idx_to_info
        tag = "OK" if present == expect_present else "MISMATCH"
        extra = ""
        if present:
            name, pw = idx_to_info[idx]
            wears_disp = ", ".join(
                f"w{w}={pw[w][0]:.2f}-{pw[w][1]:.2f}:{pw[w][2]}"
                for w in sorted(pw))
            extra = f"  [{wears_disp}]"
        print(f"  {tag:10s} idx={idx:>4}  present={present}"
              f"  (expected {expect_present}){extra}  -- {note}")

    print("\n[sample] first 20 qualifying paintkits:")
    for idx in sorted(idx_to_info)[:20]:
        name, pw = idx_to_info[idx]
        wd = " ".join(
            f"w{w}:s={pw[w][0]:.2f}-{pw[w][1]:.2f},{pw[w][2]}"
            for w in sorted(pw))
        print(f"  {idx:>4}: {name}  {wd}")

    # Write output.
    lines = [
        "# Auto-generated by tools/gen_blood_paintkits.py -- do not edit.",
        "#",
        "# Paintkits whose compositor visibly renders paint_blood /",
        "# paint_blood_buckets onto the weapon body (including the Sniper",
        "# Rifle scope lens) at Field-Tested or worse wear.",
        "#",
        "# Filter:",
        "#   1) there is a texture_lookup with texture patterns/paint_blood*",
        "#   2) it has adjust_offset != 0 (otherwise the layer is invisible)",
        "#   3) it's not nested inside a `select` block (which masks it to",
        "#      non-scope groups)",
        "#   4) translate_u/v=[0,1] and rotation=[0,360] (our compute_blood_uv",
        "#      assumes these; scale_uv is allowed to vary per paintkit and",
        "#      is stored below so compute_blood_uv can be called with the",
        "#      matching range).",
        "",
        "# paint_index (attr def_index 834 or static paintkit_proto_def_index)",
        "# -> (paintkit_internal_name,",
        "#      {wear_num: (scale_min, scale_max, texture_basename)})",
        "BloodPaintkitInfo = tuple[str, dict[int, tuple[float, float, str]]]",
        "BLOOD_PAINT_INDICES: dict[int, BloodPaintkitInfo] = {",
    ]
    for idx in sorted(idx_to_info.keys()):
        name, pw = idx_to_info[idx]
        wdict_parts = []
        for w in sorted(pw):
            smin, smax, tex = pw[w]
            wdict_parts.append(f"{w}: ({smin!r}, {smax!r}, {tex!r})")
        lines.append(f"    {idx:>4}: ({name!r}, {{{', '.join(wdict_parts)}}}),")
    lines.append("}")

    # Second map: def_index -> paint_index, for pre-built paintkit items
    # (def_index in the 15000+ range). The accurateskins inspect API
    # doesn't surface paint_index (attr 834) for these because it's in
    # static_attrs, not runtime attributes. We look it up by def_index.
    lines.append("")
    lines.append("# def_index (of the weapon/paintkit item) -> paint_index")
    lines.append("# Needed for pre-built paintkit items whose accurateskins")
    lines.append("# response carries def_index but no attr 834.")
    lines.append("DEF_INDEX_TO_PAINT_INDEX: dict[int, int] = {")
    for d in sorted(def_to_idx):
        lines.append(f"    {d:>6}: {def_to_idx[d]},")
    lines.append("}")

    OUT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT_FILE}")


if __name__ == "__main__":
    main()
