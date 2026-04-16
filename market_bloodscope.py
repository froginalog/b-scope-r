#!/usr/bin/env python3
"""
Scan mannco.store listings for a given TF2 warpaint class/wear and report
bloodscope coverage (FT + BS/WW) for each listing.

Pipeline
--------
  1. Query mannco.store's /items/get for all warpaint products matching
     --class and --wears (e.g. Sniper + Field-Tested). Mannco ships this
     JSON endpoint unauthenticated; it returns products (aggregates of
     listings) with id, slug, price, inventory count, name.
  2. For each matching product, POST /requests/itemPage.php?mode=getItemList
     to fetch individual per-listing rows. Each row's "Inspect" button
     contains the full Steam inspect URL (S<seller>A<asset>D<dispatch>) --
     no Steam inventory lookup needed.
  3. For each listing, GET https://inspecttf2.accurateskins.com/?url=<inspect>
     which runs a legit TF2 GC inspect on their side and returns the item's
     CSOEconItem including original_id (= paint_kit_seed).
  4. Feed each seed into paint_blood_circle_coverage.measure_coverage for
     Field-Tested (paint_blood.png) and Battle-Scarred (paint_blood_buckets).
  5. Write a ranked CSV with clickable mannco product URLs.

Usage
-----
  python market_bloodscope.py --wears 3,4,5 --count 100
  python market_bloodscope.py --wears 3 --max-price 1.00 --auto-resolve
  python market_bloodscope.py --warpaint "Kiln and Conquer" --wears 3

Cloudflare
----------
mannco.store is Cloudflare-protected, so we use curl_cffi (chrome TLS
fingerprint impersonation) to get past the bot challenge. A normal
`requests.get` returns 403 "Just a moment..." from their edge.
"""

import argparse
import csv
import json
import re
import sys
import time
import urllib.parse
from pathlib import Path

try:
    from curl_cffi import requests as http
except ImportError:
    print("ERROR: curl_cffi required for mannco.store Cloudflare bypass.",
          file=sys.stderr)
    print("       pip install curl_cffi", file=sys.stderr)
    sys.exit(2)

# Reuse existing bloodscope pipeline
REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR))
from paint_blood_circle_coverage import measure_coverage_fast  # noqa: E402
from paint_blood_paintkits import (                               # noqa: E402
    BLOOD_PAINT_INDICES,
    DEF_INDEX_TO_PAINT_INDEX,
    NON_BLOOD_PAINT_INDICES,
)

# Default scale range used for community paintkits whose name we can't
# match against the master schema. 610 / 632 master paintkits DO render
# paint_blood and (0.4, 0.5) is the most common scale range, so this is
# a reasonable best guess.
DEFAULT_BLOOD_SCALE = (0.4, 0.5)


# -- Config --------------------------------------------------------------------
MANNCO_BASE      = "https://mannco.store"
MANNCO_SEARCH    = f"{MANNCO_BASE}/items/get"
MANNCO_LIST      = f"{MANNCO_BASE}/requests/itemPage.php?mode=getItemList"
INSPECT_API      = "https://inspecttf2.accurateskins.com/"
SEED_CACHE_PATH  = REPO_DIR / ".seed_cache.json"

WEAR_NAMES = {
    1: "Factory New",
    2: "Minimal Wear",
    3: "Field-Tested",
    4: "Well-Worn",
    5: "Battle Scarred",
}

# Classes that have paint_blood warpaints (really any TF2 class; Sniper is default)
CLASS_NAMES = {
    "scout", "Soldier", "Pyro", "Demoman", "Heavy", "Engineer",
    "Medic", "Sniper", "Spy",
}


# -- mannco.store scraping ----------------------------------------------------
def _make_session():
    s = http.Session(impersonate="chrome")
    # Warm the session with a hit to /tf2 so Cloudflare sets its __cf_bm cookie.
    s.get(f"{MANNCO_BASE}/tf2", timeout=30)
    return s


def search_products(session, tf2_class: str, wear_name: str,
                    max_pages: int = 6, max_price_cents: int | None = None
                    ) -> list[dict]:
    """Return all mannco TF2 products matching a class + wear.

    Walks pages sorted price-ASC until we've seen all matching warpaint
    products. mannco returns 50 per page; `max_pages=6` = up to 300
    products per query, well past the typical warpaint inventory size.
    """
    wanted_tag = f"({wear_name})"
    all_products: list[dict] = []
    seen: set[int] = set()
    for page in range(1, max_pages + 1):
        params = {
            "game":  440,
            "class": tf2_class,
            "wear":  wear_name,
            "page":  page,
            "skip":  (page - 1) * 50,
            "price": "ASC",
            "q":     "-",
        }
        resp = session.get(MANNCO_SEARCH, params=params, timeout=30)
        resp.raise_for_status()
        items = resp.json()
        if not items:
            break
        for it in items:
            if it["id"] in seen:
                continue
            name = it.get("name", "")
            # Keep only actual warpaint sniper rifles at the right wear
            if wanted_tag not in name:
                continue
            if "Sniper Rifle" not in name and "Rifle" not in name:
                continue
            price_c = int(it.get("price") or 0)
            if max_price_cents is not None and price_c > max_price_cents:
                continue
            it["_wear"] = wear_name
            seen.add(it["id"])
            all_products.append(it)
        if len(items) < 50:
            break
        time.sleep(0.5)
    return all_products


# Regex used to parse individual listing rows returned by itemPage.php.
# Each row has: itemid, itemprice, inspect URL with S<seller>A<asset>D<dispatch>.
_LISTING_RE = re.compile(
    r'itemid="(?P<mannco_id>\d+)"[^<]*?itemprice="(?P<price>\d+)"'
    r'(?:.|\n)*?steam://rungame/440/[^"\']*?'
    r'\+tf_econ_item_preview%20S(?P<seller>\d+)A(?P<asset>\d+)D(?P<dispatch>\d+)',
)


def fetch_product_listings(session, product_id: int, product_url: str,
                           product_name: str) -> list[dict]:
    """POST to mannco's per-item listings endpoint and parse each row."""
    referer = f"{MANNCO_BASE}/item/{product_url}"
    resp = session.post(
        MANNCO_LIST,
        data={"page": 1, "nbPerPage": 50, "id": product_id},
        headers={"Referer": referer, "X-Requested-With": "XMLHttpRequest"},
        timeout=30,
    )
    if resp.status_code != 200:
        return []
    html = resp.text
    listings = []
    for m in _LISTING_RE.finditer(html):
        inspect = (
            f"steam://rungame/440/76561202255233023/+tf_econ_item_preview "
            f"S{m['seller']}A{m['asset']}D{m['dispatch']}"
        )
        listings.append({
            "mannco_id":       m["mannco_id"],
            "product_id":      product_id,
            "product_url":     product_url,
            "product_name":    product_name,
            "price_cents":     int(m["price"]),
            "seller_steamid":  m["seller"],
            "asset_id":        m["asset"],
            "inspect_url":     inspect,
            "mannco_page_url": f"{MANNCO_BASE}/item/{product_url}",
        })
    return listings


# -- Seed resolution ----------------------------------------------------------
# Bumping this invalidates cached seeds from older versions of the tool.
# v1: original_id (WRONG for tool-applied warpaints)
# v2: custom_paintkit_seed / attrs 866+867 (correct for live TF2)
# v3: also stores paint_index so we can filter non-blood paintkits without
#     round-tripping the inspect API again
# v4: paint_index now uses DEF_INDEX_TO_PAINT_INDEX fallback for pre-built
#     paintkit items (attribute 834 is absent; static_attrs carries it)
SEED_CACHE_VERSION = 4


def _load_seed_cache() -> dict:
    if not SEED_CACHE_PATH.exists():
        return {}
    try:
        raw = json.loads(SEED_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    # Expect {"_version": N, "entries": {...}} for new caches; legacy format
    # was {inspect_url: {"seed": ...}} with no version key.
    if isinstance(raw, dict) and raw.get("_version") == SEED_CACHE_VERSION:
        return raw.get("entries") or {}
    # Old format or older version -- discard and start fresh.
    print(f"  cache version mismatch (found v{raw.get('_version') if isinstance(raw, dict) else '?'}, "
          f"need v{SEED_CACHE_VERSION}); discarding stale entries")
    return {}


def _save_seed_cache(cache: dict) -> None:
    payload = {"_version": SEED_CACHE_VERSION, "entries": cache}
    SEED_CACHE_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _extract_seed(econitem: dict) -> int | None:
    """Pick the right `paint_kit_seed` field off a CSOEconitem.

    Priority:
      1) `custom_paintkit_seed` (already-combined u64 of attrs 866 lo + 867 hi).
         This is what the live-game compositor uses for tool-applied war paints.
      2) Manually assemble from `attribute[]` def_indices 866/867 if the helper
         field is absent.
      3) Fall back to `original_id` for native crate-dropped items where no
         custom seed was written (the weapon's creation asset_id doubled as the
         paint seed in the pre-2020 code path).

    Returns None if nothing usable is present.
    """
    if not econitem:
        return None

    # 1) helper field (int-as-string, arbitrary precision)
    cps = econitem.get("custom_paintkit_seed")
    if cps not in (None, "", 0, "0"):
        try:
            return int(cps)
        except (TypeError, ValueError):
            pass

    # 2) assemble from raw attributes
    lo = hi = None
    for attr in econitem.get("attribute") or []:
        di = attr.get("def_index")
        vb = (attr.get("value_bytes") or {}).get("data")
        if not vb or len(vb) < 4:
            continue
        # value_bytes is always a little-endian u32 for these two attributes
        u32 = vb[0] | (vb[1] << 8) | (vb[2] << 16) | (vb[3] << 24)
        if di == 866:
            lo = u32
        elif di == 867:
            hi = u32
    if lo is not None:
        return ((hi or 0) << 32) | lo

    # 3) fall back to original_id
    oid = econitem.get("original_id")
    if oid not in (None, "", 0, "0"):
        try:
            return int(oid)
        except (TypeError, ValueError):
            pass
    return None


def resolve_seed(inspect_url: str, session) -> tuple[int | None, int | None]:
    """GET inspecttf2.accurateskins.com and return (paint_kit_seed, paint_index).

    paint_index (the paintkit_proto_def_index, attr def_index 834) identifies
    WHICH paintkit was applied -- we use it to filter out paintkits whose
    compositor doesn't actually render blood (or renders it with a different
    UV transform than our math is calibrated for). See BLOOD_PAINT_INDICES.
    """
    for attempt in range(3):
        try:
            r = session.get(INSPECT_API, params={"url": inspect_url},
                            timeout=30, impersonate="chrome")
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code != 200:
                return (None, None)
            body = r.json()
            if not body.get("success"):
                return (None, None)
            econ = body.get("item", {}).get("econitem") or {}
            seed = _extract_seed(econ)
            pindex = econ.get("paint_index")
            try:
                pindex = int(pindex) if pindex is not None else None
            except (TypeError, ValueError):
                pindex = None
            # Pre-built paintkit items (def_index 15000+) don't carry attr
            # 834 at runtime -- their paint_index lives in static_attrs. Use
            # the item's def_index to look it up.
            if pindex is None:
                try:
                    d = int(econ.get("def_index") or 0)
                except (TypeError, ValueError):
                    d = 0
                if d:
                    pindex = DEF_INDEX_TO_PAINT_INDEX.get(d)
            return (seed, pindex)
        except Exception:
            time.sleep(2 * (attempt + 1))
    return (None, None)


# -- CLI ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--class", dest="tf2_class", default="Sniper",
                   help=f"TF2 class. One of: {sorted(CLASS_NAMES)}. Default Sniper.")
    p.add_argument("--wears", default="3,4,5",
                   help="Comma-separated wears (1=FN 2=MW 3=FT 4=WW 5=BS). "
                        "Default 3,4,5 (the visible-blood wears).")
    p.add_argument("--count", type=int, default=100,
                   help="Max number of listings to check (default 100).")
    p.add_argument("--warpaint", default=None,
                   help="Optional warpaint name substring filter (e.g. 'Harvest').")
    p.add_argument("--max-price", type=float, default=None,
                   help="Skip products priced above $USD.")
    p.add_argument("--max-pages", type=int, default=6,
                   help="Max mannco catalog pages per wear (default 6 = 300 products).")
    p.add_argument("--auto-resolve", action="store_true",
                   help="Resolve seeds via inspecttf2.accurateskins.com and compute bloodscope.")
    p.add_argument("--api-delay", type=float, default=1.0,
                   help="Seconds between inspect-API calls (default 1.0).")
    p.add_argument("--threshold", type=float, default=0.20,
                   help="Darkness threshold for 'bloody' pixel.")
    p.add_argument("--verified-only", action="store_true",
                   help="Only include listings whose paintkit is in the "
                        "verified BLOOD_PAINT_INDICES set. Skips community "
                        "paintkits where the scale_uv range is inferred "
                        "(less accurate).")
    p.add_argument("--out", type=Path, default=Path("market_bloodscopes.csv"))
    args = p.parse_args()

    if args.tf2_class not in CLASS_NAMES:
        print(f"unknown --class {args.tf2_class!r}; choose from {sorted(CLASS_NAMES)}",
              file=sys.stderr)
        sys.exit(2)

    wear_nums = [int(w) for w in args.wears.split(",") if w.strip()]
    for w in wear_nums:
        if w not in WEAR_NAMES:
            print(f"unknown wear number {w}", file=sys.stderr)
            sys.exit(2)

    print(f"[1/4] mannco.store: searching {args.tf2_class} warpaints, "
          f"wears={[WEAR_NAMES[w] for w in wear_nums]}...")
    session = _make_session()
    max_price_c = int(args.max_price * 100) if args.max_price else None
    products: list[dict] = []
    for wnum in wear_nums:
        ps = search_products(
            session, args.tf2_class, WEAR_NAMES[wnum],
            max_pages=args.max_pages, max_price_cents=max_price_c,
        )
        if args.warpaint:
            needle = args.warpaint.lower()
            ps = [p for p in ps if needle in p["name"].lower()]
        print(f"      {WEAR_NAMES[wnum]:16s}  {len(ps):>4d} product(s), "
              f"total inventory: {sum(int(p['nbb']) for p in ps)} listings")
        products.extend(ps)

    if not products:
        print("      no warpaint products found for these filters.")
        sys.exit(0)

    print(f"[2/4] fetching individual listings per product "
          f"(up to {args.count} total)...")
    listings: list[dict] = []
    for i, prod in enumerate(products, 1):
        if len(listings) >= args.count:
            break
        rows = fetch_product_listings(
            session, prod["id"], prod["url"], prod["name"])
        # Stamp wear_num for each row
        wear_num = None
        for num, name in WEAR_NAMES.items():
            if f"({name})" in prod["name"]:
                wear_num = num
                break
        for r_ in rows:
            r_["wear_num"]  = wear_num
            r_["wear_name"] = WEAR_NAMES.get(wear_num, "?")
            listings.append(r_)
            if len(listings) >= args.count:
                break
        if i % 10 == 0 or i == len(products):
            print(f"      progress: scanned {i}/{len(products)} products, "
                  f"{len(listings)} listings so far")
        time.sleep(0.3)

    listings = listings[:args.count]
    print(f"      collected {len(listings)} listings")

    # Seed resolution + bloodscope
    if args.auto_resolve and listings:
        print(f"[3/4] resolving {len(listings)} seeds via "
              f"inspecttf2.accurateskins.com (~{args.api_delay}s apart)...")
        cache = _load_seed_cache()
        t0 = time.time()
        for i, l in enumerate(listings, 1):
            key = l["inspect_url"]
            entry = cache.get(key) or {}
            cached_seed   = entry.get("seed")
            cached_pindex = entry.get("paint_index")
            if cached_seed and cached_pindex is not None:
                l["seed"]        = int(cached_seed)
                l["paint_index"] = int(cached_pindex)
                print(f"      [{i:>3}/{len(listings)}] {l['mannco_id']:>10}  "
                      f"(cached)  seed={l['seed']}  pi={l['paint_index']}")
            else:
                t = time.time()
                seed, pindex = resolve_seed(key, session)
                ms = int((time.time() - t) * 1000)
                if seed is None:
                    l["seed"]        = None
                    l["paint_index"] = pindex
                    print(f"      [{i:>3}/{len(listings)}] {l['mannco_id']:>10}  "
                          f"({ms} ms) FAILED")
                else:
                    l["seed"]        = seed
                    l["paint_index"] = pindex
                    cache[key] = {"seed": seed, "paint_index": pindex}
                    print(f"      [{i:>3}/{len(listings)}] {l['mannco_id']:>10}  "
                          f"({ms} ms) seed={seed}  pi={pindex}")
                if i % 10 == 0:
                    _save_seed_cache(cache)
                time.sleep(args.api_delay)
        _save_seed_cache(cache)
        print(f"      resolved in {time.time() - t0:.1f}s")

        # Three-way paint_index classification:
        #   * BLOOD_PAINT_INDICES: verified blood paintkit with known
        #     scale_uv per wear; use that scale for maximum accuracy.
        #   * NON_BLOOD_PAINT_INDICES: verified non-blood (e.g. Bamboo
        #     Brushed). Skip -- the math would produce a phantom number.
        #   * Everything else (community paintkits with generic "Paintkit
        #     N" names in items_game): compute with DEFAULT_BLOOD_SCALE.
        #     610/632 master paintkits DO composite blood, so treating
        #     these as blood-candidates is the right default. Mark them
        #     `is_blood_paintkit='inferred'` so downstream users know the
        #     scale_uv guess might be slightly off.
        print("[4/4] computing bloodscope coverage "
              "(verified + inferred paintkits, fast path)...")
        t0 = time.time()
        skipped_non_blood = 0
        inferred = 0
        verified = 0
        for l in listings:
            if not l.get("seed"):
                continue
            pi = l.get("paint_index")
            if pi is not None and pi in NON_BLOOD_PAINT_INDICES:
                l["ft_pct"] = None
                l["bs_pct"] = None
                l["is_blood_paintkit"] = False
                l["paintkit_name"] = NON_BLOOD_PAINT_INDICES[pi]
                skipped_non_blood += 1
                continue
            blood = BLOOD_PAINT_INDICES.get(pi) if pi is not None else None
            if blood is not None:
                name, per_wear = blood
                l["is_blood_paintkit"] = True
                l["paintkit_name"]     = name
                ft_info = per_wear.get(3)
                bs_info = per_wear.get(5) or per_wear.get(4)
                verified += 1
            else:
                if args.verified_only:
                    # User wants accuracy -- skip unverified.
                    l["ft_pct"] = None
                    l["bs_pct"] = None
                    l["is_blood_paintkit"] = "inferred"
                    l["paintkit_name"]     = f"(unverified, paint_index {pi})"
                    inferred += 1
                    continue
                # Unknown / community paintkit -- infer as blood with
                # default scale at both wears.
                l["is_blood_paintkit"] = "inferred"
                l["paintkit_name"]     = f"(unverified, paint_index {pi})"
                ft_info = (DEFAULT_BLOOD_SCALE[0], DEFAULT_BLOOD_SCALE[1], "paint_blood")
                bs_info = (DEFAULT_BLOOD_SCALE[0], DEFAULT_BLOOD_SCALE[1], "paint_blood_buckets")
                inferred += 1
            try:
                if ft_info is not None:
                    ft_pct, _ = measure_coverage_fast(
                        l["seed"], threshold=args.threshold,
                        scale_range=(ft_info[0], ft_info[1]))
                    l["ft_pct"] = round(ft_pct, 2)
                else:
                    l["ft_pct"] = None
                if bs_info is not None:
                    _, bs_pct = measure_coverage_fast(
                        l["seed"], threshold=args.threshold,
                        scale_range=(bs_info[0], bs_info[1]))
                    l["bs_pct"] = round(bs_pct, 2)
                else:
                    l["bs_pct"] = None
            except Exception as e:
                print(f"      seed {l['seed']} bloodscope error: {e}")
                l["ft_pct"] = None
                l["bs_pct"] = None
        print(f"      done in {time.time() - t0:.2f}s  "
              f"(verified={verified}, inferred={inferred}, "
              f"non-blood skipped={skipped_non_blood})")

    # Stamp each listing with its "own-wear bloodscope": the coverage %
    # for the SAME wear the item is actually sold as. A Field-Tested
    # listing only gets credit for the FT-pattern coverage (paint_blood);
    # a Well-Worn or Battle-Scarred listing only gets credit for the
    # BS-pattern coverage (paint_blood_buckets, which WW and BS share).
    # Cross-wear numbers are irrelevant to the buyer and must not skew
    # the ranking.
    def _own_wear_score(l) -> float:
        # Listings stamp the wear number under "wear_num" (not "wear").
        w = l.get("wear_num")
        if w == 3:
            v = l.get("ft_pct")
        elif w in (4, 5):
            v = l.get("bs_pct")
        else:
            v = None
        return float(v) if v is not None else 0.0

    for l in listings:
        l["blood_score"] = _own_wear_score(l)

    # Sort listings by own-wear bloodscope descending before writing.
    listings.sort(key=lambda l: l.get("blood_score") or 0.0, reverse=True)

    # Output CSV
    columns = ["mannco_id", "product_name", "wear_name", "price_cents",
               "seed", "paint_index", "paintkit_name", "is_blood_paintkit",
               "ft_pct", "bs_pct", "blood_score", "mannco_page_url",
               "inspect_url", "seller_steamid", "asset_id"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for l in listings:
            w.writerow({c: l.get(c, "") for c in columns})
    print(f"\nwrote {args.out} ({len(listings)} rows, sorted by blood_score)")

    # Summary (full list, sorted by own-wear score descending)
    if args.auto_resolve:
        resolved = [l for l in listings
                    if l.get("is_blood_paintkit")
                    and l.get("blood_score") is not None
                    and (l.get("ft_pct") is not None
                         or l.get("bs_pct") is not None)]
        if resolved:
            # Already sorted by blood_score (done above). Show them all.
            print(f"\n=== All bloodscopes ({len(resolved)} listings, ranked by "
                  f"own-wear coverage) ===")
            print(f"  {'score':>6s}  {'FT%':>6s}  {'BS%':>6s}  {'wear':16s}  "
                  f"{'inf':>3s}  {'price':>8s}  {'seed':>20s}  {'paintkit':30s}  mannco page")
            for l in resolved:
                sc  = l.get("blood_score") or 0.0
                ft  = l.get("ft_pct")
                bs  = l.get("bs_pct")
                ft_s = f"{ft:6.2f}" if ft is not None else "    --"
                bs_s = f"{bs:6.2f}" if bs is not None else "    --"
                inf  = "?" if l.get("is_blood_paintkit") == "inferred" else " "
                pkn  = (l.get("paintkit_name") or "").replace(
                    "(unverified, paint_index ", "pi=").rstrip(")")
                print(f"  {sc:6.2f}  {ft_s}  {bs_s}  {l['wear_name']:16s}  "
                      f"{inf:>3s}  ${l['price_cents']/100:6.2f}  "
                      f"{l['seed']:>20d}  {pkn:30.30s}  "
                      f"{l['mannco_page_url']}")
            print("  'score' = coverage on the blood pattern THIS wear uses")
            print("           (FT -> paint_blood, WW/BS -> paint_blood_buckets)")
            print("  'FT%/BS%' = raw coverage under each pattern; 'score' picks "
                  "the one that matches the listing's wear")
            print("  '?'     = community paintkit, scale range inferred as "
                  "(0.4, 0.5) so the number may be slightly off")

        non_blood = [l for l in listings
                     if l.get("seed") is not None
                     and l.get("is_blood_paintkit") is False]
        if non_blood:
            print(f"\n  ({len(non_blood)} listing(s) skipped as verified NON-blood "
                  f"paintkits -- e.g. Bamboo Brushed, Civic Duty)")


if __name__ == "__main__":
    main()
