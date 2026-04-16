#!/usr/bin/env python3
"""
Scrape TF2 warpaint listings from the Steam Community Market and report
bloodscope coverage (FT + BS/WW) for each.

Pipeline
--------
  1. Fetch up to --count listings for a TF2 market item (public, no auth).
  2. Filter listings by wear level. Default includes FT/WW/BS (the only
     wears with visible blood; FN/MW have paint_blood's adjust_offset=0).
  3. Resolve each listing's paint_kit_seed:
       * If --seeds-csv is supplied, read seeds from there (listing_id,seed).
       * Otherwise, emit the Steam inspect URL so the user can resolve it.
  4. For each resolved seed, compute bloodscope coverage for Field-Tested
     and Battle-Scarred (WW is identical to BS -- see paintkit analysis in
     the README).
  5. Write a ranked CSV and print a summary.

Limitation: Steam's Market API does NOT expose paint_kit_seed in listing
JSON. The seed lives on the TF2 Game Coordinator and must be fetched via a
GC inspect request (which requires a logged-in Steam account running a
TF2-capable client). This script handles everything up to and after the
GC query, but the query itself is left to external tooling:

  - Manually: open each inspect URL in your TF2 client, then read the
    pattern/seed number shown in-game.
  - Automated: run a Steam GC bot (e.g. node-steam-user + node-tf2) that
    takes inspect URLs and emits listing_id,seed CSV rows -- then feed
    that CSV back in via --seeds-csv.

Usage
-----
  python market_bloodscope.py "Field-Tested Harvest Sniper Rifle"
  python market_bloodscope.py "Harvest Sniper Rifle" --count 100
  python market_bloodscope.py "Harvest Sniper Rifle" --wears 3,4,5 \
      --seeds-csv my_resolved_seeds.csv --out harvest_ranked.csv
"""

import argparse
import csv
import getpass
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path

import requests

# Reuse the verified Python bloodscope pipeline
REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR))
from paint_blood_circle_coverage import measure_coverage  # noqa: E402


# -- Wear mapping --------------------------------------------------------------
WEAR_NAMES = {
    1: "Factory New",
    2: "Minimal Wear",
    3: "Field-Tested",
    4: "Well-Worn",
    5: "Battle Scarred",
}
# TF2 warpaint market_hash_names end with "(Wear)" -- e.g.
#   "Bamboo Brushed Sniper Rifle (Field-Tested)"
# A few other items use the bare wear as a prefix; we match both forms.
WEAR_MATCH = [
    (1, ("Factory New",)),
    (2, ("Minimal Wear",)),
    (3, ("Field-Tested", "Field Tested")),
    (4, ("Well-Worn", "Well Worn")),
    (5, ("Battle Scarred", "Battle-Scarred", "Battle Scared")),
]


def detect_wear(name: str) -> int | None:
    for wnum, candidates in WEAR_MATCH:
        for c in candidates:
            if f"({c})" in name or name.startswith(f"{c} ") or name.endswith(f" {c}"):
                return wnum
    return None


def with_wear(base: str, wear: int) -> str:
    """Produce market_hash_name for a warpaint at a given wear."""
    return f"{base} ({WEAR_NAMES[wear]})"


# -- Steam Community Market scraper -------------------------------------------
STEAM_MARKET_BASE = "https://steamcommunity.com/market/listings/440"
TF2_APPID    = "440"
TF2_CONTEXT  = "2"


def _extract_js_object(html: str, var_name: str) -> dict | None:
    """Pull a `var X = {...};` JS object out of the market listing HTML.

    Parses the object greedily by brace balancing from the first `{` after
    the variable name. Returns a dict or None.
    """
    m = re.search(rf"\b{re.escape(var_name)}\s*=\s*", html)
    if not m:
        return None
    i = m.end()
    if i >= len(html) or html[i] != "{":
        return None
    # Brace-balanced extraction (skip string literals).
    depth      = 0
    in_str     = False
    str_ch     = ""
    esc        = False
    start      = i
    while i < len(html):
        ch = html[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == str_ch:
                in_str = False
        else:
            if ch in '"\'':
                in_str = True
                str_ch = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(html[start:i + 1])
                    except json.JSONDecodeError:
                        return None
        i += 1
    return None


def fetch_listings_page(item_name: str, start: int = 0, count: int = 100) -> dict:
    """GET the TF2 market listing HTML and extract its JS data objects.

    Returns a dict with `listinginfo` and `assets` keyed the same way the
    JSON `/render/` endpoint used to, so the rest of the code can stay
    agnostic to the source. `start`/`count` are currently ignored because
    Steam's HTML page returns the default page; we call it once per item
    and take the (usually) ~10-100 listings it ships embedded.
    """
    url = f"{STEAM_MARKET_BASE}/{urllib.parse.quote(item_name)}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (bloodscope-market-scraper; "
            "https://github.com/froginalog/b-scope-r)"
        ),
        "Accept": "text/html,application/xhtml+xml",
    }
    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code == 429:
        raise RuntimeError("Steam Market rate-limited (HTTP 429). Wait ~5 min.")
    resp.raise_for_status()

    html = resp.text
    listinginfo = _extract_js_object(html, "g_rgListingInfo") or {}
    assets      = _extract_js_object(html, "g_rgAssets")      or {}
    return {"listinginfo": listinginfo, "assets": assets}


def fetch_all_listings(item_name: str, total: int) -> list[dict]:
    """One request = one market page's worth of embedded listings.

    Steam pages typically show up to 10 listings per page in the HTML. For
    bigger pulls we'd need to drive `?start=N` on the HTML page too, but
    most warpaints have fewer than that anyway. Kept simple for now.

    The wear is taken from the item_name (the market page URL), NOT from
    the asset's JSON -- the asset name doesn't include the wear suffix.
    """
    raw       = fetch_listings_page(item_name)
    wear_num  = detect_wear(item_name)
    listings  = parse_listings(raw)
    for l in listings:
        # Stamp the wear from the query context, since the asset doesn't
        # carry it by itself.
        if wear_num is not None:
            l["wear"]      = wear_num
            l["wear_name"] = WEAR_NAMES[wear_num]
            # Make the displayed name include the wear so downstream tools
            # don't have to reconstruct it.
            if WEAR_NAMES[wear_num] not in l["name"]:
                l["name"] = f"{l['name']} ({WEAR_NAMES[wear_num]})"
    return listings[:total]


def parse_listings(raw: dict) -> list[dict]:
    """Pull (listing_id, asset_id, wear, price, inspect URL) per listing."""
    listings = []
    listinginfo = raw.get("listinginfo") or {}
    assets_root = raw.get("assets") or {}
    # When the market page has no listings, Steam ships `g_rgAssets = [];`
    if not isinstance(assets_root, dict):
        return listings
    assets_by_id = assets_root.get(TF2_APPID, {})
    if not isinstance(assets_by_id, dict):
        return listings
    assets_by_id = assets_by_id.get(TF2_CONTEXT, {})
    if not isinstance(assets_by_id, dict):
        assets_by_id = {}

    if not isinstance(listinginfo, dict):
        return listings

    for listing_id, info in listinginfo.items():
        asset    = (info.get("asset") or {}) if isinstance(info, dict) else {}
        asset_id = str(asset.get("id") or "")
        adata    = (assets_by_id.get(asset_id) or {}) if isinstance(assets_by_id, dict) else {}
        name     = adata.get("name") or adata.get("market_hash_name") or ""

        wear_num = detect_wear(name)

        inspect_url = ""
        for action in adata.get("actions") or []:
            link = (action or {}).get("link") or ""
            if "+tf_econ_item_preview" in link or "+csgo_econ_action_preview" in link:
                inspect_url = (link
                    .replace("%listingid%", str(listing_id))
                    .replace("%assetid%",   asset_id))
                break

        price_cents = ((info.get("converted_price") or 0) +
                       (info.get("converted_fee")   or 0))

        listings.append({
            "listing_id":  str(listing_id),
            "asset_id":    asset_id,
            "name":        name,
            "wear":        wear_num,
            "wear_name":   WEAR_NAMES.get(wear_num, "?"),
            "price_cents": price_cents,
            "inspect_url": inspect_url,
        })
    return listings


# -- Seeds CSV loader ---------------------------------------------------------
def load_seeds_csv(path: Path) -> dict[str, int]:
    """listing_id -> seed mapping. Accepts either 'listing_id' or 'listingid'."""
    out: dict[str, int] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lid  = str(row.get("listing_id") or row.get("listingid") or "").strip()
            seed = str(row.get("seed") or "").strip()
            if lid and seed:
                try:
                    out[lid] = int(seed)
                except ValueError:
                    pass
    return out


# -- Node.js resolver orchestration -------------------------------------------
RESOLVER_DIR    = REPO_DIR / "tools"
RESOLVER_JS     = RESOLVER_DIR / "resolve_seeds.js"
RESOLVER_NODEMO = RESOLVER_DIR / "node_modules"


def _ensure_resolver_installed() -> str:
    """Locate Node.js and install tools/ dependencies if needed.

    Returns the path to the node executable.
    """
    node = shutil.which("node")
    if not node:
        raise RuntimeError(
            "node.js is required for --auto-resolve. "
            "Install Node 18+ from https://nodejs.org and try again."
        )
    if not RESOLVER_JS.exists():
        raise RuntimeError(f"resolver script missing at {RESOLVER_JS}")
    if not RESOLVER_NODEMO.exists():
        npm = shutil.which("npm")
        if not npm:
            raise RuntimeError("npm is required to install resolver deps.")
        print(f"  installing resolver deps (first run)... ", end="", flush=True)
        r = subprocess.run(
            [npm, "install", "--no-audit", "--no-fund"],
            cwd=RESOLVER_DIR, capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"npm install failed:\n{r.stderr}")
        print("done.")
    return node


def run_auto_resolver(unresolved_rows: list[dict],
                      username: str,
                      password: str,
                      two_factor: str | None,
                      request_delay_ms: int = 2500) -> dict[str, int]:
    """Spawn the Node.js helper to inspect listings and return listing_id -> seed.

    Writes a temp inspect_urls.csv, runs node tools/resolve_seeds.js, reads
    back the seeds.csv. Streams the helper's stdout live so the user can
    watch progress.
    """
    node = _ensure_resolver_installed()

    with tempfile.TemporaryDirectory(prefix="bloodscope_gc_") as tmpdir:
        tmp = Path(tmpdir)
        in_csv  = tmp / "inspect_urls.csv"
        out_csv = tmp / "seeds.csv"
        config  = tmp / "config.json"

        with open(in_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["listing_id", "inspect_url"])
            w.writeheader()
            for r in unresolved_rows:
                if r.get("inspect_url"):
                    w.writerow({
                        "listing_id":  r["listing_id"],
                        "inspect_url": r["inspect_url"],
                    })

        cfg = {
            "username":         username,
            "password":         password,
            "input":            str(in_csv),
            "output":           str(out_csv),
            "request_delay_ms": int(request_delay_ms),
        }
        if two_factor:
            cfg["two_factor_code"] = two_factor
        config.write_text(json.dumps(cfg), encoding="utf-8")

        est_sec = (request_delay_ms / 1000.0) * max(1, len(unresolved_rows))
        print(f"  spawning Node.js resolver (≈{est_sec:.0f}s estimated "
              f"for {len(unresolved_rows)} items)...")
        print(f"  > node {RESOLVER_JS.name} <config.json>")
        print(f"  (resolver output streams live below)")
        print("  " + "-" * 70)
        sys.stdout.flush()

        # Stream Node's stdout+stderr to our stdout line-by-line so the
        # user sees progress as each request goes out.
        proc = subprocess.Popen(
            [node, str(RESOLVER_JS), str(config)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,              # line buffered
            universal_newlines=True,
            env={**os.environ, "NODE_NO_WARNINGS": "1"},
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print("  | " + line.rstrip(), flush=True)
        rc = proc.wait()
        print("  " + "-" * 70)
        if rc != 0:
            print(f"  WARNING: resolver exited with code {rc}", file=sys.stderr)

        if not out_csv.exists():
            return {}
        return load_seeds_csv(out_csv)


# -- CLI ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "item",
        help='Market item name, e.g. "Field-Tested Harvest Sniper Rifle". '
             "Case-sensitive, matches Steam's market_hash_name exactly.",
    )
    p.add_argument("--count", type=int, default=100,
                   help="Number of listings to pull (default 100).")
    p.add_argument("--wears", default="3,4,5",
                   help="Comma-separated wear levels to keep. "
                        "1=FN 2=MW 3=FT 4=WW 5=BS. Default 3,4,5.")
    p.add_argument("--seeds-csv", type=Path,
                   help="CSV with listing_id,seed rows of pre-resolved seeds.")
    p.add_argument("--out", type=Path, default=Path("market_bloodscopes.csv"),
                   help="Output CSV path (default market_bloodscopes.csv).")
    p.add_argument("--threshold", type=float, default=0.20,
                   help="Darkness threshold for 'bloody' pixel (default 0.20).")
    # --auto-resolve: spawn the Node.js TF2 GC client to fetch seeds
    p.add_argument("--auto-resolve", action="store_true",
                   help="Automatically resolve seeds via the Node.js TF2 GC "
                        "helper (requires Steam login + Node 18+).")
    p.add_argument("--steam-user", default=os.environ.get("STEAM_USER"),
                   help="Steam account name (or set $STEAM_USER).")
    p.add_argument("--steam-pass", default=os.environ.get("STEAM_PASS"),
                   help="Steam password (or set $STEAM_PASS). Prompted if omitted.")
    p.add_argument("--steam-2fa", default=os.environ.get("STEAM_2FA"),
                   help="Steam Guard mobile code (or set $STEAM_2FA).")
    p.add_argument("--gc-delay-ms", type=int, default=2500,
                   help="Delay between GC inspect requests (default 2500 ms, "
                        "matches the TF2 client's internal cooldown).")
    args = p.parse_args()

    wears = {int(w) for w in args.wears.split(",") if w.strip()}

    # 1. Scrape
    # TF2 warpaints are listed per-wear on the market, so if the item name
    # doesn't already include a wear prefix, fetch each wear separately.
    item_has_wear = detect_wear(args.item) is not None
    if item_has_wear:
        queries = [args.item]
    else:
        queries = [with_wear(args.item, w) for w in sorted(wears)]

    listings: list[dict] = []
    remaining = args.count
    for q in queries:
        if remaining <= 0:
            break
        per_q = remaining if item_has_wear else max(1, args.count // len(queries))
        print(f"Fetching up to {per_q} listings for {q!r}...")
        try:
            page = fetch_all_listings(q, per_q)
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else "?"
            if code == 404:
                print(f"  not found (404): {q}")
                continue
            print(f"  HTTP error: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"  got {len(page)} listings")
        listings.extend(page)
        remaining = args.count - len(listings)
        # Small pause between wear queries to stay polite
        if q != queries[-1]:
            time.sleep(1.0)

    if not listings:
        print("No listings returned. Verify the item name on Steam Market.",
              file=sys.stderr)
        sys.exit(1)

    # 2. Filter by wear
    filtered = [l for l in listings if l["wear"] in wears]
    wear_label = ", ".join(WEAR_NAMES[w] for w in sorted(wears))
    print(f"\n{len(filtered)}/{len(listings)} match wears {sorted(wears)} ({wear_label})")

    # 3. Resolve seeds from CSV if provided
    seeds_map: dict[str, int] = {}
    if args.seeds_csv:
        if not args.seeds_csv.exists():
            print(f"  warning: --seeds-csv {args.seeds_csv} not found; "
                  f"treating all listings as unresolved")
        else:
            seeds_map = load_seeds_csv(args.seeds_csv)
            print(f"  loaded {len(seeds_map)} seeds from {args.seeds_csv}")

    # 3b. Auto-resolve any still-unresolved listings via Node.js GC helper
    if args.auto_resolve:
        unresolved_rows = [l for l in filtered
                           if l["listing_id"] not in seeds_map
                           and l["inspect_url"]]
        if unresolved_rows:
            user = args.steam_user
            pw   = args.steam_pass
            if not user:
                user = input("Steam account name: ").strip()
            if not pw:
                pw = getpass.getpass("Steam password: ")
            two_fa = args.steam_2fa
            if not two_fa:
                maybe = input("Steam Guard mobile code (blank to skip): ").strip()
                two_fa = maybe if maybe else None
            try:
                resolved = run_auto_resolver(
                    unresolved_rows, user, pw, two_fa,
                    request_delay_ms=args.gc_delay_ms,
                )
            except RuntimeError as e:
                print(f"  auto-resolve failed: {e}", file=sys.stderr)
                resolved = {}
            seeds_map.update(resolved)
            print(f"  auto-resolved {len(resolved)}/{len(unresolved_rows)} seeds")

    # 4. Compute bloodscope for resolved seeds
    rows = []
    for i, l in enumerate(filtered, 1):
        seed   = seeds_map.get(l["listing_id"])
        ft_pct = bs_pct = None
        if seed is not None:
            print(f"    [{i}/{len(filtered)}] listing {l['listing_id']}  seed {seed} ...",
                  end="", flush=True)
            try:
                ft = measure_coverage(seed, 3, threshold=args.threshold)
                bs = measure_coverage(seed, 5, threshold=args.threshold)
                ft_pct = round(ft["coverage_pct"], 2)
                bs_pct = round(bs["coverage_pct"], 2)
                print(f"  FT={ft_pct}%  BS={bs_pct}%")
            except Exception as ex:
                print(f"  failed: {ex}")
        rows.append({**l, "seed": seed, "ft_pct": ft_pct, "bs_pct": bs_pct})

    # 5. Write CSV
    fieldnames = ["listing_id", "asset_id", "name", "wear", "wear_name",
                  "price_cents", "seed", "ft_pct", "bs_pct", "inspect_url"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {args.out} ({len(rows)} rows)")

    # 6. Human-readable summary
    resolved   = [r for r in rows if r["seed"] is not None]
    unresolved = [r for r in rows if r["seed"] is None]

    if resolved:
        resolved.sort(
            key=lambda r: max(r["ft_pct"] or 0, r["bs_pct"] or 0),
            reverse=True,
        )
        print(f"\n=== Top bloodscopes ({len(resolved)} resolved) ===")
        print(f"  {'wear':15s}  {'FT%':>6s}  {'BS%':>6s}  "
              f"{'seed':>15s}  {'price':>8s}  listing_id")
        for r in resolved[:25]:
            print(f"  {r['wear_name']:15s}  {r['ft_pct']:6.2f}  {r['bs_pct']:6.2f}  "
                  f"{r['seed']:>15d}  ${r['price_cents']/100:6.2f}  {r['listing_id']}")

    if unresolved:
        print(f"\n=== {len(unresolved)} listings need seed resolution ===")
        print("Steam Market does not expose paint_kit_seed; see module docstring.")
        print("Inspect URLs for the first 5 unresolved listings:")
        for r in unresolved[:5]:
            print(f"  listing {r['listing_id']}  {r['wear_name']:15s}  "
                  f"${r['price_cents']/100:6.2f}")
            if r["inspect_url"]:
                print(f"    {r['inspect_url']}")
        print("\nTo resolve: inspect each via TF2 or a Steam GC bot, save the")
        print("(listing_id, seed) pairs to a CSV, then rerun with --seeds-csv.")


if __name__ == "__main__":
    main()
