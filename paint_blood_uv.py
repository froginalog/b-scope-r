#!/usr/bin/env python3
"""
TF2 paint_blood war paint UV transform calculator.

Confirmed pipeline (fully verified against ground truth):
=========================================================

1. uint64 nRandomSeed = paint_kit_seed (attribute value cast to uint64)
   In production: nRandomSeed = pItem->GetOriginalID()
   For simulation: use the paint_kit_seed attribute value directly.

2. GetSeed() bit-deinterleave (ctexturecompositor.cpp:1652):
     seedhi <- even-indexed bits of nRandomSeed  -> streams[0]
     seedlo <- odd-indexed  bits of nRandomSeed  -> streams[1]

3. streams[0].SetSeed(seedhi),  streams[1].SetSeed(seedlo)
   Both use Park-Miller LCG + Bays-Durham shuffle (vstdlib/random.cpp).

4. Stage tree (depth-first: self, first-child, next-sibling):
   Root combine          [stream 0]
   |- texture A          [stream 1, 7 RandomFloat calls: u,v,rot,scale,adjB,adjO,adjG]
   |- combine B          [stream 0]
      |- blood texture   [stream 1, picks up state after texture A's 7 calls]

5. Blood texture stage call order (ComputeRandomValuesThis, ctexturecompositor.cpp:576):
     [flipU - skipped, allow_flip_u=False]
     [flipV - skipped, allow_flip_v=False]
     translateU   RandomFloat(0.0, 1.0)
     translateV   RandomFloat(0.0, 1.0)
     rotation     RandomFloat(0.0, 360.0)    degrees
     scaleUV      RandomFloat(0.4, 0.5)      confirmed from GT
     adjustBlack  RandomFloat(0.0, 0.0)      -> 0
     adjustOffset RandomFloat(1.0, 1.0)      -> 1
     adjustGamma  RandomFloat(1.0, 1.0)      -> 1

   Note: C++ uses float32 arithmetic; use_float32=True for maximum accuracy.

Ground truth (loadouts.tf):
  seed 53 -> rotation=193.978  u=0.606  v=0.670  scale=0.429
  seed 72 -> rotation=21.384   u=0.662  v=0.767  scale=0.498

RNG sanity: set_seed(72) -> first RandomFloat(0,1) = 0.5430998
"""

import sys

# -- Source Engine RNG constants -----------------------------------------------
NTAB             = 32
IA               = 16_807
IM               = 2_147_483_647
IQ               = 127_773
IR               = 2_836
NDIV             = 1 + (IM - 1) // NTAB   # 67_108_864
AM               = 1.0 / IM
RNMX             = 1.0 - 1.2e-7
MAX_RANDOM_RANGE = 0x7FFF_FFFF

try:
    import numpy as np
    _HAVE_NUMPY = True
    _AM_F32   = np.float32(AM)
    _RNMX_F32 = np.float32(RNMX)
except ImportError:
    _HAVE_NUMPY = False


# -- Source Engine RNG ---------------------------------------------------------
class SourceRNG:
    """
    CUniformRandomStream (vstdlib/random.cpp).
    Park-Miller LCG + Bays-Durham shuffle table (Numerical Recipes ran1).
    """

    def __init__(self) -> None:
        self._idum: int  = 0
        self._iy:   int  = 0
        self._iv:   list = [0] * NTAB

    def set_seed(self, seed: int) -> None:
        # m_idum = (iSeed < 0) ? iSeed : -iSeed;  m_iy = 0;
        self._idum = seed if seed < 0 else -seed
        self._iy   = 0
        self._iv   = [0] * NTAB

    def _gen(self) -> int:
        if self._idum <= 0 or self._iy == 0:
            self._idum = 1 if -self._idum < 1 else -self._idum
            for j in range(NTAB + 7, -1, -1):
                k = self._idum // IQ
                self._idum = IA * (self._idum - k * IQ) - IR * k
                if self._idum < 0:
                    self._idum += IM
                if j < NTAB:
                    self._iv[j] = self._idum
            self._iy = self._iv[0]

        k = self._idum // IQ
        self._idum = IA * (self._idum - k * IQ) - IR * k
        if self._idum < 0:
            self._idum += IM

        j = self._iy // NDIV
        if j >= NTAB or j < 0:
            j &= NTAB - 1
        self._iy    = self._iv[j]
        self._iv[j] = self._idum
        return self._iy

    def random_float(self, lo: float = 0.0, hi: float = 1.0,
                     use_float32: bool = False) -> float:
        raw = self._gen()
        if use_float32 and _HAVE_NUMPY:
            fl = float(_AM_F32 * raw)
            if fl > float(_RNMX_F32):
                fl = float(_RNMX_F32)
            return float(np.float32(fl) * np.float32(hi - lo) + np.float32(lo))
        else:
            fl = AM * raw
            if fl > RNMX:
                fl = RNMX
            return fl * (hi - lo) + lo

    def random_int(self, lo: int, hi: int) -> int:
        x = hi - lo + 1
        if x <= 1 or MAX_RANDOM_RANGE < x - 1:
            return lo
        max_ok = MAX_RANDOM_RANGE - (MAX_RANDOM_RANGE + 1) % x
        while True:
            n = self._gen()
            if n <= max_ok:
                return lo + n % x


# -- GetSeed bit-deinterleave --------------------------------------------------
def get_seed(n_random_seed: int):
    """
    CTextureCompositor::GetSeed (ctexturecompositor.cpp:1652).

    Even-indexed bits -> seedhi -> streams[0]
    Odd-indexed  bits -> seedlo -> streams[1]
    """
    hi = lo = 0
    for i in range(32):
        hi |= (n_random_seed & (1 << (2 * i)))     >> i
        lo |= (n_random_seed & (1 << (2 * i + 1))) >> (i + 1)
    return hi & 0xFFFF_FFFF, lo & 0xFFFF_FFFF


def _s32(u: int) -> int:
    """uint32 -> signed int32 (SetSeed takes signed int in C++)."""
    return u - 0x1_0000_0000 if u > 0x7FFF_FFFF else u


# -- Confirmed paint_blood config ----------------------------------------------
# Derived from ground truth (seeds 53 and 72) + ctexturecompositor.cpp source.
# Ranges that differ from (0,0) or (1,1) must be extracted from TF2 VPK for
# other paint kits; for paint_blood these are confirmed.
BLOOD_STREAM_IDX  = 1          # stream[1] (seeded with seedlo = odd bits)
BLOOD_PRE_CALLS   = 7          # 7 float calls consumed by texture-A on stream[1]
                               # before blood texture stage runs

# Compositor KeyValues ranges for the blood texture stage:
BLOOD_TRANSLATE_U   = (0.0, 1.0)
BLOOD_TRANSLATE_V   = (0.0, 1.0)
BLOOD_ROTATION      = (0.0, 360.0)   # degrees
BLOOD_SCALE_UV      = (0.4, 0.5)     # confirmed: clean [0.4, 0.5] range
BLOOD_ADJUST_BLACK  = (0.0, 0.0)
BLOOD_ADJUST_OFFSET = (1.0, 1.0)
BLOOD_ADJUST_GAMMA  = (1.0, 1.0)
BLOOD_ALLOW_FLIP_U  = False
BLOOD_ALLOW_FLIP_V  = False


# -- Core compute --------------------------------------------------------------
def compute_blood_uv(paint_kit_seed: int, use_float32: bool = True) -> dict:
    """
    Compute the paint_blood UV transform for the given paint_kit_seed.

    paint_kit_seed : integer value of the 'paint_kit_seed' item attribute.
    use_float32    : match C++ float32 precision (recommended: True).

    Returns a dict with keys:
      seed_hi, seed_lo  - deinterleaved 32-bit seeds
      u, v              - translateU, translateV  in [0, 1]
      rotation          - rotation in degrees [0, 360)
      scale             - scaleUV in [0.4, 0.5]
      adjust_black      - 0.0 (fixed for blood)
      adjust_white      - 1.0 (= black + offset, fixed)
      adjust_gamma      - 1.0 (fixed for blood)
      flip_u, flip_v    - False (not enabled for blood)
    """
    n64 = paint_kit_seed & 0xFFFF_FFFF_FFFF_FFFF
    seed_hi, seed_lo = get_seed(n64)

    streams = [SourceRNG(), SourceRNG()]
    streams[0].set_seed(_s32(seed_hi))
    streams[1].set_seed(_s32(seed_lo))

    rng = streams[BLOOD_STREAM_IDX]
    f32 = use_float32

    # Consume BLOOD_PRE_CALLS calls (texture-A stage on same stream)
    for _ in range(BLOOD_PRE_CALLS):
        rng.random_float(0.0, 1.0, use_float32=f32)

    # Blood texture stage: ComputeRandomValuesThis call order
    # flipU / flipV: not enabled, no RNG calls
    u     = rng.random_float(*BLOOD_TRANSLATE_U,   use_float32=f32)
    v     = rng.random_float(*BLOOD_TRANSLATE_V,   use_float32=f32)
    rot   = rng.random_float(*BLOOD_ROTATION,      use_float32=f32)
    scale = rng.random_float(*BLOOD_SCALE_UV,      use_float32=f32)
    adj_b = rng.random_float(*BLOOD_ADJUST_BLACK,  use_float32=f32)
    adj_o = rng.random_float(*BLOOD_ADJUST_OFFSET, use_float32=f32)
    adj_g = rng.random_float(*BLOOD_ADJUST_GAMMA,  use_float32=f32)

    return dict(
        paint_kit_seed = paint_kit_seed,
        seed_hi        = seed_hi,
        seed_lo        = seed_lo,
        u              = u,
        v              = v,
        rotation       = rot,
        scale          = scale,
        adjust_black   = adj_b,
        adjust_white   = adj_b + adj_o,
        adjust_gamma   = adj_g,
        flip_u         = False,
        flip_v         = False,
    )


# -- Ground truth & verification -----------------------------------------------
GROUND_TRUTH = {
    53: dict(rotation=193.978, u=0.606, v=0.670, scale=0.429),
    72: dict(rotation=21.384,  u=0.662, v=0.767, scale=0.498),
}


def verify_rng() -> bool:
    """set_seed(72) -> first RandomFloat(0,1) = 0.5430998  (Park-Miller sanity)."""
    rng = SourceRNG()
    rng.set_seed(72)
    f = rng.random_float(0.0, 1.0, use_float32=False)
    ok = abs(f - 0.5430998) < 1e-5
    print("RNG sanity: set_seed(72) -> %.7f  [%s]" % (f, "OK" if ok else "FAIL"))
    return ok


def verify_ground_truth(use_float32: bool = True) -> bool:
    tol = {"u": 0.002, "v": 0.002, "rotation": 0.05, "scale": 0.002}
    all_ok = True
    for pks, gt in GROUND_TRUTH.items():
        r = compute_blood_uv(pks, use_float32=use_float32)
        parts = ["seed %2d:" % pks]
        for key in ("u", "v", "rotation", "scale"):
            got = round(r[key], 3)
            exp = gt[key]
            ok  = abs(got - exp) < tol[key]
            if not ok:
                all_ok = False
            parts.append("  %s=%.3f(%s%.3f)" % (key, got, "==" if ok else "!=", exp))
        print("".join(parts))
    return all_ok


# -- Main ----------------------------------------------------------------------
def main() -> None:
    print("=== TF2 paint_blood UV Transform Calculator ===\n")

    verify_rng()
    print()

    print("=== Ground Truth Verification ===")
    ok = verify_ground_truth(use_float32=_HAVE_NUMPY)
    print("Result: %s\n" % ("ALL MATCH" if ok else "mismatch - check config"))

    # Command-line seeds
    if len(sys.argv) > 1:
        print("=== Computed Seeds ===")
        for arg in sys.argv[1:]:
            pks = int(arg)
            r = compute_blood_uv(pks)
            print(
                "  seed %6d:  rotation=%7.3f  u=%.3f  v=%.3f  scale=%.3f"
                "  (seedhi=%d, seedlo=%d)" % (
                    pks, round(r["rotation"], 3), round(r["u"], 3),
                    round(r["v"], 3), round(r["scale"], 3),
                    r["seed_hi"], r["seed_lo"])
            )
    else:
        # Demo: print a range of seeds
        print("=== Sample Seeds ===")
        print("  %-8s  %-9s  %-7s  %-7s  %-7s" % (
            "seed", "rotation", "u", "v", "scale"))
        for pks in list(range(1, 11)) + [53, 72, 100, 1000]:
            r = compute_blood_uv(pks)
            print("  %-8d  %-9.3f  %-7.3f  %-7.3f  %-7.3f" % (
                pks, round(r["rotation"], 3), round(r["u"], 3),
                round(r["v"], 3), round(r["scale"], 3)))


if __name__ == "__main__":
    main()
