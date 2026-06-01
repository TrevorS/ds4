#!/usr/bin/env python3
"""Diff two directories of CUDA fingerprint-tap (.fp) captures.

The taps (ds4_cuda.cu:ds4_gpu_fingerprint_tap_f32, gated by DS4_CUDA_TAP_PREFIX)
emit one line per (tag, layer, pos):

    tag=<tag> L=<n> pos=<n> n=<n> sum=<f> sumsq=<f> amax=<f> nan=<n> xor=0x<hex8>

Two gates:
  * strict  (byte_xor): must be bit-identical. This is the regression tripwire
    for refactors that CLAIM to be numerically no-ops (cache changes, scheduling,
    the MoE-filter restore, ...). Any xor delta means the bits changed.
  * soft    (rms/amax): sqrt(sumsq/n) is the per-tensor RMS magnitude; flag a
    relative drift beyond tolerance. Catches large numeric drift even when a
    benign reduction-order change has already tripped the strict xor gate.

NOTE: byte_xor is per-binary / per-arch (GB10 sm_121a) — golden files are not
portable across SM versions. nan_count > 0 in the candidate always fails.

Exit 0 = pass, 1 = fail, 2 = usage / missing-input error.
"""
import argparse
import math
import os
import sys
from glob import glob

FIELDS = ("sum", "sumsq", "amax", "nan", "xor")

# Tags that are NOT stable equality signals and must not be gated on.
# ffn_moe_gate taps g->routed_gate, a per-expert AUX scratch buffer that the MoE
# gate/up kernel writes only when write_aux is set (the debug/materialize path).
# In a normal decode forward write_aux is off, so the buffer is never written and
# the tap reads stale, run-to-run-nondeterministic memory (finite FLT_MAX + NaN
# tails). ds4.c documents the same at the C20 tap site ("scratch-noise — prefer
# the clean ffn_moe_out tap"). ffn_moe_out is the post-sum, fully-written signal.
DEFAULT_IGNORE_TAGS = {"ffn_moe_gate"}


def parse_line(line):
    """Parse one tap line into ((tag, L, pos), {field: value}). None if not a tap line."""
    line = line.strip()
    if not line.startswith("tag="):
        return None
    kv = {}
    for tok in line.split():
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        kv[k] = v
    try:
        key = (kv["tag"], int(kv["L"]), int(kv["pos"]))
        rec = {
            "n": int(kv["n"]),
            "sum": float(kv["sum"]),
            "sumsq": float(kv.get("sumsq", "nan")),
            "amax": float(kv["amax"]),
            "nan": int(kv["nan"]),
            "xor": int(kv["xor"], 16),
        }
    except (KeyError, ValueError):
        return None
    return key, rec


def load_dir(d, ignore_tags=frozenset()):
    """Load all .fp files in dir → ({(tag,L,pos): rec}, n_files, n_ignored).
    Last line per key wins (append mode). Records whose tag is in ignore_tags
    are dropped on both sides so they never affect the gate."""
    out = {}
    ignored = 0
    files = sorted(glob(os.path.join(d, "*.fp")))
    for path in files:
        with open(path) as fh:
            for line in fh:
                parsed = parse_line(line)
                if parsed is None:
                    continue
                key, rec = parsed
                if key[0] in ignore_tags:
                    ignored += 1
                    continue
                out[key] = rec  # later line wins
    return out, len(files), ignored


def rms(rec):
    if rec["n"] <= 0 or not math.isfinite(rec["sumsq"]):
        return float("nan")
    return math.sqrt(max(0.0, rec["sumsq"]) / rec["n"])


def rel(a, b):
    """Relative delta of a vs reference b."""
    denom = max(abs(b), 1e-12)
    return abs(a - b) / denom


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("golden_dir")
    ap.add_argument("candidate_dir")
    ap.add_argument("--rms-tol", type=float, default=1e-3,
                    help="relative RMS drift tolerance for the soft gate (default 1e-3)")
    ap.add_argument("--amax-tol", type=float, default=1e-3,
                    help="relative amax drift tolerance for the soft gate (default 1e-3)")
    ap.add_argument("--no-strict", action="store_true",
                    help="skip the bit-exact byte_xor gate (soft drift gate only)")
    ap.add_argument("--ignore-tag", action="append", default=[], metavar="TAG",
                    help="additionally drop this tag from the gate (repeatable)")
    ap.add_argument("--no-default-ignores", action="store_true",
                    help=f"do not auto-skip known scratch tags ({', '.join(sorted(DEFAULT_IGNORE_TAGS))})")
    args = ap.parse_args()

    ignore_tags = set(args.ignore_tag)
    if not args.no_default_ignores:
        ignore_tags |= DEFAULT_IGNORE_TAGS

    for d in (args.golden_dir, args.candidate_dir):
        if not os.path.isdir(d):
            print(f"cuda_tap_diff: not a directory: {d}", file=sys.stderr)
            return 2

    frozen_ignore = frozenset(ignore_tags)
    golden, gfiles, gign = load_dir(args.golden_dir, frozen_ignore)
    cand, cfiles, cign = load_dir(args.candidate_dir, frozen_ignore)

    if not golden:
        print(f"cuda_tap_diff: no tap records in golden dir {args.golden_dir} "
              f"({gfiles} .fp files)", file=sys.stderr)
        return 2
    if not cand:
        print(f"cuda_tap_diff: no tap records in candidate dir {args.candidate_dir} "
              f"({cfiles} .fp files)", file=sys.stderr)
        return 2

    gkeys, ckeys = set(golden), set(cand)
    missing = sorted(gkeys - ckeys)
    extra = sorted(ckeys - gkeys)
    shared = sorted(gkeys & ckeys)

    xor_fails, rms_fails, nan_fails = [], [], []
    worst_rms = (0.0, None)
    worst_amax = (0.0, None)

    for key in shared:
        g, c = golden[key], cand[key]
        if c["nan"] > 0:
            nan_fails.append((key, c["nan"]))
        if not args.no_strict and g["xor"] != c["xor"]:
            xor_fails.append(key)
        gr, cr = rms(g), rms(c)
        if math.isfinite(gr) and math.isfinite(cr):
            d = rel(cr, gr)
            if d > worst_rms[0]:
                worst_rms = (d, key)
            if d > args.rms_tol:
                rms_fails.append((key, gr, cr, d))
        da = rel(c["amax"], g["amax"])
        if da > worst_amax[0]:
            worst_amax = (da, key)

    def fmt_key(k):
        return f"{k[0]}/L{k[1]}/p{k[2]}"

    ign_note = f"  ignored={gign}/{cign} ({','.join(sorted(ignore_tags))})" if ignore_tags else ""
    print(f"cuda_tap_diff: golden={len(golden)} taps ({gfiles} files)  "
          f"candidate={len(cand)} taps ({cfiles} files)  shared={len(shared)}{ign_note}")
    if worst_rms[1]:
        print(f"  worst RMS drift : {worst_rms[0]:.3e} @ {fmt_key(worst_rms[1])}")
    if worst_amax[1]:
        print(f"  worst amax drift: {worst_amax[0]:.3e} @ {fmt_key(worst_amax[1])}")

    ok = True
    if missing:
        ok = False
        print(f"  FAIL: {len(missing)} golden taps absent in candidate "
              f"(first: {fmt_key(missing[0])})")
    if extra:
        # extra taps are a warning, not a hard fail (candidate may add coverage)
        print(f"  warn: {len(extra)} candidate taps absent in golden "
              f"(first: {fmt_key(extra[0])})")
    if nan_fails:
        ok = False
        print(f"  FAIL: {len(nan_fails)} taps with NaN/Inf in candidate "
              f"(first: {fmt_key(nan_fails[0][0])} nan={nan_fails[0][1]})")
    if xor_fails:
        ok = False
        print(f"  FAIL: {len(xor_fails)} taps differ bit-exactly (byte_xor) "
              f"— first diverging: {fmt_key(xor_fails[0])}")
    if rms_fails:
        ok = False
        rms_fails.sort(key=lambda r: -r[3])
        k, gr, cr, d = rms_fails[0]
        print(f"  FAIL: {len(rms_fails)} taps exceed rms-tol={args.rms_tol:.1e} "
              f"— worst {fmt_key(k)}: rms {gr:.6g} -> {cr:.6g} (rel {d:.3e})")

    print("cuda_tap_diff: " + ("OK" if ok else "MISMATCH"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
