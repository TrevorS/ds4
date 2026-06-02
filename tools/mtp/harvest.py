"""Harvest FastMTP training data: drive `ds4 --mtp-harvest` ONCE (model loaded a
single time) over a corpus, then pack the per-doc (base_HC, token) shards to npz.

ds4 --mtp-harvest writes, per doc (one corpus line, raw-tokenized, fresh session
-> clean per-doc HCs): <out>/shard_NNNNN.mhcd (MHCD HC records) + .tok (token ids),
guaranteed 1:1 aligned (same prefill produces both). This packer converts each to
shard_NNNNN.npz {tokens int32[N], hc float32[N, hc_dim]} and removes the raw files.

The trainer forms FastMTP windows (h_i, t_{i+1..i+K}) -> targets t_{i+2..i+K+1},
skipping i=0 (BOS) and the last K positions.

Corpus: one document per line. HC stored fp32 (BOS/massive-activation channels
overflow fp16). Usage: harvest.py --corpus FILE --out DIR [--model ds4flash.gguf].
"""

import argparse
import glob
import json
import os
import struct
import subprocess
import sys

import numpy as np

ROOT = "/home/trevor/Projects/ds4"
DS4 = os.path.join(ROOT, "ds4")
MHCD_MAGIC = 0x4D484344


def pack_shard(mhcd_path):
    """Read shard.mhcd + shard.mhcd.tok -> (tokens, hc[N, hc_dim]) or (None, None)."""
    tok_path = mhcd_path + ".tok"
    if not os.path.exists(tok_path):
        return None, None
    td = open(tok_path, "rb").read()
    n = struct.unpack_from("<i", td, 0)[0]
    toks = np.array(struct.unpack_from("<%di" % n, td, 4), np.int32)
    data = open(mhcd_path, "rb").read()
    off, recs = 0, {}
    while off < len(data):
        magic, pos, hcd = struct.unpack_from("<III", data, off)
        off += 12
        if magic != MHCD_MAGIC:
            return None, None
        hc = np.frombuffer(data, np.float32, hcd, off).copy()
        off += hcd * 4
        recs[pos] = hc
    if not recs or len(recs) != n:
        return None, None
    return toks, np.stack([recs[i] for i in range(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="ds4flash.gguf")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # 1) one ds4 process, model loaded once, all docs.
    print("harvest: running ds4 --mtp-harvest (single model load) ...")
    env = dict(os.environ, DS4_CUDA_FAST_VERIFY="1")
    rc = subprocess.run(
        [
            DS4,
            "--cuda",
            "-m",
            args.model,
            "--mtp-harvest",
            os.path.abspath(args.corpus),
            os.path.abspath(args.out),
        ],
        cwd=ROOT,
        env=env,
    ).returncode
    if rc != 0:
        print(f"harvest: ds4 --mtp-harvest exited {rc}", file=sys.stderr)
        return 1

    # 2) pack raw shards -> npz, drop raw.
    manifest, total, hc_dim = [], 0, 0
    for mh in sorted(glob.glob(os.path.join(args.out, "shard_*.mhcd"))):
        toks, hc = pack_shard(mh)
        if toks is None or hc is None:
            print(f"  skip {os.path.basename(mh)} (align/parse)")
            continue
        npz = mh[: -len(".mhcd")] + ".npz"
        np.savez(npz, tokens=toks, hc=hc)
        manifest.append({"shard": os.path.basename(npz), "n": int(hc.shape[0])})
        total += hc.shape[0]
        hc_dim = hc.shape[1]
        os.unlink(mh)
        os.unlink(mh + ".tok")
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(
            {
                "shards": manifest,
                "total_positions": total,
                "hc_dim": hc_dim,
                "n_docs": len(manifest),
            },
            f,
            indent=2,
        )
    print(
        f"harvest done: {len(manifest)} docs, {total} positions, hc_dim={hc_dim} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
