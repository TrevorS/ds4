"""Harvest FastMTP training data from ds4: for each corpus doc, prefill it through
ds4 (the only thing that forwards V4-Flash here) and capture per-position base HC
(combined_prev_hc) via DS4_MTP_HC_DUMP, aligned with the doc's token ids.

Output per doc: shard_NNNNN.npz {tokens int32[N], hc float16[N, hc_dim]}, position
i = (token t_i, base HC h_i after t_i). The trainer forms FastMTP windows
(h_i, t_{i+1..i+K}) -> targets t_{i+2..i+K+1}); it skips i=0 (BOS massive
activation) and the last K positions. This is FIXED-DATA harvest (raw corpus text,
no chat template) — fast (~350 t/s prefill); self-distill (gen first) is a later
swap of the corpus source.

Corpus: one document per line, OR a .jsonl with a "text" field.
Usage: harvest.py --corpus FILE --out DIR [--model ds4flash.gguf] [--max-docs N]
                   [--min-tokens 16] [--max-tokens 512]
Run with the venv python (numpy). ds4 binary must be built (make cuda-spark).
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

ROOT = "/home/trevor/Projects/ds4"
DS4 = os.path.join(ROOT, "ds4")
MHCD_MAGIC = 0x4D484344


def read_corpus(path):
    docs = []
    with open(path) as f:
        if path.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line)["text"])
        else:
            for line in f:
                line = line.rstrip("\n")
                if line.strip():
                    docs.append(line)
    return docs


def harvest_doc(model, text, hc_dim_out):
    """One ds4 prefill: returns (tokens list[int], hc array [N, hc_dim]) aligned.
    Tokens come from the .tok sidecar (exact post-template prefill ids); HCs from
    the MHCD dump — both from the SAME prefill, so alignment is guaranteed."""
    with tempfile.NamedTemporaryFile(suffix=".mhcd", delete=False) as tf:
        tmp = tf.name
    tok_path = tmp + ".tok"
    try:
        env = dict(os.environ, DS4_MTP_HC_DUMP=tmp, DS4_CUDA_FAST_VERIFY="1")
        subprocess.run([DS4, "--cuda", "-m", model, "-p", text, "-n", "1", "--temp", "0"],
                       capture_output=True, cwd=ROOT, env=env)
        data = open(tmp, "rb").read()
        tdata = open(tok_path, "rb").read() if os.path.exists(tok_path) else b""
    finally:
        for p in (tmp, tok_path):
            if os.path.exists(p):
                os.unlink(p)
    if not tdata:
        return None, None
    n = struct.unpack_from("<i", tdata, 0)[0]
    toks = list(struct.unpack_from("<%di" % n, tdata, 4))
    off, recs = 0, {}
    while off < len(data):
        magic, pos, hcd = struct.unpack_from("<III", data, off); off += 12
        if magic != MHCD_MAGIC:
            return None, None
        hc = np.frombuffer(data, np.float32, hcd, off).copy(); off += hcd * 4
        recs[pos] = hc
        hc_dim_out[0] = hcd
    if not recs:
        return None, None
    hc = np.stack([recs[i] for i in range(len(recs))])  # [N, hc_dim]
    return toks, hc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="ds4flash.gguf")
    ap.add_argument("--max-docs", type=int, default=0)
    ap.add_argument("--min-tokens", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    docs = read_corpus(args.corpus)
    if args.max_docs:
        docs = docs[:args.max_docs]
    print(f"harvest: {len(docs)} docs -> {args.out}")

    manifest, total_pos, kept = [], 0, 0
    hc_dim_box = [0]
    for k, text in enumerate(docs):
        toks, hc = harvest_doc(args.model, text, hc_dim_box)
        if toks is None or hc is None:
            print(f"  [{k}] skip (no data)"); continue
        if len(toks) < args.min_tokens:
            print(f"  [{k}] skip (tokens={len(toks)} < {args.min_tokens})"); continue
        if len(toks) > args.max_tokens:
            print(f"  [{k}] skip (>{args.max_tokens} tokens)"); continue
        if hc.shape[0] != len(toks):
            print(f"  [{k}] skip (align mismatch: {hc.shape[0]} hc vs {len(toks)} tok)"); continue
        path = os.path.join(args.out, f"shard_{k:05d}.npz")
        # fp32 (not fp16): BOS/massive-activation channels overflow fp16's 65504
        # max. bf16 would fit but numpy lacks a native dtype; fp32 is safe.
        np.savez(path, tokens=np.array(toks, np.int32), hc=hc.astype(np.float32))
        manifest.append({"shard": os.path.basename(path), "n": int(hc.shape[0])})
        total_pos += hc.shape[0]; kept += 1
        if kept % 20 == 0:
            print(f"  kept {kept} docs, {total_pos} positions")
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({"shards": manifest, "total_positions": total_pos,
                   "hc_dim": hc_dim_box[0], "n_docs_kept": kept}, f, indent=2)
    print(f"harvest done: {kept}/{len(docs)} docs, {total_pos} positions, hc_dim={hc_dim_box[0]}")


if __name__ == "__main__":
    sys.exit(main())
