"""gamut.bench — throughput matrix under a GPU/thermal monitor (replaces bench-with-monitor.sh).

Runs ds4-bench across a matrix of decode paths (plain / mtp-greedy / mtp-sample)
x N iters, back-to-back under one continuous GpuMonitor stream so per-cell
thermal/throttle signals and the transitions between them are visible. Each cell
is one ds4-bench invocation (a ctx-frontier sweep); results parse out of the
per-cell bench.csv. No bash, no respawn loops, no quoting traps.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .capture import ROOT, PERF
from .monitor import GpuMonitor

MODEL_DEFAULT = str(Path.home() / "models/ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf")
MTP_DEFAULT = str(Path.home() / "models/ds4/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf")
PROMPT_DEFAULT = str(ROOT / "tests/long_context_story_prompt.txt")

# (name, use_mtp, use_temp)
MATRIX_CELLS = [("plain", False, False), ("mtp-greedy", True, False), ("mtp-sample", True, True)]


@dataclass
class BenchCfg:
    label: str
    model: str = MODEL_DEFAULT
    mtp: str = MTP_DEFAULT
    prompt_file: str = PROMPT_DEFAULT
    matrix: bool = False
    use_mtp: bool = True
    use_temp: bool = False
    iters: int = 1
    ctx_start: int = 4096
    ctx_max: int = 32768
    step_mul: int = 2
    gen_tokens: int = 32
    fast_verify: bool = False
    extra_env: dict | None = None


def _cell_args(cfg: BenchCfg, use_mtp: bool, use_temp: bool, csv_path: str) -> list[str]:
    a = ["--prompt-file", cfg.prompt_file, "-m", cfg.model,
         "--ctx-start", str(cfg.ctx_start), "--ctx-max", str(cfg.ctx_max),
         "--step-mul", str(cfg.step_mul), "--gen-tokens", str(cfg.gen_tokens)]
    if use_mtp:
        a += ["--mtp", cfg.mtp, "--mtp-draft", "2"]
    if use_temp:
        a += ["--temp", "1.0", "--top-p", "0.95", "--seed", "1234"]
    a += ["--csv", csv_path]
    return a


def _parse_bench_csv(path: str) -> list[dict]:
    try:
        return [{k: _num(v) for k, v in row.items()} for row in csv.DictReader(open(path))]
    except OSError:
        return []


def _num(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return x


def _fit_prompt(prompt_file: str, ctx_max: int, out_dir: Path) -> str:
    """ds4-bench refuses to start when the prompt tokenizes to fewer tokens than
    --ctx-max ('prompt has N tokens, need at least --ctx-max=M'). The story prompt
    is ~30.5k tokens, short of the 32k frontier, so stitch copies into the run dir
    until it's long enough (rough ~5 bytes/token + slack). Ported from the legacy
    bench-with-monitor harness."""
    need_bytes = ctx_max * 5 + 8192
    try:
        src_bytes = os.path.getsize(prompt_file)
    except OSError:
        return prompt_file
    if src_bytes >= need_bytes or src_bytes == 0:
        return prompt_file
    copies = (need_bytes + src_bytes - 1) // src_bytes
    stitched = out_dir / "prompt.txt"
    data = Path(prompt_file).read_bytes()
    stitched.write_bytes(data * copies)
    print(f"## prompt stitched: {copies}x copies → {stitched} "
          f"({stitched.stat().st_size} bytes, target≈{ctx_max} tok)", flush=True)
    return str(stitched)


def run(cfg: BenchCfg) -> dict:
    out = PERF / "runs" / cfg.label
    out.mkdir(parents=True, exist_ok=True)
    ds4_bench = str(ROOT / "ds4-bench")
    Path("/tmp/ds4.lock").unlink(missing_ok=True)
    cfg.prompt_file = _fit_prompt(cfg.prompt_file, cfg.ctx_max, out)

    cells = MATRIX_CELLS if cfg.matrix else [("single", cfg.use_mtp, cfg.use_temp)]
    env = dict(os.environ)
    if cfg.fast_verify:
        env["DS4_CUDA_FAST_VERIFY"] = "1"
    if cfg.extra_env:
        env.update(cfg.extra_env)

    results: dict = {"label": cfg.label, "cells": {}}
    with GpuMonitor(str(out)) as mon:
        for name, use_mtp, use_temp in cells:
            cell_rows = []
            for it in range(1, cfg.iters + 1):
                stage = f"{name}#{it}"
                mon.set_stage(stage)
                cell_dir = out / name / f"iter-{it:03d}" if (cfg.matrix or cfg.iters > 1) else out
                cell_dir.mkdir(parents=True, exist_ok=True)
                csv_path = str(cell_dir / "bench.csv")
                log_path = str(cell_dir / "bench.log")
                print(f"## [{stage}] {time.strftime('%H:%M:%S')}", flush=True)
                with open(log_path, "w") as lf:
                    rc = subprocess.run([ds4_bench, *_cell_args(cfg, use_mtp, use_temp, csv_path)],
                                        env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
                if rc != 0:
                    print(f"## [{stage}] FAILED rc={rc} — continuing", flush=True)
                cell_rows.append({"iter": it, "rows": _parse_bench_csv(csv_path), "rc": rc})
                mon.set_stage("idle")
            results["cells"][name] = _aggregate_cell(cell_rows)

    summ = mon.summary()
    results["monitor"] = summ
    (out / "summary.json").write_text(json.dumps({"results": results, "monitor": summ}, indent=2))
    (out / "summary.txt").write_text(_render_summary(results, summ))
    return results


def _aggregate_cell(cell_rows: list[dict]) -> dict:
    """Mean gen_tps / prefill_tps per ctx across iters."""
    by_ctx: dict[int, dict[str, list[float]]] = {}
    for cr in cell_rows:
        for row in cr["rows"]:
            ctx = int(row.get("ctx_tokens", 0))
            d = by_ctx.setdefault(ctx, {"gen": [], "pf": []})
            if isinstance(row.get("gen_tps"), float):
                d["gen"].append(row["gen_tps"])
            if isinstance(row.get("prefill_tps"), float):
                d["pf"].append(row["prefill_tps"])
    out = {}
    for ctx, d in sorted(by_ctx.items()):
        out[ctx] = {"gen_tps": _mean(d["gen"]), "prefill_tps": _mean(d["pf"]),
                    "n": len(d["gen"])}
    return out


def _mean(xs):
    return round(sum(xs) / len(xs), 2) if xs else None


def _render_summary(results: dict, mon: dict) -> str:
    L = [f"# gamut bench · {results['label']}", ""]
    for name, ctxs in results["cells"].items():
        parts = [f"{c // 1024}k:{d['gen_tps']}(n{d['n']})" for c, d in ctxs.items()]
        L.append(f"{name:12} decode  " + "  ".join(parts))
    L.append("")
    g = mon.get("busy") or {}
    if g:
        L.append(f"GPU busy: sm_mean={_f(g.get('sm_mean'))}MHz peak={_f(g.get('sm_peak'))} "
                 f"power_mean={_f(g.get('power_mean'))}W temp_peak={_f(g.get('temp_peak'))}C")
        thr = g.get("throttled") or {}
        if thr:
            L.append("throttle (busy samples): " + ", ".join(f"{k}×{v}" for k, v in thr.items()))
    return "\n".join(L)


def _f(x):
    return f"{x:.0f}" if isinstance(x, (int, float)) else "—"
