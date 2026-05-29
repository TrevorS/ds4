#!/usr/bin/env python3
"""Parse DS4_MTP_TIMING=1 stderr into structured stats.

Reads stderr (file or stdin), extracts every `ds4: mtp timing ...` line, and
aggregates:

  * total_steps             — number of spec steps
  * total_drafted           — sum of drafted across steps
  * total_committed         — sum of committed across steps
  * accept_rate             — total_committed / total_drafted
  * tokens_emitted          — total_steps + total_committed (1 base + committed drafts per step)
  * committed_dist          — histogram {0: n, 1: n, 2: n, ...}
  * step_time_ms.{mean,p50,p90,p99}
  * step_kinds              — count per timing variant (combined, sample, decode2, margin-skip, micro)

Usage:
  parse_timing.py path/to/stderr.log       → markdown summary on stdout
  parse_timing.py --json out.json stderr   → also write JSON sidecar
  ... | parse_timing.py -                  → stdin mode
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

# Examples:
#   ds4: mtp timing combined drafted=2 committed=1 total=320.764 ms
#   ds4: mtp timing sample drafted=2 committed=2 resampled=0 total=123.671 ms
#   ds4: mtp timing decode2 drafted=2 committed=2 draft=12.0 ms snapshot=1.2 ms verify=80.4 ms total=93.6 ms
#   ds4: mtp timing margin-skip drafted=2 committed=1 margin=1.4 threshold=2.0 ...
#   ds4: mtp timing micro drafted=2 committed=1 ... total=X ms
RX = re.compile(
    r"^ds4: mtp timing (?P<kind>\S+).*?drafted=(?P<d>\d+).*?committed=(?P<c>\d+).*?total=(?P<t>[\d.]+)\s*ms"
)


def parse(stream):
    steps = []
    kinds: dict[str, int] = {}
    for line in stream:
        m = RX.search(line)
        if not m:
            continue
        steps.append(
            {
                "kind": m.group("kind"),
                "drafted": int(m.group("d")),
                "committed": int(m.group("c")),
                "total_ms": float(m.group("t")),
            }
        )
        kinds[m.group("kind")] = kinds.get(m.group("kind"), 0) + 1
    return steps, kinds


def percentile(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def aggregate(steps, kinds):
    if not steps:
        return {"total_steps": 0}
    total_d = sum(s["drafted"] for s in steps)
    total_c = sum(s["committed"] for s in steps)
    dist: dict[int, int] = {}
    for s in steps:
        dist[s["committed"]] = dist.get(s["committed"], 0) + 1
    times = [s["total_ms"] for s in steps]
    return {
        "total_steps": len(steps),
        "total_drafted": total_d,
        "total_committed": total_c,
        "accept_rate": (total_c / total_d) if total_d else 0.0,
        "tokens_emitted": len(steps) + total_c,
        "committed_dist": dist,
        "step_time_ms": {
            "mean": statistics.mean(times),
            "p50": percentile(times, 50),
            "p90": percentile(times, 90),
            "p99": percentile(times, 99),
        },
        "step_kinds": kinds,
        "implied_decode_tps": (
            (len(steps) + total_c) / (sum(times) / 1000.0) if sum(times) > 0 else 0.0
        ),
    }


def markdown(label: str, agg: dict) -> str:
    if agg["total_steps"] == 0:
        return f"# {label}\n\n_no mtp timing samples found_\n"
    dist = agg["committed_dist"]
    drafts = sorted(dist.keys())
    out = [
        f"# {label}",
        "",
        f"- spec steps: **{agg['total_steps']}**",
        f"- accept rate: **{agg['accept_rate'] * 100:.1f}%** ({agg['total_committed']}/{agg['total_drafted']})",
        f"- tokens emitted: {agg['tokens_emitted']}",
        f"- implied decode tps: **{agg['implied_decode_tps']:.2f}**",
        f"- step time ms: mean {agg['step_time_ms']['mean']:.1f}, p50 {agg['step_time_ms']['p50']:.1f}, p90 {agg['step_time_ms']['p90']:.1f}, p99 {agg['step_time_ms']['p99']:.1f}",
        "",
        "| committed | count | fraction |",
        "| ---------:| -----:| --------:|",
    ]
    total = sum(dist.values())
    for k in drafts:
        out.append(f"| {k} | {dist[k]} | {dist[k] / total * 100:.1f}% |")
    out.append("")
    if agg["step_kinds"]:
        out.append("kinds: " + ", ".join(f"{k}={v}" for k, v in agg["step_kinds"].items()))
    return "\n".join(out) + "\n"


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("input", help="path to stderr log, or - for stdin")
    p.add_argument("--label", default="mtp", help="label for the report header")
    p.add_argument("--json", default=None, help="write JSON sidecar here")
    args = p.parse_args(argv)

    if args.input == "-":
        steps, kinds = parse(sys.stdin)
    else:
        with open(args.input) as fp:
            steps, kinds = parse(fp)

    agg = aggregate(steps, kinds)
    print(markdown(args.label, agg))
    if args.json:
        Path(args.json).write_text(json.dumps({"label": args.label, **agg}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
