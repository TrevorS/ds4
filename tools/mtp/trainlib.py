"""Training harness helpers for the FastMTP fine-tune: offline-capable wandb,
GPU/thermal monitoring (GB10 hard-off survival), resumable checkpointing, seeding,
and a non-finite guard. Kept separate so train_mtp.py stays focused on the math.
"""

import os
import random
import subprocess
import time

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gpu_stats() -> dict:
    """nvidia-smi temp/util/power (memory is 'Not Supported' on GB10)."""
    try:
        out = (
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu,utilization.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            .stdout.strip()
            .split(",")
        )
        return {
            "gpu_temp": float(out[0]),
            "gpu_util": float(out[1]),
            "gpu_power": float(out[2]),
        }
    except Exception:
        return {}


class ThermalGuard:
    """GB10 long-run soak can hard-power-off the box. Before a step, if temp is
    over max_c, block (logging) until it cools to cool_c. Returns paused seconds."""

    def __init__(self, max_c=84.0, cool_c=70.0, poll_s=5.0):
        self.max_c, self.cool_c, self.poll_s = max_c, cool_c, poll_s

    def maybe_cooldown(self) -> float:
        t = gpu_stats().get("gpu_temp")
        if t is None or t < self.max_c:
            return 0.0
        t0 = time.time()
        print(
            f"\n[thermal] {t:.0f}C >= {self.max_c:.0f}C — cooling to {self.cool_c:.0f}C ...",
            flush=True,
        )
        while True:
            time.sleep(self.poll_s)
            t = gpu_stats().get("gpu_temp", 0.0)
            if t <= self.cool_c:
                break
        dt = time.time() - t0
        print(f"[thermal] cooled to {t:.0f}C in {dt:.0f}s", flush=True)
        return dt


def init_wandb(
    project: str,
    name: str,
    config: dict,
    run_id: str | None = None,
    mode: str = "offline",
):
    """Offline by default (no auth needed; sync later with `wandb sync`)."""
    import wandb

    os.environ.setdefault("WANDB_MODE", mode)
    run = wandb.init(
        project=project, name=name, config=config, id=run_id, resume="allow"
    )
    return run, wandb


def is_finite(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all())


class Checkpointer:
    """Resumable checkpoints (trainable weights + optimizer + step + RNG) plus a
    best-by-metric copy. Frequent saves = a thermal hard-off costs minutes."""

    def __init__(self, out_dir: str, keep_last: int = 3):
        self.dir = out_dir
        self.keep = keep_last
        os.makedirs(out_dir, exist_ok=True)
        self.best_metric = None
        self.saved = []

    def _state(self, head, opt, step, epoch, config, metric):
        return {
            "trainable": {
                n: p.detach().cpu()
                for n, p in head.named_parameters()
                if p.requires_grad
            },
            "opt": opt.state_dict(),
            "step": step,
            "epoch": epoch,
            "config": config,
            "metric": metric,
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        }

    def save(self, head, opt, step, epoch, config, metric=None, tag=None):
        st = self._state(head, opt, step, epoch, config, metric)
        path = os.path.join(self.dir, f"ckpt_{tag if tag else step}.pt")
        torch.save(st, path)
        self.saved.append(path)
        while (
            len(
                [
                    p
                    for p in self.saved
                    if os.path.basename(p).startswith("ckpt_") and "best" not in p
                ]
            )
            > self.keep
        ):
            old = self.saved.pop(0)
            if os.path.exists(old) and "best" not in old:
                os.remove(old)
        if metric is not None and (
            self.best_metric is None or metric > self.best_metric
        ):
            self.best_metric = metric
            torch.save(st, os.path.join(self.dir, "ckpt_best.pt"))
        return path

    def export_trainable(self, head, path, extra=None):
        """fp16 trainable-only dump for export_gguf (no optimizer/rng)."""
        ckpt = {
            n: p.detach().to(torch.float16).cpu()
            for n, p in head.named_parameters()
            if p.requires_grad
        }
        torch.save({"trainable": ckpt, **(extra or {})}, path)
        return path

    @staticmethod
    def load(path, head, opt=None):
        st = torch.load(path, map_location="cpu", weights_only=False)
        own = dict(head.named_parameters())
        for n, v in st["trainable"].items():
            if n in own:
                own[n].data.copy_(v.to(own[n].dtype).to(own[n].device))
        if opt is not None and "opt" in st:
            opt.load_state_dict(st["opt"])
        rng = st.get("rng", {})
        if rng:
            torch.set_rng_state(rng["torch"])
            if rng.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])
        return st.get("step", 0), st.get("epoch", 0)
