"""FastMTP fine-tune of the DeepSeek-V4-Flash MTP head (Lever A).

Warm-start the original head, FREEZE the 256 routed experts, train the
conditioning path to do recursive multi-step drafting. K-step recursive unroll
with exponential-decay weighted CE (beta=0.6), AdamW + cosine, grad checkpointing,
gradient accumulation. Tooling (trainlib): offline wandb (per-step-k loss + accept
proxy + GPU temp), resumable checkpoints, thermal cooldown, nan guard, seeding.

THE TWO LOAD-BEARING INDEX SHIFTS (in unroll_steps; mirror FastMTP 2.1 + ds4):
  base position i (skip i=0=BOS); draft step k=1..K:
    input  = project(prev[i], tok=t_{i+k}) ;  target = t_{i+k+1}
    rotary position = i+k  ;  valid 1<=i<=N-2-k ;  prev_{k+1}[i] = out_k[i]
  k=1 <-> ds4 drafts[0] (~79-88%); k=2 <-> drafts[1] (~22-60%, the target).

Inputs: harvested .npz shards {tokens int32[N], hc float32[N,hc_dim]}.
Output: resumable + best checkpoints, fp16 trainable-only export for export_gguf.
DOES NOT modify ds4. GB10 GPU. SMOKE FIRST: --max-docs 2 --K 2 --max-seq 64.
"""

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/trevor/Projects/ds4/tools/mtp")
import mtp_model as MM  # noqa: E402
import trainlib as TL  # noqa: E402

try:
    from tqdm import tqdm
except Exception:

    def tqdm(x, **k):
        return x


def alpha_weights(K, beta):
    w = [beta**k for k in range(K)]
    s = sum(w)
    return [x / s for x in w]


def list_shards(shards_dir):
    man_path = os.path.join(shards_dir, "manifest.json")
    classes = {}
    if os.path.exists(man_path):
        man = json.load(open(man_path))
        for e in man.get("shards", []):
            classes[os.path.join(shards_dir, e["shard"])] = e.get("class", "all")
    shards = sorted(glob.glob(os.path.join(shards_dir, "shard_*.npz")))
    return shards, classes


def load_shard(path, max_seq, device, dtype, cfg):
    d = np.load(path)
    toks = torch.from_numpy(d["tokens"].astype(np.int64))[:max_seq].to(device)
    N = toks.shape[0]
    hc = (
        torch.from_numpy(d["hc"][:N])
        .to(device, dtype)
        .reshape(N, cfg.hc_mult, cfg.hidden_size)
    )
    return toks, hc


def unroll_steps(head, tokens, hc, K, device, dtype):
    """Yield (k, logits[L,vocab], targets[L]) per draft step. Shared by train loss
    and accept eval; the recursion (prev_{k+1}[i]=out_k[i]) and the index shifts
    live here. THE correctness-critical block."""
    N = tokens.shape[0]
    prev = hc  # prev[i] = base HC h_i  (i=0 BOS, never used since i starts at 1)
    for k in range(1, K + 1):
        i_hi = N - 2 - k  # i in [1, i_hi]; need t_{i+k}, t_{i+k+1}
        if i_hi < 1:
            break
        idx = torch.arange(1, i_hi + 1, device=device)
        in_tok = tokens[idx + k]  # t_{i+k}
        tgt = tokens[idx + k + 1]  # t_{i+k+1}
        prev_in = prev.index_select(0, idx)  # [L,hc,D]
        pos = (idx + k).unsqueeze(0)  # [1,L] rotary absolute position
        L = idx.shape[0]
        input_hc = head.project(in_tok.unsqueeze(0), prev_in.unsqueeze(0))
        pe, mask = head.rope_mask(L, pos, device, dtype)
        out = head.decode(input_hc, pos, mask, pe)  # [1,L,hc,D]
        logits = head.to_logits(out)[0]  # [L,vocab]
        yield k, logits, tgt
        prev = prev.index_copy(0, idx, out[0])  # recursion for next step


def doc_loss(head, tokens, hc, K, alpha, device, dtype):
    total, perk = None, {}
    for k, logits, tgt in unroll_steps(head, tokens, hc, K, device, dtype):
        lk = F.cross_entropy(logits.float(), tgt)
        total = alpha[k - 1] * lk if total is None else total + alpha[k - 1] * lk
        perk[f"loss_k{k}"] = float(lk.detach())
    return total, perk


@torch.no_grad()
def accept_proxy(head, shards, classes, K, max_seq, device, dtype, cfg):
    """Held-out per-step-k top-1 agreement (proxy for MTP draft acceptance), +
    per-class breakdown. This is the REAL target metric (CE down != accept up)."""
    head.eval()
    acc = {k: [0, 0] for k in range(1, K + 1)}
    per_cls = {}
    for sp in shards:
        toks, hc = load_shard(sp, max_seq, device, dtype, cfg)
        if toks.shape[0] < K + 3:
            continue
        cls = classes.get(sp, "all")
        pc = per_cls.setdefault(cls, {k: [0, 0] for k in range(1, K + 1)})
        for k, logits, tgt in unroll_steps(head, toks, hc, K, device, dtype):
            corr = int((logits.argmax(-1) == tgt).sum())
            tot = int(tgt.shape[0])
            acc[k][0] += corr
            acc[k][1] += tot
            pc[k][0] += corr
            pc[k][1] += tot
    head.train()
    out = {f"accept_k{k}": (acc[k][0] / acc[k][1] if acc[k][1] else 0.0) for k in acc}
    for cls, pc in per_cls.items():
        for k in pc:
            if pc[k][1]:
                out[f"accept/{cls}_k{k}"] = pc[k][0] / pc[k][1]
    # headline best-metric: mean accept over the CHAIN we're fixing (k>=2)
    chain = [out[f"accept_k{k}"] for k in range(2, K + 1) if f"accept_k{k}" in out]
    out["accept_chain"] = (
        sum(chain) / len(chain) if chain else out.get("accept_k1", 0.0)
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--ckpt-dir", default="tools/mtp/ckpt")
    ap.add_argument("--export", default="tools/mtp/mtp_finetuned.pt")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--beta", type=float, default=0.6)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-seq", type=int, default=256)
    ap.add_argument("--max-docs", type=int, default=0)
    ap.add_argument("--eval-frac", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-temp", type=float, default=84.0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--project", default="ds4-fastmtp")
    ap.add_argument("--name", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    TL.set_seed(args.seed)
    device, dtype = args.device, getattr(torch, args.dtype)
    alpha = alpha_weights(args.K, args.beta)
    run_name = args.name or f"K{args.K}_lr{args.lr:.0e}_{int(time.time())}"
    cfg = vars(args) | {"alpha": alpha, "run_name": run_name}
    print(
        f"K={args.K} beta={args.beta} alpha={[round(a, 3) for a in alpha]} lr={args.lr} accum={args.grad_accum}"
    )

    head = MM.DeepseekV4MtpHead.from_pt(dtype=dtype).to(device)
    head.freeze_for_finetune()
    head.decoder.gradient_checkpointing = True
    head.train()
    n_train = sum(p.numel() for p in head.trainable_parameters())
    print(f"trainable params: {n_train / 1e6:.1f}M (experts frozen)")

    shards, classes = list_shards(args.shards)
    if args.max_docs:
        shards = shards[: args.max_docs]
    n_eval = max(1, int(len(shards) * args.eval_frac)) if len(shards) > 4 else 0
    eval_shards = shards[len(shards) - n_eval :] if n_eval else []
    train_shards = shards[: len(shards) - n_eval] if n_eval else shards
    print(f"shards: {len(train_shards)} train / {len(eval_shards)} eval")

    opt = torch.optim.AdamW(head.trainable_parameters(), lr=args.lr, betas=(0.9, 0.95))
    total_steps = max(1, (len(train_shards) // args.grad_accum) * args.epochs)
    warmup_steps = max(1, int(total_steps * args.warmup))

    def lr_at(step):
        if step < warmup_steps:
            return args.lr * step / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))

    ckpt = TL.Checkpointer(args.ckpt_dir)
    guard = TL.ThermalGuard(max_c=args.max_temp)
    start_step, start_epoch = 0, 0
    if args.resume:
        start_step, start_epoch = TL.Checkpointer.load(args.resume, head, opt)
        print(f"resumed from {args.resume} @ step {start_step}, epoch {start_epoch}")

    run, wandb = TL.init_wandb(args.project, run_name, cfg)
    step = start_step
    t_last = time.time()
    for epoch in range(start_epoch, args.epochs):
        opt.zero_grad(set_to_none=True)
        micro = 0
        for sp in tqdm(train_shards, desc=f"epoch {epoch}"):
            toks, hc = load_shard(sp, args.max_seq, device, dtype, head.cfg)
            if toks.shape[0] < args.K + 3:
                continue
            loss, perk = doc_loss(head, toks, hc, args.K, alpha, device, dtype)
            if loss is None or not TL.is_finite(loss):
                print("  skip doc (non-finite/empty loss)")
                continue
            (loss / args.grad_accum).backward()
            micro += 1
            if micro < args.grad_accum:
                continue
            guard.maybe_cooldown()
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            gnorm = float(
                torch.nn.utils.clip_grad_norm_(head.trainable_parameters(), 1.0)
            )
            opt.step()
            opt.zero_grad(set_to_none=True)
            micro = 0
            step += 1

            n_pos = int(toks.shape[0])
            tps = (n_pos * args.grad_accum) / max(1e-6, time.time() - t_last)
            t_last = time.time()
            log = {
                "step": step,
                "epoch": epoch,
                "lr": lr_at(step),
                "grad_norm": gnorm,
                "loss": float(loss.detach()),
                "tok_s": tps,
                **perk,
                **TL.gpu_stats(),
            }
            if step % args.eval_every == 0 and eval_shards:
                log.update(
                    accept_proxy(
                        head,
                        eval_shards,
                        classes,
                        args.K,
                        args.max_seq,
                        device,
                        dtype,
                        head.cfg,
                    )
                )
                print(
                    f"  [eval] step {step} accept_chain={log.get('accept_chain', 0):.3f} "
                    + " ".join(
                        f"k{k}={log.get(f'accept_k{k}', 0):.2f}"
                        for k in range(1, args.K + 1)
                    )
                )
            wandb.log(log)
            if step % args.ckpt_every == 0:
                ckpt.save(head, opt, step, epoch, cfg, metric=log.get("accept_chain"))

    final = (
        accept_proxy(
            head, eval_shards, classes, args.K, args.max_seq, device, dtype, head.cfg
        )
        if eval_shards
        else {}
    )
    ckpt.save(
        head, opt, step, args.epochs, cfg, metric=final.get("accept_chain"), tag="final"
    )
    ckpt.export_trainable(head, args.export, extra={"K": args.K, "beta": args.beta})
    print(f"done. final {final}\nexport -> {args.export}, ckpts -> {args.ckpt_dir}")
    run.finish()


if __name__ == "__main__":
    sys.exit(main())
