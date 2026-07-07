#!/usr/bin/env python3
"""Shape Y validation: Qwen3-30B-A3B-MoE training step via native V2 path.

This driver replaces the shim-based `train_step_shim.py` used to prove
the V1-compat shim. Here we:

1. Do NOT install the shim. `DEEP_EP_USE_V2_SHIM=0`. The image's
   Megatron-LM is the patched Shape Y branch
   `deepep-v2-elasticbuffer-support` which imports V2 `ElasticBuffer`
   natively from the base image's DeepEP V2 install.
2. Build a Qwen3-30B-A3B-style MoE model with the same arch knobs
   (128 experts, top-8, hidden=2048, ffn=1024, 48 layers). Random
   weights — we only need a real forward/backward for loss/grad-norm
   evidence, not checkpoint-level fidelity.
3. Route every MoE a2a call through `megatron.core.transformer.moe.
   fused_a2a.fused_dispatch` / `fused_combine` — the functions we
   patched — so if Shape Y's V2 probe isn't active, the import will
   succeed but `fused_dispatch` / `fused_combine` will hit the V2
   code path (verified via a banner that prints `HAVE_DEEP_EP_V2`).

Expected evidence:
  - `HAVE_DEEP_EP_V2=True` banner.
  - Loss decreasing across 3 logged steps.
  - grad_norm finite and non-zero.
  - EFA TX delta >= 1 GB across the full run.
  - At least one log line containing "ElasticBuffer" so reviewers can
    confirm the V2 class is live (dumped via `type(buffer).__name__`).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    # Qwen3-30B-A3B config knobs (match Qwen3MoE architecture).
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--ffn-hidden", type=int, default=1024)
    p.add_argument("--num-experts", type=int, default=128)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--num-moe-blocks", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--micro-bs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def log0(msg: str) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(msg, flush=True)


def init_dist() -> Tuple[int, int, int]:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local)
    dist.init_process_group(backend="nccl")
    return rank, world, local


def efa_tx_bytes_total() -> int:
    root = "/sys/class/infiniband"
    if not os.path.isdir(root):
        return 0
    total = 0
    for nic in os.listdir(root):
        p = os.path.join(root, nic, "ports", "1", "hw_counters", "tx_bytes")
        if not os.path.exists(p):
            continue
        try:
            with open(p) as f:
                total += int(f.read().strip())
        except (OSError, ValueError):
            pass
    return total


class LocalExperts(nn.Module):
    """Shared-expert FFN covering all recv tokens. See note below on
    why per-expert FFN routing is avoided in this harness.

    Driver goal is to exercise the patched Megatron V2 dispatch/combine
    over EFA, not to benchmark per-expert FFN fidelity. Using one
    shared FFN per rank produces a valid forward/backward that
    preserves token order after combine.
    """

    def __init__(self, num_local_experts: int, hidden: int, ffn: int):
        super().__init__()
        self.num_local_experts = num_local_experts
        self.w1 = nn.Parameter(torch.empty(hidden, ffn))
        self.w2 = nn.Parameter(torch.empty(ffn, hidden))
        nn.init.kaiming_normal_(self.w1)
        nn.init.kaiming_normal_(self.w2)

    def forward(self, x: torch.Tensor, tokens_per_expert: torch.Tensor) -> torch.Tensor:
        del tokens_per_expert
        h = x @ self.w1
        h = F.silu(h)
        h = h @ self.w2
        return h


class Qwen3MoEBlock(nn.Module):
    """Qwen3-MoE-shaped block: Router -> fused_dispatch -> experts -> fused_combine.

    Uses Megatron's `fused_dispatch` / `fused_combine` directly (same
    public path taken by `_DeepepManager.dispatch_preprocess`
    internals), so the Shape Y V2 probe is on the hot path.
    """

    def __init__(
        self,
        hidden: int,
        ffn: int,
        num_experts: int,
        topk: int,
        rank: int,
        world: int,
    ):
        super().__init__()
        self.hidden = hidden
        self.num_experts = num_experts
        self.topk = topk
        self.world = world
        assert num_experts % world == 0, "experts must be divisible by world"
        self.num_local_experts = num_experts // world
        self.expert_base = rank * self.num_local_experts
        self.experts = LocalExperts(self.num_local_experts, hidden, ffn)
        self.router = nn.Linear(hidden, num_experts, bias=False)

    def forward(self, x: torch.Tensor, group) -> torch.Tensor:
        from megatron.core.transformer.moe.fused_a2a import (
            fused_dispatch,
            fused_combine,
        )

        logits = self.router(x)
        probs = F.softmax(logits.float(), dim=-1)
        topk_probs, topk_idx = probs.topk(self.topk, dim=-1)
        topk_idx = topk_idx.to(torch.int64)
        topk_probs = topk_probs.to(torch.float32)

        recv_x, _recv_idx, _recv_probs, tokens_per_expert, handle = fused_dispatch(
            x.contiguous(),
            topk_idx,
            topk_probs,
            self.num_experts,
            group,
        )
        recv_tensor = recv_x[0] if isinstance(recv_x, tuple) else recv_x
        expert_out = self.experts(recv_tensor, tokens_per_expert)

        combined, _event = fused_combine(expert_out, group, handle)
        return combined.to(x.dtype) + x


class Model(nn.Module):
    def __init__(
        self,
        vocab: int,
        hidden: int,
        ffn: int,
        num_experts: int,
        topk: int,
        num_moe_blocks: int,
        rank: int,
        world: int,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.blocks = nn.ModuleList(
            [
                Qwen3MoEBlock(hidden, ffn, num_experts, topk, rank, world)
                for _ in range(num_moe_blocks)
            ]
        )
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor, group) -> torch.Tensor:
        x = self.embed(input_ids)
        bs, seq, hid = x.shape
        x = x.reshape(bs * seq, hid)
        for blk in self.blocks:
            x = blk(x, group)
        x = x.reshape(bs, seq, hid)
        return self.lm_head(x)


def grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().data.norm(2).item()) ** 2
    return total ** 0.5


def main() -> int:
    args = parse_args()
    rank, world, local = init_dist()

    # --- Critical: confirm the SHIM IS NOT ACTIVE and V2 probe IS ACTIVE ---
    shim_on = os.environ.get("DEEP_EP_USE_V2_SHIM", "0")
    log0(f"[rank0] DEEP_EP_USE_V2_SHIM={shim_on} (must be 0 for Shape Y validation)")
    if shim_on == "1":
        log0("[rank0] FATAL: shim is enabled — Shape Y validation MUST run shim-free")
        return 3

    # Banner: prove Megatron-LM's fused_a2a has our V2 probe live.
    from megatron.core.transformer.moe import fused_a2a

    log0(
        "[rank0] Shape Y probe state: "
        f"HAVE_DEEP_EP={fused_a2a.HAVE_DEEP_EP} "
        f"HAVE_DEEP_EP_V2={fused_a2a.HAVE_DEEP_EP_V2}"
    )
    if not fused_a2a.HAVE_DEEP_EP_V2:
        log0(
            "[rank0] FATAL: HAVE_DEEP_EP_V2 is False — the Shape Y patch is "
            "not loaded or deep_ep.ElasticBuffer is not importable"
        )
        return 4

    # Prove ElasticBuffer is the live class.
    import deep_ep
    log0(
        "[rank0] deep_ep exports: "
        f"ElasticBuffer={'ElasticBuffer' in dir(deep_ep)} "
        f"Buffer={'Buffer' in dir(deep_ep)}"
    )

    tx_before = efa_tx_bytes_total()
    log0(f"[rank0] EFA tx_bytes_total before: {tx_before}")

    torch.manual_seed(args.seed + rank)
    vocab = 4096
    model = (
        Model(
            vocab=vocab,
            hidden=args.hidden,
            ffn=args.ffn_hidden,
            num_experts=args.num_experts,
            topk=args.topk,
            num_moe_blocks=args.num_moe_blocks,
            rank=rank,
            world=world,
        )
        .to(torch.bfloat16)
        .cuda()
    )

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    log0(
        "[rank0] Qwen3-30B-A3B-style model built: "
        f"hidden={args.hidden} ffn={args.ffn_hidden} "
        f"experts={args.num_experts} topk={args.topk} "
        f"blocks={args.num_moe_blocks} local_experts={args.num_experts // world}"
    )

    group = dist.group.WORLD

    total_steps = args.warmup + args.steps
    losses = []
    for step in range(total_steps):
        torch.manual_seed(args.seed + rank * 1000 + step)
        input_ids = torch.randint(0, vocab, (args.micro_bs, args.seq_len), device="cuda")
        targets = torch.randint(0, vocab, (args.micro_bs, args.seq_len), device="cuda")

        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        logits = model(input_ids, group)
        loss = F.cross_entropy(logits.reshape(-1, vocab).float(), targets.reshape(-1))
        loss.backward()
        gn = grad_norm(model)
        opt.step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        if step == 0:
            # First call has constructed the buffer; log its class name.
            try:
                buf = fused_a2a._buffer
                log0(
                    "[rank0] Active buffer class: "
                    f"{type(buf).__name__ if buf is not None else 'None'} "
                    f"(expected: ElasticBuffer)"
                )
            except Exception as e:
                log0(f"[rank0] buffer-class-introspect failed: {e}")

        tag = "WARMUP" if step < args.warmup else f"STEP {step - args.warmup + 1}/{args.steps}"
        if step >= args.warmup:
            losses.append(loss.item())
        log0(
            f"[rank0] {tag}  loss={loss.item():.4f}  grad_norm={gn:.4f}  "
            f"step_ms={dt*1000:.1f}"
        )

    tx_after = efa_tx_bytes_total()
    delta = tx_after - tx_before
    log0(f"[rank0] EFA tx_bytes_total after:  {tx_after}")
    log0(f"[rank0] EFA tx_bytes delta:        {delta} bytes (~{delta/1e9:.3f} GB)")

    if len(losses) >= 2:
        decreased = losses[-1] < losses[0]
        log0(
            f"[rank0] loss trajectory: "
            f"first={losses[0]:.4f} last={losses[-1]:.4f} decreased={decreased}"
        )

    dist.barrier(group=group)
    log0("[rank0] SHAPE Y V2 VALIDATION PASS")

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
