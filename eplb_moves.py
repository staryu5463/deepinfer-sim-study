"""EPLB 重排的代价：每次重排有多少 physical 槽位换了 expert（= 要搬的权重），以及 policy 本身的计算耗时。

搬运字节：V4.1 每个 routed expert 的 FP4 权重 = (5120*4608 + 2304*5120) * 0.5 B ≈ 17.7 MB（不含 scale）。
"""

import os
import json
import sys
import time
from dataclasses import asdict

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eplb_sim as m  # noqa: E402

EXPERT_BYTES = (5120 * 4608 + 2304 * 5120) * 0.5
cal = dict(sigma_pop=0.5, sigma_seg=1.0, sigma_drift=0.1)
out = {}
for interval in (10000, 2500):
    sc = m.Scenario(**{**asdict(m.DECODE), **cal, "layers": 4, "interval": interval, "seed": 3})
    gen = torch.Generator(device=m.DEV)
    gen.manual_seed(sc.seed)
    drift = m.Drift(sc, gen)
    prev, moved = None, []
    for cyc in range(4):
        hist = []
        for s0 in range(0, sc.window, sc.batch_steps):
            idx = m.sample_routing(sc, drift.log_pop(), sc.batch_steps, gen)
            hist.append(m.logical_counts(sc, idx))
            drift.advance(sc.batch_steps)
        pl = m.eplb_placement(sc, torch.cat(hist))
        if prev is not None:
            moved.append(float((pl.phy2log != prev.phy2log).float().mean()))
        prev = pl
        drift.advance(sc.interval - sc.window)
    frac = sum(moved) / len(moved)
    out[interval] = {"moved_slot_fraction": round(frac, 3)}
    print(interval, out[interval], flush=True)

# 全规模 policy 计算耗时：40 层 × 1000 步窗口 × 384 expert（本机 RTX 4070，B200 上会更快）
hist = torch.randint(0, 2000, (1000, 40, 384), device=m.DEV).float()
torch.cuda.synchronize()
t = time.time()
m.policy.rebalance_experts_global(hist, 448, 7)
torch.cuda.synchronize()
policy_s = time.time() - t
moved_slots = out[10000]["moved_slot_fraction"] * 448 * 40
per_gpu_bytes = moved_slots * EXPERT_BYTES / 64
out["full_scale"] = {
    "policy_compute_s_on_4070": round(policy_s, 2),
    "moved_experts_total_40_layers": round(moved_slots),
    "bytes_per_gpu_GB": round(per_gpu_bytes / 1e9, 2),
    "transfer_ms_at_400GBps": round(per_gpu_bytes / 400e9 * 1e3, 1),
}
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "eplb_moves.json"), "w"), indent=1)
