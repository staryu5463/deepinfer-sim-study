"""多 DP rank 耦合下的 DSpark 验证器（EP64 decode：每层 MoE 要等所有 rank 的 attention 做完）。

真值（每步）：
    A_r   = f_loc * (a*G_r + b*G_r^2) + imb_loc*G_r + ctx_r + phi*c_seq*S_r      本 rank 的 attention 侧时间（局部）
    M     = (1-f_loc) * (a*Gbar + b*Gbar^2) + imb_moe*Gbar                      MoE 时间（全局，按各 rank 平均 token 数）
    fwd   = T0 + max_r A_r + M                                                  所有 rank 测到的 forward_ms 都是这个全局值
    smp   = max_r (1-phi)*c_seq*S_r                                             draft 模型也有 EP MoE，同样同步
对称时（所有 rank 的 G、S、ctx 相同）退化为单 rank 校准模型：T0 + B(G) - T0 + ...，校准参数直接沿用。

策略：
    fixed5   每个 rank 固定验证 5 个
    dspark   每个 rank 各自跑 MTPVarlen::schedule；估时器喂全局 forward_ms 和自己的 graph size（与源码一致）
    coord    协调版（上界）：中心化地选一个公共 attention 水位 H，各 rank 按 confidence 从高到低填 token 直到 A_r 达到 H，
             再在所有 H 里取全局吞吐最高的；与 dspark 用相同的 confidence 信息
    coord_lite 实用版：每步只交换两个标量（上一步的全局吞吐 λ、各 rank 预测 attention 时间的最大值 H）。
             rank 内 token 的边际代价：不超过 H 时只算 MoE 份额，超过 H 时算 attention+MoE；按 conf >= λ*代价 取 token。
"""

from __future__ import annotations

import os

import json
import sys
from dataclasses import dataclass, replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calibrate_dspark as cal  # noqa: E402
import dspark_sim as d  # noqa: E402

NEXTN = d.NEXTN


@dataclass
class MR:
    ranks: int = 64
    mean_seqs: int = 60
    sd_seqs: float = 0.0  # 各 rank 请求数的离散度（decode LB 不完美）
    ctx: str = "bench32k"
    f_loc: float = 0.45  # 逐 token 开销里 attention 侧（局部）占比
    f_loc_belief: float | None = None  # 协调策略内部模型以为的占比（None=知道真值）；用来测模型误差下的鲁棒性
    steps: int = 1500
    warmup: int = 300
    seed: int = 0


def make_worlds(mr: MR, acc: d.Accept):
    rng = np.random.default_rng(mr.seed)
    S = np.maximum(8, np.round(mr.mean_seqs + mr.sd_seqs * rng.standard_normal(mr.ranks))).astype(int)
    return [d.World(d.Workload(conc=int(S[r]), ctx=mr.ctx, steps=mr.steps, warmup=mr.warmup, seed=mr.seed * 1000 + r), acc)
            for r in range(mr.ranks)]


def local_time(hw: d.HW, mr: MR, G, S, ctx_sum):
    G = np.asarray(G, dtype=float)
    return mr.f_loc * (hw.a * G + hw.b * G * G) + mr.f_loc * hw.imb * G + ctx_sum + hw.phi * hw.c_seq * S


def moe_time(hw: d.HW, mr: MR, Gbar):
    return (1 - mr.f_loc) * (hw.a * Gbar + hw.b * Gbar * Gbar) + (1 - mr.f_loc) * hw.imb * Gbar


def _views(worlds, t, prev_conf):
    out = []
    for r, w in enumerate(worlds):
        order = np.argsort(-w.L[t], kind="stable")
        out.append((order, d.StepView(w.S, w.L[t][order], prev_conf[r][order], w.conf[t][order], w.fresh[t][order])))
    return out


def _prefix(v: d.StepView, hw, mr):
    """本 rank 按当前 confidence 的 cum 降序加 token 的前缀表：第 k 项 = 加了 k 个 draft 后的 (增益累计, 本地时间, 下一 token 增益, 下一 token 本地边际)。"""
    cur = np.where(v.fresh[:, None], 0.0, v.conf_cur)
    cum = np.cumprod(cur, axis=1)
    order = d._sorted_queue(cum)
    gains = cum.reshape(-1)[order]
    seq_of = order // NEXTN
    w = d.pairs(v.L)
    p_tok = (6.0 / hw.bw + 8192.0 / hw.fp4) * 1e3
    base_ctx = float((68.0 * w / hw.bw * 1e3).sum() + (w * p_tok).sum())
    k = np.arange(len(order) + 1)
    G = v.S + k
    ctx = base_ctx + np.concatenate([[0.0], np.cumsum(w[seq_of] * p_tok)])
    A = local_time(hw, mr, G, v.S, ctx)
    E = v.S + np.concatenate([[0.0], np.cumsum(gains)])
    ok = G <= hw.graphs[-1]
    return dict(order=order, seq_of=seq_of, E=E[ok], A=A[ok], G=G[ok], gains=gains)


def _q_from_k(v: d.StepView, pre, k: int) -> np.ndarray:
    q = np.ones(v.S, dtype=int)
    np.add.at(q, pre["seq_of"][:k], 1)
    return q


def _k_at_level(pre, H: float) -> int:
    return max(0, int(np.searchsorted(pre["A"], H, side="right")) - 1)


def simulate(worlds, hw: d.HW, mr: MR, policy: str) -> dict:
    R = len(worlds)
    rng = np.random.default_rng(777)
    baseline = np.zeros(hw.graphs[-1] + 1)
    baseline[hw.graphs] = hw.B(hw.graphs) * (1 + 0.003 * rng.standard_normal(len(hw.graphs)))
    ctxs = [{"fwd": d.ForwardRuntimeEstimator(), "smp": d.SamplerEstimator(), "ca": d.CtxAwareEstimator(), "baseline": baseline}
            for _ in range(R)]
    prev_conf = [w.conf[0] for w in worlds]
    lam_prev, H_prev = None, None
    H_lite = None
    tot_tok = tot_time = 0.0
    Gstd, n = [], 0
    for t in range(mr.steps):
        views = _views(worlds, t, prev_conf)
        qs = []
        if policy == "fixed5":
            qs = [d.schedule_fixed(v, hw, None) for _, v in views]
        elif policy == "dspark":
            qs = [d.schedule_dspark(v, hw, ctxs[r]) for r, (_, v) in enumerate(views)]
        elif policy == "coord":
            pres = [_prefix(v, hw, mr) for _, v in views]
            lo = min(p["A"][0] for p in pres)
            hi = max(p["A"][-1] for p in pres)
            best = None
            for H in np.linspace(lo, hi, 60):
                ks = [_k_at_level(p, H) for p in pres]
                E = sum(p["E"][k] for p, k in zip(pres, ks))
                Amax = max(p["A"][k] for p, k in zip(pres, ks))
                Gbar = float(np.mean([p["G"][k] for p, k in zip(pres, ks)]))
                smp_est = max((1 - hw.phi) * hw.c_seq * v.S for _, v in views)
                T = hw.T0 + Amax + moe_time(hw, mr, Gbar) + smp_est
                if best is None or E / T > best[0]:
                    best = (E / T, ks)
            qs = [_q_from_k(v, p, k) for (_, v), p, k in zip(views, pres, best[1])]
        elif policy == "coord_lite":
            mb = replace(mr, f_loc=mr.f_loc_belief) if mr.f_loc_belief is not None else mr
            pres = [_prefix(v, hw, mb) for _, v in views]
            if H_lite is None:
                H_lite = float(np.median([p["A"][min(len(p["A"]) - 1, int(0.7 * NEXTN * v.S))] for p, (_, v) in zip(pres, views)]))
            ks = [_k_at_level(p, H_lite) for p in pres]
            qs = [_q_from_k(v, p, k) for (_, v), p, k in zip(views, pres, ks)]
            # 每个 rank 上报两个标量：在水位 H 处下一个 token 的增益 / 本地边际时间，以及 1 / 本地边际时间（all-reduce 求和）
            s_sum = z_sum = 0.0
            for p, k in zip(pres, ks):
                if k + 1 < len(p["A"]):
                    da = max(p["A"][k + 1] - p["A"][k], 1e-6)
                    s_sum += p["gains"][k] / da
                    z_sum += 1.0 / da
            Gbar = float(np.mean([p["G"][k] for p, k in zip(pres, ks)]))
            dM = (moe_time(hw, mb, Gbar + 1) - moe_time(hw, mb, Gbar))
            lam = lam_prev if lam_prev is not None else 0.0
            grad = s_sum - lam * (1.0 + dM / R * z_sum)  # Dinkelbach: dE/dH - λ dT/dH
            scale = max(abs(s_sum), 1e-9)
            H_lite *= float(np.exp(0.05 * np.tanh(grad / scale * 3)))
        else:
            raise ValueError(policy)

        T, fwd, smp, A = _true_time(hw, mr, views, qs, noise=1.0, t=t, worlds=worlds, return_parts=True)
        tokens = 0.0
        for r, ((order, v), q) in enumerate(zip(views, qs)):
            w = worlds[r]
            acc_mask = w.U[t][order] < w.pi[t][order]
            lead = np.cumprod(acc_mask, axis=1)
            tokens += float((1 + (lead * (np.arange(NEXTN) < (q - 1)[:, None])).sum(1)).sum())
            g = hw.graph_of(int(q.sum()))
            ctxs[r]["fwd"].update(g, (fwd - baseline[g]) / g)
            ctxs[r]["smp"].update(v.S, smp)
            prev_conf[r] = w.conf[t]
        lam_prev = tokens / T  # 全局吞吐（tokens/ms），一次 all-reduce 即可得到
        H_prev = float(np.max(A))
        if t >= mr.warmup:
            tot_tok += tokens
            tot_time += T
            Gstd.append(float(np.std([q.sum() for q in qs]) / np.mean([q.sum() for q in qs])))
            n += 1
    return {"throughput_per_rank": tot_tok / tot_time * 1e3 / R, "G_cv_across_ranks": float(np.mean(Gstd))}


def _true_time(hw, mr, views, qs, noise=1.0, t=0, worlds=None, return_parts=False):
    A = []
    Gs = []
    for (_, v), q in zip(views, qs):
        G = int(q.sum())
        g = hw.graph_of(G)
        Gs.append(g)
        ctx_sum = float(hw.ctx_cost(v.L, q).sum())
        A.append(local_time(hw, mr, g, v.S, ctx_sum))
    A = np.array(A)
    Gbar = float(np.mean(Gs))
    nz = worlds[0].fwd_noise[t] if (worlds is not None and noise) else 0.0
    fwd = (hw.T0 + float(A.max()) + moe_time(hw, mr, Gbar)) * (1 + hw.noise * nz)
    smp = float(max((1 - hw.phi) * hw.c_seq * v.S for _, v in views))
    T = fwd + smp
    if return_parts:
        return T, fwd, smp, A
    return T


def run(mr: MR, policies=("fixed5", "dspark", "coord_lite", "coord")):
    x = [json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "dspark_calibration.json")))["params"][k] for k in cal.NAMES]
    hw, acc = cal.build(x)
    worlds = make_worlds(mr, acc)
    res = {p: simulate(worlds, hw, mr, p) for p in policies}
    base = res["dspark"]["throughput_per_rank"]
    for p in res:
        res[p]["gain_vs_dspark_%"] = round(100 * (res[p]["throughput_per_rank"] / base - 1), 2)
    return res


if __name__ == "__main__":
    mr = MR(ranks=int(sys.argv[1]) if len(sys.argv) > 1 else 16, steps=600, warmup=150)
    print(json.dumps(run(mr, ("fixed5", "dspark", "coord_lite")), indent=1))
