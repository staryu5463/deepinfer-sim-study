"""EPLB 验证器：用 DeepInfer 真实的 policy.py，在 token 级路由模拟下复现 moe_lb，并评估副本分流策略。

指标与 deepinfer/model_executor/eplb/__init__.py 完全一致：
    moe_lb = sum_{step,layer} max_rank(tokens) / sum_{step,layer} mean_rank(tokens)

路由模型（每层独立）：
    logit[token, e] = log_pop[e](t) + pref[segment(token), e] + Gumbel
    每个 token 取 top-K 个不同 expert。
    log_pop = sigma_pop * z + drift(t)，drift 是 AR(1)（时间尺度 tau 步）。
    pref 是 segment 级偏好（同一请求的连续 token 路由相关），sigma_seg 控制相关强度。

副本分流策略（同一份路由样本上对比，common random numbers）：
    none        不做 EPLB（无冗余，初始放置）
    hash        现状：(src_rank + token_idx * 23333) % n_replica   ← DeepGEMM sm100_mega_gate.cuh:205
    rr          零通信改进：同一 src rank 上第 k 个选中 expert e 的 token → 副本 (src_rank + k) % n_replica
    fresh       放置用"当下真实热度"重算（去掉统计滞后），分流仍用 hash
    dyn_greedy  每步全局负载感知分流，单遍贪心水位填充（反例：短视，反而比 hash 差）
    dyn_cd      每步全局负载感知分流，从均分出发 3 轮坐标下降（可实现的近似）
    dyn_lp      每步全局负载感知分流，线性规划精确解（动态分流的理论上界）
    动态分流都需要每层多一次 per-expert 计数交换
"""

from __future__ import annotations

import os

import argparse
import importlib.util
import json
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

# DeepInfer 源码位置：环境变量 DEEPINFER_SRC，默认与本仓库同级的 ../DeepInfer
DEEPINFER_SRC = os.environ.get("DEEPINFER_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DeepInfer"))
POLICY_PATH = os.path.join(DEEPINFER_SRC, "deepinfer", "model_executor", "eplb", "policy.py")
if not os.path.exists(POLICY_PATH):
    raise FileNotFoundError(f"找不到 {POLICY_PATH}；请 git clone https://github.com/deepseek-ai/DeepInfer 并设置 DEEPINFER_SRC")
_spec = importlib.util.spec_from_file_location("deepinfer_eplb_policy", POLICY_PATH)
policy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(policy)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POLICIES = ("none", "hash", "rr", "fresh", "dyn_greedy", "dyn_cd", "dyn_lp")


@dataclass
class Scenario:
    name: str
    ranks: int
    experts: int = 384
    topk: int = 6
    min_extra_per_rank: int = 1  # 文档：decode 每卡 1 个冗余，prefill 每卡 4 个
    layers: int = 2
    tokens_per_rank: int = 276  # decode: 60 seqs * ~4.6 query tokens
    seg_len_choices: tuple[int, ...] = (4, 5, 5, 6)  # decode: 每个请求本步的 query token 数
    sigma_pop: float = 0.5
    sigma_seg: float = 1.0
    sigma_drift: float = 0.2
    tau_drift: float = 20000.0
    window: int = 1000  # DEEPINFER_EXPERT_COUNT_WINDOW_SIZE
    interval: int = 10000  # DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL
    measure_steps: int = 300  # 每个周期在 interval 内均匀抽样测量的步数
    cycles: int = 3
    batch_steps: int = 8
    seed: int = 0

    @property
    def num_extra(self) -> int:
        # 同 eplb.resolve_num_extra_experts
        m = self.min_extra_per_rank * self.ranks
        return m + (-(self.experts + m) % self.ranks)

    @property
    def num_physical(self) -> int:
        return self.experts + self.num_extra

    @property
    def num_local(self) -> int:
        return self.num_physical // self.ranks


DECODE = Scenario(name="decode_ep64", ranks=64)
PREFILL = Scenario(
    name="prefill_ep4",
    ranks=4,
    min_extra_per_rank=4,
    tokens_per_rank=16384,
    seg_len_choices=(512,),
    window=100,
    interval=1000,
    measure_steps=100,
    cycles=4,
    batch_steps=2,
)


class Drift:
    def __init__(self, sc: Scenario, gen: torch.Generator) -> None:
        self.sc, self.gen = sc, gen
        self.base = sc.sigma_pop * torch.randn(sc.layers, sc.experts, generator=gen, device=DEV)
        self.d = sc.sigma_drift * torch.randn(sc.layers, sc.experts, generator=gen, device=DEV)

    def advance(self, k: int) -> None:
        if k <= 0:
            return
        a = math.exp(-k / self.sc.tau_drift)
        eps = torch.randn(self.d.shape, generator=self.gen, device=DEV)
        self.d = a * self.d + math.sqrt(max(0.0, 1 - a * a)) * self.sc.sigma_drift * eps

    def log_pop(self) -> torch.Tensor:
        lp = self.base + self.d
        return lp - torch.logsumexp(lp, dim=-1, keepdim=True)


def segment_ids(sc: Scenario, gen: torch.Generator) -> tuple[torch.Tensor, int]:
    """单个 rank 一步内 token -> segment 编号（连续 token 属于同一 segment）。"""
    lens = []
    total = 0
    choices = sc.seg_len_choices
    while total < sc.tokens_per_rank:
        n = choices[int(torch.randint(len(choices), (1,), generator=gen, device=DEV))]
        n = min(n, sc.tokens_per_rank - total)
        lens.append(n)
        total += n
    lens_t = torch.tensor(lens, device=DEV)
    return torch.repeat_interleave(torch.arange(len(lens), device=DEV), lens_t), len(lens)


def sample_routing(sc: Scenario, log_pop: torch.Tensor, n_steps: int, gen: torch.Generator) -> torch.Tensor:
    """返回 topk_idx: [n_steps, layers, ranks, tokens_per_rank, topk]（rank 内 token 顺序 = gate kernel 的 token_idx）。"""
    seg, n_seg = segment_ids(sc, gen)  # 同一批步共享 segment 切分形状，偏好各自独立
    T, E = sc.tokens_per_rank, sc.experts
    out = torch.empty(n_steps, sc.layers, sc.ranks, T, sc.topk, dtype=torch.int32, device=DEV)
    for s in range(n_steps):
        for layer in range(sc.layers):
            pref = sc.sigma_seg * torch.randn(sc.ranks, n_seg, E, generator=gen, device=DEV)
            u = torch.rand(sc.ranks, T, E, generator=gen, device=DEV).clamp_(1e-20, 1 - 1e-7)
            logits = log_pop[layer] + pref[:, seg, :] - torch.log(-torch.log(u))
            out[s, layer] = logits.topk(sc.topk, dim=-1).indices.int()
    return out


def logical_counts(sc: Scenario, idx: torch.Tensor) -> torch.Tensor:
    """[steps, layers, ranks, T, K] -> [steps, layers, experts]"""
    S, L = idx.shape[:2]
    flat = idx.reshape(S, L, -1).long()
    cnt = torch.zeros(S, L, sc.experts, device=DEV)
    cnt.scatter_add_(-1, flat, torch.ones_like(flat, dtype=torch.float))
    return cnt


@dataclass
class Placement:
    phy2log: torch.Tensor  # [L, P]
    log2phy: torch.Tensor  # [L, E, dup]
    logcnt: torch.Tensor  # [L, E]
    local: int


def no_eplb_placement(sc: Scenario) -> Placement:
    L, E = sc.layers, sc.experts
    phy2log = torch.arange(E, device=DEV).int().expand(L, E).contiguous()
    log2phy = torch.arange(E, device=DEV).int().view(1, E, 1).expand(L, E, 1).contiguous()
    return Placement(phy2log, log2phy, torch.ones(L, E, dtype=torch.int, device=DEV), E // sc.ranks)


def eplb_placement(sc: Scenario, history: torch.Tensor) -> Placement:
    """history: [steps, layers, experts]，直接调用 DeepInfer 的 rebalance_experts_global。"""
    p2l, l2p, cnt = policy.rebalance_experts_global(history.float(), sc.num_physical, sc.num_local)
    return Placement(p2l, l2p, cnt, sc.num_local)


def dest_loads(sc: Scenario, idx: torch.Tensor, pl: Placement, mode: str) -> torch.Tensor:
    """idx: [S, L, R, T, K] -> 每个目的 rank 收到的 token-expert 数 [S, L, R]。"""
    S, L, R, T, K = idx.shape
    e = idx.long()
    layer_ix = torch.arange(L, device=DEV).view(1, L, 1, 1, 1)
    c = pl.logcnt[layer_ix, e].long()  # 副本数
    src = torch.arange(R, device=DEV).view(1, 1, R, 1, 1)
    if mode == "hash":
        tok = torch.arange(T, device=DEV).view(1, 1, 1, T, 1)
        dup = (src + tok * 23333) % c
    elif mode == "rr":
        # 同一 (step, layer, src rank, expert) 的第 k 次出现 → 副本 (src + k) % c
        key = ((torch.arange(S, device=DEV).view(S, 1, 1, 1, 1) * L + layer_ix) * R + src) * sc.experts + e
        flat = key.reshape(-1)
        order = torch.argsort(flat, stable=True)  # 稳定排序保持 token 顺序
        sorted_key = flat[order]
        pos = torch.arange(flat.numel(), device=DEV)
        start = torch.ones_like(sorted_key, dtype=torch.bool)
        start[1:] = sorted_key[1:] != sorted_key[:-1]
        seg_start = torch.cummax(torch.where(start, pos, torch.zeros_like(pos)), 0).values
        occ = torch.empty_like(flat)
        occ[order] = pos - seg_start
        dup = (src + occ.view_as(e)) % c
    else:
        raise ValueError(mode)
    phys = pl.log2phy[layer_ix, e, dup].long()
    dest = phys // pl.local
    loads = torch.zeros(S, L, R, device=DEV)
    loads.scatter_add_(-1, dest.reshape(S, L, -1), torch.ones(S, L, R * T * K, device=DEV))
    return loads


def _water_fill(lv: np.ndarray, amount: float) -> np.ndarray:
    """把 amount 填进水位 lv，返回每格加了多少（使被填的格子齐平）。"""
    order = np.argsort(lv)
    s = lv[order]
    for j in range(1, len(s) + 1):
        h = (amount + s[:j].sum()) / j
        if j == len(s) or h <= s[j]:
            break
    add = np.zeros_like(lv)
    add[order] = np.maximum(0.0, h - s)
    return add


def _replica_structure(sc: Scenario, pl: Placement, layer: int):
    p2l = pl.phy2log[layer].cpu().numpy()
    logcnt = pl.logcnt[layer].cpu().numpy()
    ranks_of = [[] for _ in range(sc.experts)]
    for p, lg in enumerate(p2l):
        ranks_of[lg].append(p // pl.local)
    single = logcnt == 1
    single_rank = np.array([ranks_of[lg][0] for lg in range(sc.experts)])
    multi = np.where(~single)[0]
    # 同一 rank 上同一 expert 的多个副本合并成一个"格子"
    multi_ranks = {int(lg): np.unique(ranks_of[lg]) for lg in multi}
    return single, single_rank, multi, multi_ranks


def dyn_loads(sc: Scenario, idx: torch.Tensor, pl: Placement, method: str = "lp", passes: int = 3) -> torch.Tensor:
    """每步全局负载感知分流。非冗余 expert 的负载固定，只重新分配冗余 expert 的 token。

    method="lp": 线性规划求 min-max 精确解（动态分流的理论上界）。
    method="cd": 从均分出发做 passes 轮坐标下降（每次只重排一个 expert，其余固定），可在 kernel 里近似实现。
    method="greedy": 单遍按热度降序水位填充（反例：短视，会比 hash 更差）。
    """
    from scipy.optimize import linprog

    S, L = idx.shape[:2]
    cnt = logical_counts(sc, idx).cpu().numpy()  # [S, L, E]
    R = sc.ranks
    out = np.zeros((S, L, R))
    for layer in range(L):
        single, single_rank, multi, multi_ranks = _replica_structure(sc, pl, layer)
        cells = [(int(lg), int(r)) for lg in multi for r in multi_ranks[int(lg)]]
        for s in range(S):
            n = cnt[s, layer]
            base = np.bincount(single_rank[single], weights=n[single], minlength=R).astype(float)
            if method == "lp":
                nv = len(cells) + 1  # 最后一个变量是 H
                c = np.zeros(nv)
                c[-1] = 1.0
                A_ub = np.zeros((R, nv))
                for k, (_, r) in enumerate(cells):
                    A_ub[r, k] = 1.0
                A_ub[:, -1] = -1.0
                b_ub = -base
                A_eq = np.zeros((len(multi), nv))
                expert_row = {int(lg): i for i, lg in enumerate(multi)}
                for k, (lg, _) in enumerate(cells):
                    A_eq[expert_row[lg], k] = 1.0
                b_eq = n[multi]
                res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * nv, method="highs")
                assert res.status == 0, res.message
                load = base.copy()
                for k, (_, r) in enumerate(cells):
                    load[r] += res.x[k]
            else:
                alloc = {}
                load = base.copy()
                for lg in multi:
                    rs = multi_ranks[int(lg)]
                    a = np.full(len(rs), n[lg] / len(rs)) if method == "cd" else np.zeros(len(rs))
                    alloc[int(lg)] = a
                    load[rs] += a
                order = multi[np.argsort(-n[multi])]
                for _ in range(passes if method == "cd" else 1):
                    for lg in order:
                        rs = multi_ranks[int(lg)]
                        load[rs] -= alloc[int(lg)]
                        alloc[int(lg)] = _water_fill(load[rs], n[lg])
                        load[rs] += alloc[int(lg)]
            out[s, layer] = load
    return torch.tensor(out, device=DEV)


def moe_lb(loads: torch.Tensor) -> float:
    return float(loads.max(-1).values.sum() / loads.mean(-1).sum())


def local_search(sc: Scenario, history: torch.Tensor, pl: Placement, max_swaps: int = 300) -> Placement:
    """在 DeepInfer 原 policy 的结果上做两两交换局部搜索，目标与 _batch_greedy_indices 相同：
    sum_{step} max_{rank} (该 rank 上各副本的负载)，副本负载 = 该 expert 在该步的 token 数 / 副本数。
    只交换不同 rank 上的两个槽位，副本数不变，所以 log2phy 可以直接按新 phy2log 重建。"""
    L, P = pl.phy2log.shape
    R, local = sc.ranks, pl.local
    hist = history.float()  # [W, L, E]
    p2l = pl.phy2log.clone().long()
    for layer in range(L):
        vec = (hist[:, layer, :] / pl.logcnt[layer].float()).T  # [E, W] 每副本负载
        slot_vec = vec[p2l[layer]]  # [P, W]
        rank_of_slot = torch.arange(P, device=DEV) // local
        loads = torch.zeros(R, vec.shape[1], device=DEV).index_add_(0, rank_of_slot, slot_vec)
        for _ in range(max_swaps):
            top = loads.topk(3, dim=0)  # 每步前三大的 rank
            obj = top.values[0].sum()
            # 选对目标贡献最大的 rank（最常成为峰值的那个）
            i = int(torch.bincount(top.indices[0], minlength=R).argmax())
            a_slots = torch.arange(i * local, (i + 1) * local, device=DEV)
            other = torch.nonzero(rank_of_slot != i).squeeze(1)
            va = slot_vec[a_slots]  # [local, W]
            vb = slot_vec[other]  # [O, W]
            j = rank_of_slot[other]  # [O]
            new_i = loads[i].unsqueeze(0).unsqueeze(0) - va.unsqueeze(1) + vb.unsqueeze(0)  # [local, O, W]
            new_j = loads[j].unsqueeze(0) - vb.unsqueeze(0) + va.unsqueeze(1)  # [local, O, W]
            # 排除 i、j 后其余 rank 的峰值：取前三中第一个不是 i 也不是 j 的
            idx, val = top.indices, top.values  # [3, W]
            jj = j.view(1, -1, 1)
            c0 = (idx[0] != i).unsqueeze(0).unsqueeze(0) & (idx[0].view(1, 1, -1) != jj)
            c1 = (idx[1] != i).unsqueeze(0).unsqueeze(0) & (idx[1].view(1, 1, -1) != jj)
            rest = torch.where(c0, val[0], torch.where(c1, val[1], val[2]))  # [1, O, W]
            new_obj = torch.maximum(torch.maximum(new_i, new_j), rest).sum(-1)  # [local, O]
            best = new_obj.min()
            if best >= obj - 1e-6:
                break
            ka, kb = divmod(int(new_obj.argmin()), other.numel())
            sa, sb = int(a_slots[ka]), int(other[kb])
            jr = int(rank_of_slot[sb])
            loads[i] += slot_vec[sb] - slot_vec[sa]
            loads[jr] += slot_vec[sa] - slot_vec[sb]
            p2l[layer, sa], p2l[layer, sb] = p2l[layer, sb].clone(), p2l[layer, sa].clone()
            slot_vec[[sa, sb]] = slot_vec[[sb, sa]]
    # 按新 phy2log 重建 log2phy
    l2p = torch.full_like(pl.log2phy, -1)
    for layer in range(L):
        fill = torch.zeros(sc.experts, dtype=torch.long, device=DEV)
        for p_ in range(P):
            e = int(p2l[layer, p_])
            l2p[layer, e, fill[e]] = p_
            fill[e] += 1
    return Placement(p2l.int(), l2p, pl.logcnt, local)


def run(sc: Scenario, policies=POLICIES, verbose=True) -> dict:
    gen = torch.Generator(device=DEV)
    gen.manual_seed(sc.seed)
    drift = Drift(sc, gen)
    acc = {p: [] for p in policies}
    expected_acc = {p: [] for p in ("hash", "fresh")}  # 只看期望负载（无采样噪声）的不均衡，分离"放置误差"
    t0 = time.time()
    for cyc in range(sc.cycles):
        # 1) 统计窗口：连续 window 步
        hist = []
        for s0 in range(0, sc.window, sc.batch_steps):
            n = min(sc.batch_steps, sc.window - s0)
            idx = sample_routing(sc, drift.log_pop(), n, gen)
            hist.append(logical_counts(sc, idx))
            drift.advance(n)
        history = torch.cat(hist)  # [window, L, E]
        pl_eplb = eplb_placement(sc, history)
        pl_ls = local_search(sc, history, pl_eplb) if any(p.endswith('_ls') for p in policies) else None
        pl_none = no_eplb_placement(sc)
        # 2) 在接下来的 interval 内均匀抽样测量
        offsets = np.sort(np.random.default_rng(sc.seed * 1000 + cyc).choice(sc.interval, sc.measure_steps, replace=False))
        prev = 0
        for off in offsets:
            drift.advance(int(off - prev))
            prev = int(off)
            lp = drift.log_pop()
            idx = sample_routing(sc, lp, 1, gen)
            need_fresh = "fresh" in policies
            if need_fresh:
                tot = sc.ranks * sc.tokens_per_rank * sc.topk
                pl_fresh = eplb_placement(sc, (lp.exp() * tot).unsqueeze(0))
            for p in policies:
                if p == "none":
                    acc[p].append(dest_loads(sc, idx, pl_none, "hash"))
                elif p == "hash":
                    acc[p].append(dest_loads(sc, idx, pl_eplb, "hash"))
                elif p == "rr":
                    acc[p].append(dest_loads(sc, idx, pl_eplb, "rr"))
                elif p == "hash_ls":
                    acc[p].append(dest_loads(sc, idx, pl_ls, "hash"))
                elif p == "dyn_cd_ls":
                    acc[p].append(dyn_loads(sc, idx, pl_ls, method="cd"))
                elif p == "fresh":
                    acc[p].append(dest_loads(sc, idx, pl_fresh, "hash"))
                elif p.startswith("dyn_"):
                    acc[p].append(dyn_loads(sc, idx, pl_eplb, method=p[4:]))
            # 期望负载：tokens * p_e / 副本数，按放置汇总到 rank
            for p, pl in (("hash", pl_eplb), ("fresh", pl_fresh if need_fresh else pl_eplb)):
                per_phys = lp.exp().gather(-1, pl.phy2log.long()) / pl.logcnt.gather(-1, pl.phy2log.long())
                expected_acc[p].append(per_phys.view(sc.layers, sc.ranks, pl.local).sum(-1).unsqueeze(0))
        drift.advance(sc.interval - prev - sc.window)
        if verbose:
            cur = {p: round(moe_lb(torch.cat(acc[p])), 4) for p in policies}
            print(f"[{sc.name}] cycle {cyc + 1}/{sc.cycles} {time.time() - t0:.0f}s {cur}", flush=True)
    res = {p: moe_lb(torch.cat(acc[p])) for p in policies}
    res["expected_only_hash"] = moe_lb(torch.cat(expected_acc["hash"]))
    res["expected_only_fresh"] = moe_lb(torch.cat(expected_acc["fresh"]))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["decode", "prefill"], default="decode")
    ap.add_argument("--sigma-pop", type=float)
    ap.add_argument("--sigma-seg", type=float)
    ap.add_argument("--sigma-drift", type=float)
    ap.add_argument("--cycles", type=int)
    ap.add_argument("--measure-steps", type=int)
    ap.add_argument("--layers", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--out")
    a = ap.parse_args()
    sc = DECODE if a.scenario == "decode" else PREFILL
    kw = {k: v for k, v in dict(sigma_pop=a.sigma_pop, sigma_seg=a.sigma_seg, sigma_drift=a.sigma_drift, cycles=a.cycles,
                                measure_steps=a.measure_steps, layers=a.layers).items() if v is not None}
    sc = Scenario(**{**asdict(sc), **kw, "seed": a.seed})
    res = run(sc, tuple(a.policies.split(",")))
    out = {"scenario": asdict(sc), "num_physical": sc.num_physical, "num_local": sc.num_local, "moe_lb": res}
    print(json.dumps(out["moe_lb"], indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
