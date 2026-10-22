"""L0 正确性：确认 dspark_sim 的移植与 DeepInfer 源码行为一致。

1. GPU 分配阶段：与仓库自带的参考实现 compute_seqlens_q_by_confidence_rates_ref（原样拷贝，见下）逐元素比对，
   参数网格取自其 pytest 参数化（schedule_mtp_varlen.py）。
2. CPU best_bsz 扫描：与 mtp_varlen.hpp MTPVarlen::schedule 的逐行 Python 直译（循环版）比对向量化版本。
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dspark_sim as d  # noqa: E402


# ---- 原样拷贝自 deepinfer/model_executor/layers/tilelang_ops/schedule_mtp_varlen.py（仅删注释）
def compute_seqlens_q_by_confidence_rates_ref(total_q, remap, prev_confidence_rates, seqlens_q):
    num_seqs = remap.shape[0]
    nextn = prev_confidence_rates.shape[1]
    mask = remap != -1
    safe_remap = remap.clamp(min=0)
    gathered_rates = prev_confidence_rates[safe_remap]
    gathered_rates = torch.where(mask.unsqueeze(1), gathered_rates, torch.zeros_like(gathered_rates))
    cum_rates = torch.cumprod(gathered_rates, dim=1)
    extra_budget = total_q - num_seqs
    flat_rates = cum_rates.reshape(-1)
    sorted_indices = torch.argsort(flat_rates, descending=True, stable=True)
    ranks = torch.empty(len(flat_rates), device=remap.device, dtype=torch.int32)
    ranks[sorted_indices] = torch.arange(len(flat_rates), device=remap.device, dtype=torch.int32)
    ranks = ranks.reshape(num_seqs, nextn)
    wins = (ranks < extra_budget).to(torch.int32).sum(dim=1)
    seqlens_q.copy_(1 + wins)


def test_alloc_matches_reference():
    cases = [(32, 7, 32), (32, 7, 256), (4, 3, 8), (16, 7, 40), (64, 5, 128), (128, 3, 200), (256, 7, 300), (8, 1, 8), (1, 5, 4)]
    n = 0
    for seed in range(20):
        g = torch.Generator().manual_seed(seed)
        for num_seqs, nextn, total_q in cases:
            for has_remap in (True, False):
                n_prev = num_seqs + 16
                rates = torch.rand(n_prev, nextn, generator=g) * 0.8 + 0.1
                if seed % 3 == 0:
                    rates = (rates * 4).round() / 4  # 制造大量同值，检验 tie-break
                remap = torch.full((num_seqs,), -1, dtype=torch.int32)
                if has_remap:
                    k = num_seqs // 2
                    matched = torch.randperm(num_seqs, generator=g)[:k]
                    remap[matched] = torch.randint(0, n_prev, (k,), generator=g, dtype=torch.int32)
                ref = torch.empty(num_seqs, dtype=torch.int32)
                compute_seqlens_q_by_confidence_rates_ref(total_q, remap, rates, ref)

                fresh = (remap == -1).numpy()
                conf = rates[remap.clamp(min=0)].numpy()
                old_nextn = d.NEXTN
                d.NEXTN = nextn
                try:
                    v = d.StepView(num_seqs, np.zeros(num_seqs), conf, conf, fresh)
                    mine = d._alloc(v, total_q, None)
                finally:
                    d.NEXTN = old_nextn
                assert np.array_equal(mine, ref.numpy()), (seed, num_seqs, nextn, total_q, has_remap, mine, ref)
                n += 1
    print(f"alloc vs reference: {n} cases identical")


def best_bsz_literal(S, cum_prev_flat_sorted, graphs, baseline, slope, sampler_ms):
    """mtp_varlen.hpp 第 230–263 行的逐行直译。"""
    best_throughput, best_bsz = 0.0, 0
    cur_expected, cur_bsz, cur_bsz_idx = float(S), S, 0

    def update_best():
        nonlocal cur_bsz_idx, best_throughput, best_bsz
        while cur_bsz_idx < len(graphs) and cur_bsz > graphs[cur_bsz_idx]:
            cur_bsz_idx += 1
        if cur_bsz_idx == len(graphs):
            return
        graph_size = graphs[cur_bsz_idx]
        time_ms = baseline[graph_size] + slope * graph_size + sampler_ms
        thr = cur_expected / time_ms
        if thr > best_throughput:
            best_throughput, best_bsz = thr, cur_bsz

    update_best()
    for a in cum_prev_flat_sorted:
        cur_expected += a
        cur_bsz += 1
        update_best()
    return best_bsz


def test_best_bsz_matches_literal():
    rng = np.random.default_rng(0)
    hw = d.HW(max_graph=200)
    baseline = np.zeros(hw.graphs[-1] + 1)
    baseline[hw.graphs] = hw.B(hw.graphs)
    n = 0
    for _ in range(500):
        S = int(rng.integers(1, 33))
        conf = rng.random((S, d.NEXTN)) * 0.8 + 0.1
        if rng.random() < 0.3:
            conf = np.round(conf * 4) / 4
        fresh = rng.random(S) < 0.1
        slope, smp = float(rng.random() * 0.02), float(rng.random() * 2)
        v = d.StepView(S, np.full(S, 1000.0), conf, conf, fresh)
        mine = d._best_bsz(v, hw, baseline, lambda g, b, x: baseline[g] + slope * g + smp, None)
        prev = np.where(fresh[:, None], 1.0, conf)
        cum = np.cumprod(prev, axis=1).reshape(-1)
        order = np.lexsort((np.arange(cum.size), -cum))
        lit = best_bsz_literal(S, cum[order], hw.graphs, baseline, slope, smp)
        assert mine == lit, (mine, lit)
        n += 1
    print(f"best_bsz vs literal C++ transliteration: {n} cases identical")


if __name__ == "__main__":
    test_alloc_matches_reference()
    test_best_bsz_matches_literal()
