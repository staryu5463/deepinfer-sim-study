"""EPLB 重排后做"迁移最小化重标号"：负载均衡质量只取决于哪些 expert 分在同一张卡上（一个划分），
卡的编号、卡内槽位顺序都可以任意置换。用匈牙利算法把新划分的每一组匹配到重叠最多的旧卡，
卡内保留已在原位的 expert，只搬真正换组的那部分。均衡质量与原方案逐位相同（同一个划分）。
"""

import os
import json
import sys
from dataclasses import asdict

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eplb_sim as m  # noqa: E402


def relabel(prev_p2l: np.ndarray, new_p2l: np.ndarray, R: int, local: int) -> np.ndarray:
    """单层：返回与 new_p2l 等价（同一划分、同副本数）但尽量保持 prev 位置的 phy2log。"""
    prev_g = prev_p2l.reshape(R, local)
    new_g = new_p2l.reshape(R, local)
    overlap = np.zeros((R, R))
    for i in range(R):  # 新组 i
        for j in range(R):  # 旧卡 j
            a, b = np.bincount(new_g[i], minlength=400), np.bincount(prev_g[j], minlength=400)
            overlap[i, j] = np.minimum(a, b).sum()
    rows, cols = linear_sum_assignment(-overlap)
    out = np.empty_like(prev_g)
    for i, j in zip(rows, cols):
        want = list(new_g[i])
        slot = [-1] * local
        for s, e in enumerate(prev_g[j]):  # 原位保留
            if e in want:
                slot[s] = e
                want.remove(e)
        it = iter(want)
        out[j] = [e if e != -1 else next(it) for e in slot]
    return out.reshape(-1)


def balance_signature(p2l: np.ndarray, R: int, local: int):
    return sorted(tuple(sorted(g)) for g in p2l.reshape(R, local).tolist())


if __name__ == "__main__":
    cal = dict(sigma_pop=0.5, sigma_seg=1.0, sigma_drift=0.1)
    res = {}
    for interval in (10000, 2500):
        sc = m.Scenario(**{**asdict(m.DECODE), **cal, "layers": 2, "interval": interval, "seed": 5})
        gen = torch.Generator(device=m.DEV)
        gen.manual_seed(sc.seed)
        drift = m.Drift(sc, gen)
        prev, orig_moved, new_moved = None, [], []
        for cyc in range(4):
            hist = []
            for _ in range(0, sc.window, sc.batch_steps):
                hist.append(m.logical_counts(sc, m.sample_routing(sc, drift.log_pop(), sc.batch_steps, gen)))
                drift.advance(sc.batch_steps)
            p2l = m.eplb_placement(sc, torch.cat(hist)).phy2log.cpu().numpy()
            if prev is not None:
                for layer in range(sc.layers):
                    r = relabel(prev[layer], p2l[layer], sc.ranks, sc.num_local)
                    assert balance_signature(r, sc.ranks, sc.num_local) == balance_signature(p2l[layer], sc.ranks, sc.num_local)
                    orig_moved.append(float((p2l[layer] != prev[layer]).mean()))
                    new_moved.append(float((r != prev[layer]).mean()))
                    p2l[layer] = r  # 下一轮以重标号后的布局为"旧布局"
            prev = p2l
            drift.advance(sc.interval - sc.window)
        res[interval] = {"moved_original": round(float(np.mean(orig_moved)), 3), "moved_relabeled": round(float(np.mean(new_moved)), 3)}
        print(interval, res[interval], flush=True)
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "eplb_relabel.json"), "w"), indent=1)
