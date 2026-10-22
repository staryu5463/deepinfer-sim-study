"""decode 负载不均衡网格：路由策略 × 路由信息延迟 × 长度分布形状 × 验证策略（EP64，2 seed）。"""

import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dlb_sim as s  # noqa: E402


def one(a):
    shape, routing, stale, policy, seed = a[:5]
    routers = a[5] if len(a) > 5 else 1
    r = s.simulate(s.Cfg(ranks=64, shape=shape, routing=routing, stale_steps=stale, policy=policy, routers=routers,
                         steps=2000, warmup=500, seed=seed))
    return a, r


if __name__ == "__main__":
    grid = sys.argv[1] if len(sys.argv) > 1 else "base"
    if grid == "base":
        jobs = [(sh, ro, stl, po, sd) for sh in ("toolagent", "conversation") for ro in ("random", "jsq_count", "jsq_work")
                for stl in (0, 20, 70) for po in ("dspark", "coord_lite") for sd in range(2)
                if not (ro == "random" and stl > 0)]
    elif grid == "own":  # 公平版：统一路由器记得自己的分配，只有离开信息过期
        jobs = [(sh, ro, stl, po, sd) for sh in ("toolagent", "conversation") for ro in ("jsq_own", "p2c_own")
                for stl in (20, 70) for po in ("dspark", "coord_lite") for sd in range(2)]
    elif grid == "routers":  # 多个独立路由器：每个只记得自己的分配
        jobs = [("toolagent", ro, stl, po, sd, K) for ro in ("jsq_own", "p2c_own") for stl in (20, 70) for K in (4, 16)
                for po in ("dspark", "coord_lite") for sd in range(2)]
    else:  # 路由层修复：p2c、按离开量修正的 jsq，与协调方案叠加
        jobs = [(sh, ro, stl, po, sd) for sh in ("toolagent", "conversation") for ro in ("p2c", "jsq_dep")
                for stl in (0, 20, 70) for po in ("dspark", "coord_lite") for sd in range(2)]
    with Pool(20) as pool:
        out = pool.map(one, jobs)
    json.dump([[list(a), r] for a, r in out], open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"dlb_grid_{grid}_raw.json"), "w"))
    agg = {}
    for a, r in out:
        sh, ro, stl, po = a[:4]
        if len(a) > 5:
            ro = f"{ro}_K{a[5]}"
        agg.setdefault((sh, ro, stl, po), []).append(r)
    rows = {}
    for (sh, ro, stl, po), rs in sorted(agg.items()):
        rows[f"{sh}/{ro}/stale{stl}/{po}"] = {k: round(float(np.mean([r[k] for r in rs])), 3) for k in rs[0]}
    for (sh, ro, stl) in sorted({k[:3] for k in agg}):
        b, c = rows[f"{sh}/{ro}/stale{stl}/dspark"], rows[f"{sh}/{ro}/stale{stl}/coord_lite"]
        rows[f"{sh}/{ro}/stale{stl}/coord_gain_%"] = round(100 * (c["throughput_per_gpu"] / b["throughput_per_gpu"] - 1), 2)
    for k, v in rows.items():
        print(k, json.dumps(v))
    json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"dlb_grid_{grid}.json"), "w"), indent=1)
