"""跨卡协调的两项补充检验：(1) 每用户延迟 TPOT 是否变差；(2) 小规模 EP（8 卡）是否仍成立。"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mr_sim as m  # noqa: E402


def one(a):
    ranks, sd, seed = a
    r = m.run(m.MR(ranks=ranks, sd_seqs=sd, ctx="agent", steps=1000, warmup=250, seed=seed), ("fixed5", "dspark", "coord_lite"))
    return a, {p: (r[p]["throughput_per_rank"], r[p]["tpot_ms"]) for p in r}


if __name__ == "__main__":
    jobs = [(R, sd, s) for R in (8, 64) for sd in (0, 6) for s in range(3)]
    with Pool(12) as pool:
        out = pool.map(one, jobs)
    agg = {}
    for (R, sd, s), r in out:
        for p, (thr, tpot) in r.items():
            agg.setdefault((R, sd, p), []).append((thr, tpot))
    res = {}
    for (R, sd, p), v in sorted(agg.items()):
        base = np.mean([x[0] for x in agg[(R, sd, "dspark")]])
        base_t = np.mean([x[1] for x in agg[(R, sd, "dspark")]])
        res[f"EP{R}/sd{sd}/{p}"] = {"thr_vs_dspark_%": round(100 * (np.mean([x[0] for x in v]) / base - 1), 2),
                                   "tpot_ms": round(float(np.mean([x[1] for x in v])), 3),
                                   "tpot_vs_dspark_%": round(100 * (np.mean([x[1] for x in v]) / base_t - 1), 2)}
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "mr_latency_scale.json"), "w"), indent=1)
