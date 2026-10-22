"""rank 间不均衡损失了多少吞吐，协调策略能追回多少（agent 上下文，64 rank，3 seed）。"""
import os
import json
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mr_sim as m  # noqa: E402


def one(a):
    sd, seed = a
    r = m.run(m.MR(ranks=64, sd_seqs=sd, ctx="agent", steps=1000, warmup=250, seed=seed), ("dspark", "coord_lite"))
    return a, {p: r[p]["throughput_per_rank"] for p in r}


if __name__ == "__main__":
    jobs = [(sd, s) for sd in (0, 6, 12) for s in range(3)]
    with Pool(9) as pool:
        out = pool.map(one, jobs)
    agg = {}
    for (sd, s), r in out:
        for p, v in r.items():
            agg.setdefault((sd, p), []).append(v)
    base = np.mean(agg[(0, "dspark")])
    res = {f"sd{sd}/{p}": round(100 * (np.mean(v) / base - 1), 2) for (sd, p), v in sorted(agg.items())}
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "mr_recovery.json"), "w"), indent=1)
