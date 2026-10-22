"""多 rank 耦合 A/B：rank 间请求数不均（sd_seqs）× 上下文分布 × attention 占比 f_loc。"""

import os
import json
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mr_sim as m  # noqa: E402

POLS = ("fixed5", "dspark", "coord_lite", "coord")


def one(args):
    ctx, sd, f_loc, seed = args
    r = m.run(m.MR(ranks=64, mean_seqs=60, sd_seqs=sd, ctx=ctx, f_loc=f_loc, steps=1000, warmup=250, seed=seed), POLS)
    return args, {p: (r[p]["gain_vs_dspark_%"], r[p]["G_cv_across_ranks"]) for p in POLS}


if __name__ == "__main__":
    grid = [(c, sd, 0.45) for c in ("bench32k", "agent", "longmix") for sd in (0, 3, 6)]
    grid += [("agent", 6, f) for f in (0.3, 0.6)]
    jobs = [(c, sd, f, s) for (c, sd, f) in grid for s in range(3)]
    with Pool(min(len(jobs), 20)) as pool:
        out = pool.map(one, jobs)
    rows = {}
    for (c, sd, f, s), r in out:
        rows.setdefault(f"{c}/sd{sd}/floc{f}", []).append(r)
    summary = {}
    for k, rs in rows.items():
        summary[k] = {p: (round(float(np.mean([r[p][0] for r in rs])), 2), round(float(np.std([r[p][0] for r in rs])), 2),
                          round(float(np.mean([r[p][1] for r in rs])), 3)) for p in POLS}
        print(k, json.dumps(summary[k]), flush=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "mr_ab.json"), "w") as f:
        json.dump(summary, f, indent=1)
