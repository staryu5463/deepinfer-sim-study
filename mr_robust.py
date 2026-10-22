"""协调策略在耗时模型估错时的鲁棒性：真实 attention 占比 0.45，策略内部分别以为是 0.3 / 0.45 / 0.6。"""
import os
import json
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mr_sim as m  # noqa: E402


def one(a):
    belief, seed = a
    r = m.run(m.MR(ranks=64, sd_seqs=6, ctx="agent", f_loc=0.45, f_loc_belief=belief, steps=1000, warmup=250, seed=seed),
              ("dspark", "coord_lite"))
    return a, r["coord_lite"]["gain_vs_dspark_%"]


if __name__ == "__main__":
    jobs = [(b, s) for b in (0.3, 0.45, 0.6) for s in range(3)]
    with Pool(9) as pool:
        out = pool.map(one, jobs)
    res = {}
    for (b, s), g in out:
        res.setdefault(str(b), []).append(g)
    print(json.dumps({b: (round(sum(v) / len(v), 2), v) for b, v in res.items()}))
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "mr_robustness.json"), "w"))
