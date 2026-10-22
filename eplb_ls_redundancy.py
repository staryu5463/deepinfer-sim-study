"""EPLB 两个新杠杆：(1) 在原 policy 结果上做局部搜索；(2) 每卡冗余 expert 数 1/2/3（decode 显存不紧）。校准设置，3 seed。"""
import os
import json
import sys
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eplb_sim as m  # noqa: E402

cal = dict(sigma_pop=0.5, sigma_seg=1.0, sigma_drift=0.1)
rows = []
for extra in (1, 2, 3):
    pols = ("hash", "dyn_cd")
    vals = {p: [] for p in pols}
    for seed in range(3):
        sc = m.Scenario(**{**asdict(m.DECODE), **cal, "min_extra_per_rank": extra, "cycles": 2, "measure_steps": 60, "seed": seed})
        r = m.run(sc, pols, verbose=False)
        for p in pols:
            vals[p].append(r[p])
    row = {"extra_per_rank": extra, "local_experts": sc.num_local, **{p: (round(float(np.mean(v)), 4), round(float(np.std(v)), 4)) for p, v in vals.items()}}
    rows.append(row)
    print(json.dumps(row), flush=True)
    json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "eplb_ls_redundancy.json"), "w"), indent=1)
