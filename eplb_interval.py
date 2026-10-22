"""EPLB：在校准设置上比较"缩短重排间隔"与"动态分流"，并给出多 seed 区间。"""

import os
import json
import sys
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eplb_sim as m  # noqa: E402

out_path = sys.argv[1]
calibrated = [dict(sigma_pop=0.5, sigma_seg=1.0, sigma_drift=0.1), dict(sigma_pop=0.5, sigma_seg=2.0, sigma_drift=0.1)]
rows = []
for cal in calibrated:
    for interval in (10000, 5000, 2500):
        vals = {p: [] for p in ("hash", "rr", "dyn_cd")}
        for seed in range(3):
            # 测量步数按间隔等比缩放，保证每个周期内的抽样密度相同
            sc = m.Scenario(**{**asdict(m.DECODE), **cal, "interval": interval, "cycles": max(2, 20000 // interval),
                               "measure_steps": max(20, 60 * interval // 10000), "seed": seed})
            r = m.run(sc, tuple(vals), verbose=False)
            for p in vals:
                vals[p].append(r[p])
        row = {**cal, "interval": interval, **{p: (round(float(np.mean(v)), 4), round(float(np.std(v)), 4)) for p, v in vals.items()}}
        rows.append(row)
        print(json.dumps(row), flush=True)
        with open(out_path, "w") as f:
            json.dump(rows, f, indent=1)
