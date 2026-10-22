"""EPLB 校准扫描：对未知的路由统计参数做网格扫描，筛出能复现 DeepSeek 实测 moe_lb 的设置，再在这些设置上比较策略。

DeepSeek 实测（doc/inference_deployment_qa.md）：decode EP64 moe_lb = 1.090，prefill EP4 = 1.074。
"""

import os
import itertools
import json
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eplb_sim as m  # noqa: E402

scenario = sys.argv[1]
out_path = sys.argv[2]
base = m.DECODE if scenario == "decode" else m.PREFILL
if scenario == "decode":
    grid = itertools.product([0.5, 1.0], [0.0, 1.0, 2.0], [0.0, 0.1, 0.2])
    policies = ("hash", "rr", "fresh", "dyn_cd")
    kw = dict(cycles=2, measure_steps=60)
else:
    grid = itertools.product([0.5, 1.0], [0.5, 1.0, 2.0], [0.0, 0.1, 0.2])
    policies = ("hash", "rr", "fresh", "dyn_cd")
    kw = dict(cycles=3, measure_steps=60)

rows = []
for sigma_pop, sigma_seg, sigma_drift in grid:
    sc = m.Scenario(**{**asdict(base), **kw, "sigma_pop": sigma_pop, "sigma_seg": sigma_seg, "sigma_drift": sigma_drift})
    res = m.run(sc, policies, verbose=False)
    row = dict(sigma_pop=sigma_pop, sigma_seg=sigma_seg, sigma_drift=sigma_drift, **{k: round(v, 4) for k, v in res.items()})
    rows.append(row)
    print(json.dumps(row), flush=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=1)
