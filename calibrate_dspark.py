"""L1 校准：用 DeepSeek 公开表格（doc/inference_deployment_qa.md，EP64/DP64，32K 上下文 benchmark）拟合仿真器的未知参数。

拟合集 conc = 16, 60, 96；留出集 conc = 32, 48, 72（拟合时不看，用来检验仿真器能否外推）。
"""

import os
import json
import sys

import numpy as np
from scipy.optimize import differential_evolution

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dspark_sim as d  # noqa: E402

# conc: (dyn 平均验证长度, fixed τ, dyn τ, fixed 吞吐, dyn 吞吐)
TABLE = {
    16: (4.44, 3.32, 3.24, 4891, 4816),
    32: (3.65, 3.25, 3.17, 8964, 9111),
    48: (3.94, 3.30, 3.26, 12032, 12467),
    60: (3.58, 3.30, 3.21, 13730, 14466),
    72: (3.24, 3.29, 3.12, 14962, 15961),
    96: (3.24, 3.30, 3.15, 16817, 18660),
}
FIT = (16, 60, 96)
HOLDOUT = (32, 48, 72)
NAMES = ["T0", "a", "b", "c_seq", "phi", "m", "s_mu", "s_eta"]
BOUNDS = [(6, 12), (0.002, 0.02), (0.0, 2e-5), (0.0, 0.06), (0.0, 1.0), (-0.5, 3.0), (0.1, 2.5), (0.05, 2.0)]
STEPS, WARMUP = 900, 200


def build(x):
    T0, a, b, c_seq, phi, m, s_mu, s_eta = x
    return d.HW(T0=T0, a=a, b=b, c_seq=c_seq, phi=phi), d.Accept(m=m, s_mu=s_mu, s_eta=s_eta)


def sim_row(x, conc, steps=STEPS, warmup=WARMUP, seed=0):
    hw, acc = build(x)
    r = d.run_all(d.Workload(conc=conc, steps=steps, warmup=warmup, seed=seed), hw, acc, ("fixed5", "dspark"))
    f, y = r["fixed5"], r["dspark"]
    return dict(verified=y["verified"], tau_f=f["tau"], tau_d=y["tau"], thr_f=f["throughput"], thr_d=y["throughput"],
                gain=y["throughput"] / f["throughput"] - 1)


def loss(x, concs=FIT):
    tot = 0.0
    for c in concs:
        v, tf, td, Tf, Td = TABLE[c]
        s = sim_row(x, c)
        tot += ((s["thr_f"] - Tf) / Tf / 0.01) ** 2  # 吞吐 1% 精度
        tot += ((s["tau_f"] - tf) / 0.03) ** 2
        tot += ((s["tau_d"] - td) / 0.03) ** 2
        tot += ((s["verified"] - v) / 0.15) ** 2
        tot += ((s["gain"] - (Td / Tf - 1)) / 0.01) ** 2  # 收益 1 个百分点精度
    return tot


def report(x):
    rows = {}
    for c in sorted(TABLE):
        v, tf, td, Tf, Td = TABLE[c]
        s = sim_row(x, c, steps=2000, warmup=300)
        rows[c] = {
            "set": "fit" if c in FIT else "holdout",
            "verified": (round(s["verified"], 2), v),
            "tau_fixed": (round(s["tau_f"], 2), tf),
            "tau_dyn": (round(s["tau_d"], 2), td),
            "thr_fixed": (round(s["thr_f"]), Tf),
            "gain_%": (round(100 * s["gain"], 1), round(100 * (Td / Tf - 1), 1)),
        }
    return rows


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "dspark_calibration.json")
    res = differential_evolution(loss, BOUNDS, seed=0, maxiter=40, popsize=12, tol=1e-6, workers=-1, updating="deferred",
                                 polish=False, disp=True)
    x = res.x
    rows = report(x)
    print(json.dumps(dict(zip(NAMES, map(float, x))), indent=1))
    for c, r in rows.items():
        print(c, r)
    with open(out, "w") as f:
        json.dump({"params": dict(zip(NAMES, map(float, x))), "loss": float(res.fun), "rows": {str(k): v for k, v in rows.items()}}, f, indent=1)
