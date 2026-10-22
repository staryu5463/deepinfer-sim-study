"""L1 校准（扩展版）：在原 8 个参数之外，让 confidence head 可以有系统性偏差（conf_scale、conf_bias）。

问题：DeepSeek 表格里 conc=16 时 DSpark 比固定 5 慢 1.5%，原校准复现不出来（+0.2%）。
如果加入"head 未校准"后拟合显著变好、且能复现这一点，说明线上 head 可能有偏差，
那么用已经算出来的 accept_prob 做在线校准就有实际收益。拟合集/留出集与原版相同。
"""

import json
import os
import sys

from scipy.optimize import differential_evolution

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calibrate_dspark as cal  # noqa: E402
import dspark_sim as d  # noqa: E402

NAMES = cal.NAMES + ["conf_scale", "conf_bias"]
BOUNDS = cal.BOUNDS + [(0.3, 2.0), (-2.0, 2.0)]


_base_build = cal.build


def build(x):
    hw, acc = _base_build(x[:8])
    acc.conf_scale, acc.conf_bias = x[8], x[9]
    return hw, acc


cal.build = build  # 复用原版的 loss / report


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "dspark_calibration_v2.json")
    res = differential_evolution(cal.loss, BOUNDS, seed=0, maxiter=40, popsize=12, tol=1e-6, workers=-1,
                                 updating="deferred", polish=False, disp=True)
    rows = cal.report(res.x)
    print(json.dumps(dict(zip(NAMES, map(float, res.x))), indent=1), "loss", res.fun)
    for c, r in rows.items():
        print(c, r)
    with open(out, "w") as f:
        json.dump({"params": dict(zip(NAMES, map(float, res.x))), "loss": float(res.fun),
                   "rows": {str(k): v for k, v in rows.items()}}, f, indent=1)
