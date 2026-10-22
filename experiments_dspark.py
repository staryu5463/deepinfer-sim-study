"""L2 A/B：在校准后的仿真器上比较 DSpark 现状与改进方案。

策略：
    fixed5   固定验证 5 个 draft
    dspark   现状（MTPVarlen::schedule + ForwardRuntimeEstimator，逐行移植）
    ca_time  只换估时模型：forward 残差按 [G, Σ q_i w_i, Σ w_i] 回归（per-seq 的上下文开销不再摊到 token 上）
    ca       估时模型 + 代价感知排序：按 cum_conf / 边际开销(L_i) 排序选 token（CPU 定 total_q 与 GPU 分配都改）
    oracle   真实接受率 + 真实开销（上界，不可实现）

每个设置跑多个 seed（同 seed 下各策略共享随机数），报告相对 dspark 的吞吐提升均值与 95% 区间。
"""

import os
import json
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calibrate_dspark as cal  # noqa: E402
import dspark_sim as d  # noqa: E402

POLS = ("fixed5", "dspark", "ca_time", "ca", "oracle")


def load_params(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "dspark_calibration.json")):
    with open(path) as f:
        params = json.load(f)["params"]
    return [params[k] for k in cal.NAMES]


def run_setting(x, ctx, conc, seeds, hw_override=None, steps=2500, warmup=400):
    hw, acc = cal.build(x)
    if hw_override:
        hw = replace(hw, **hw_override)
    per = {p: [] for p in POLS}
    detail = {p: [] for p in POLS}
    for s in seeds:
        r = d.run_all(d.Workload(conc=conc, ctx=ctx, steps=steps, warmup=warmup, seed=s), hw, acc, POLS)
        for p in POLS:
            per[p].append(r[p]["throughput"] / r["dspark"]["throughput"] - 1)
            detail[p].append(r[p])
    out = {}
    for p in POLS:
        g = 100 * np.array(per[p])
        half = 1.96 * g.std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else 0.0
        out[p] = {
            "gain_vs_dspark_%": round(float(g.mean()), 2),
            "ci95": round(float(half), 2),
            "verified": round(float(np.mean([r["verified"] for r in detail[p]])), 2),
            "tau": round(float(np.mean([r["tau"] for r in detail[p]])), 3),
            "thr": round(float(np.mean([r["throughput"] for r in detail[p]]))),
        }
    return out


if __name__ == "__main__":
    x = load_params()
    seeds = range(6)
    results = {}
    for ctx in ("bench32k", "chat", "agent", "longmix"):
        for conc in (32, 60, 96):
            key = f"{ctx}/conc{conc}"
            results[key] = run_setting(x, ctx, conc, seeds)
            print(key, json.dumps({p: (v["gain_vs_dspark_%"], v["ci95"], v["verified"]) for p, v in results[key].items()}), flush=True)
    # 敏感性：上下文开销对硬件效率假设的依赖（有效带宽 / FP4 算力各 ×0.5 和 ×2）
    sens = {}
    for name, ov in {"very_slow_hw(x0.25)": dict(bw=1.5e12, fp4=1.25e15), "slow_hw(x0.5)": dict(bw=3e12, fp4=2.5e15), "fast_hw(x2)": dict(bw=12e12, fp4=10e15)}.items():
        for ctx in ("agent", "longmix"):
            key = f"{name}/{ctx}/conc60"
            sens[key] = run_setting(x, ctx, 60, seeds, hw_override=ov)
            print(key, json.dumps({p: (v["gain_vs_dspark_%"], v["ci95"], v["verified"]) for p, v in sens[key].items()}), flush=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "dspark_ab.json"), "w") as f:
        json.dump({"main": results, "sensitivity": sens}, f, indent=1)
