"""用 DeepSeek 线上真实参数重建 decode 负载不均衡模型（EP64 一个实例）。

参数来源：
- 每卡并发 61、每用户 230 tok/s、平均输入 77K、平均输出约 720（pain_econ.py 从官方线上数据推出）。
- 长度分布的形状来自 Kimi 生产 trace（Mooncake toolagent / conversation），缩放到上面的均值；上下文上限 1M。
- 时间模型沿用已校准的单 rank 模型（results/dspark_calibration.json）与多 rank 耦合模型（mr_sim.py）：
    每步时间 = T0 + max_r 本地 attention 时间 A_r + MoE 时间 × moe_lb + max_r draft/采样时间
- 闭环：总请求数固定为 R × 61；某请求结束就路由一个新请求进来（产能打满时的线上状态）。

路由（decode LB）：
    random      随机
    jsq_count   发到请求数最少的卡
    jsq_work    发到"加入后预测 attention 时间最小"的卡（同时考虑请求数与 KV 长度，对应 DeepSeek 的二维均衡）
    p2c         随机挑两张卡，发给其中预测 attention 时间较小的（power-of-two-choices，对过期信息鲁棒）
    jsq_dep     jsq_work，但先按"快照之后预计已结束的请求数"修正过期快照（请求寿命约 225 步，按比例扣减）
    jsq_own / p2c_own  路由器精确记得自己在快照之后往每张卡分了多少请求（统一路由器），只有"请求何时结束"是过期信息
    stale_steps 路由用 stale_steps 步之前的状态快照（LB 跨实例全局路由，信息有延迟）
不均衡损失 = 1 - Σ T_balanced / Σ T_actual，同一批决策下把各项的 max 换成 mean 得到"完美均衡"时间，并按来源拆解。
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calibrate_dspark as cal  # noqa: E402
import dspark_sim as d  # noqa: E402
import mr_sim as m  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NEXTN = d.NEXTN


def length_samplers(shape: str, rng: np.random.Generator, mean_in=77e3, mean_out=719.0, cap_in=1_000_000):
    rows = [json.loads(line) for line in open(os.path.join(HERE, "data", f"{shape}_trace.jsonl"))]
    ins = np.array([r["input_length"] for r in rows], dtype=float)
    outs = np.array([max(1, r["output_length"]) for r in rows], dtype=float)
    ins *= mean_in / ins.mean()
    outs *= mean_out / outs.mean()

    def sample(n):
        idx = rng.integers(0, len(rows), n)
        return np.minimum(ins[idx], cap_in), np.maximum(1.0, np.round(outs[idx]))

    return sample


@dataclass
class Cfg:
    ranks: int = 64
    conc: int = 61
    shape: str = "toolagent"
    routing: str = "jsq_work"
    stale_steps: int = 0
    routers: int = 1  # *_own 路由时的独立路由器个数：每个只记得自己的分配
    policy: str = "dspark"
    f_loc: float = 0.45
    moe_lb: float = 1.09
    steps: int = 2500
    warmup: int = 500
    seed: int = 0


class Ranks:
    """每个 rank 的活跃请求（结构数组）。"""

    FIELDS = ("L", "rem", "mu", "eps", "fresh")

    def __init__(self, R):
        self.data = [{f: np.zeros(0) for f in self.FIELDS} | {"conf": np.zeros((0, NEXTN))} for _ in range(R)]

    def add(self, r, L, rem, mu, eps):
        x = self.data[r]
        x["L"] = np.append(x["L"], L)
        x["rem"] = np.append(x["rem"], rem)
        x["mu"] = np.append(x["mu"], mu)
        x["eps"] = np.append(x["eps"], eps)
        x["fresh"] = np.append(x["fresh"], 1.0)
        x["conf"] = np.vstack([x["conf"], np.ones((1, NEXTN))])

    def keep(self, r, mask):
        x = self.data[r]
        for f in self.FIELDS:
            x[f] = x[f][mask]
        x["conf"] = x["conf"][mask]


def simulate(cfg: Cfg) -> dict:
    x = [json.load(open(os.path.join(HERE, "results", "dspark_calibration.json")))["params"][k] for k in cal.NAMES]
    hw, acc = cal.build(x)
    mr = m.MR(ranks=cfg.ranks, f_loc=cfg.f_loc)
    rng = np.random.default_rng(cfg.seed)
    sample = length_samplers(cfg.shape, rng)
    R = cfg.ranks
    st = Ranks(R)
    p_tok = (6.0 / hw.bw + 8192.0 / hw.fp4) * 1e3
    kvc = 68.0 / hw.bw * 1e3

    def pred_attn(S, L_sum_pairs_tok, L_sum_pairs_seq):
        # 每卡 attention 侧预测（按平均每请求 4 个 token）：用于 jsq_work 路由
        return m.local_time(hw, mr, 4 * S, S, L_sum_pairs_tok * p_tok * 4 + L_sum_pairs_seq * kvc)

    # 初始化：每卡 conc 个请求，剩余输出取"剩余寿命"（长度加权后均匀截断），近似稳态
    for r in range(R):
        L, out = sample(cfg.conc)
        rem = np.maximum(1, np.round(out * rng.random(cfg.conc)))
        for i in range(cfg.conc):
            st.add(r, L[i], rem[i], acc.m + acc.s_mu * rng.standard_normal(), acc.s_eps * rng.standard_normal())

    snap_hist = []
    assign_hist = []  # 每步路由器往各卡分配的 (请求数, pairs) —— *_own 路由器自己知道
    baseline = np.zeros(hw.graphs[-1] + 1)
    baseline[hw.graphs] = hw.B(hw.graphs)
    ctxs = [{"fwd": d.ForwardRuntimeEstimator(), "smp": d.SamplerEstimator(), "ca": d.CtxAwareEstimator(), "baseline": baseline}
            for _ in range(R)]
    H_lite, lam_prev = None, None
    tot = {"tok": 0.0, "T": 0.0, "T_bal": 0.0, "T_att_bal": 0.0, "T_moe_bal": 0.0}
    S_cv = []
    j = np.arange(NEXTN)
    for t in range(cfg.steps):
        views, pis, orders = [], [], []
        for r in range(R):
            a = st.data[r]
            S = len(a["L"])
            logit = a["mu"][:, None] + a["eps"][:, None] - acc.kappa * j + acc.s_nu * rng.standard_normal((S, NEXTN))
            pi = 1 / (1 + np.exp(-logit))
            conf = 1 / (1 + np.exp(-(logit + acc.s_eta * rng.standard_normal((S, NEXTN)))))
            order = np.argsort(-a["L"], kind="stable")
            views.append(d.StepView(S, a["L"][order], a["conf"][order], conf[order], a["fresh"][order] > 0))
            pis.append(pi[order])
            orders.append(order)
            a["conf"] = conf  # 本步 draft 的 confidence，下一步作为 conf_prev

        if cfg.policy == "fixed5":
            qs = [d.schedule_fixed(v, hw, None) for v in views]
        elif cfg.policy == "dspark" or (cfg.policy == "coord_lite" and lam_prev is None):
            qs = [d.schedule_dspark(v, hw, ctxs[r]) for r, v in enumerate(views)]
        else:  # coord_lite：每步 all-reduce 两个标量，把各 rank 填到公共 attention 水位 H
            pres = [m._prefix(v, hw, mr) for v in views]
            if H_lite is None:
                H_lite = float(np.median([p["A"][min(len(p["A"]) - 1, int(0.7 * NEXTN * v.S))] for p, v in zip(pres, views)]))
            ks = [m._k_at_level(p, H_lite) for p in pres]
            qs = [m._q_from_k(v, p, k) for v, p, k in zip(views, pres, ks)]
            s_sum = z_sum = 0.0
            for p, k in zip(pres, ks):
                if k + 1 < len(p["A"]):
                    da = max(p["A"][k + 1] - p["A"][k], 1e-6)
                    s_sum += p["gains"][k] / da
                    z_sum += 1.0 / da
            Gbar = float(np.mean([p["G"][k] for p, k in zip(pres, ks)]))
            dM = m.moe_time(hw, mr, Gbar + 1) - m.moe_time(hw, mr, Gbar)
            grad = s_sum - lam_prev * (1.0 + dM / R * z_sum)
            H_lite *= float(np.exp(0.05 * np.tanh(grad / max(abs(s_sum), 1e-9) * 3)))

        # 时间：每 rank 本地 attention，MoE 全局
        A, smp, Gs = np.zeros(R), np.zeros(R), np.zeros(R)
        tokens = 0.0
        n_done = 0
        for r in range(R):
            v, q, pi, order = views[r], qs[r], pis[r], orders[r]
            g = hw.graph_of(int(q.sum()))
            Gs[r] = g
            A[r] = m.local_time(hw, mr, g, v.S, float(hw.ctx_cost(v.L, q).sum()))
            smp[r] = (1 - hw.phi) * hw.c_seq * v.S
            lead = np.cumprod(rng.random((v.S, NEXTN)) < pi, axis=1)
            accepted = (lead * (j < (q - 1)[:, None])).sum(1)
            produced = 1 + accepted
            tokens += float(produced.sum())
            a = st.data[r]
            inv = np.empty_like(order)
            inv[order] = np.arange(len(order))
            prod_orig = produced[inv]
            a["rem"] = a["rem"] - prod_orig
            a["L"] = a["L"] + prod_orig
            a["eps"] = acc.rho * a["eps"] + np.sqrt(1 - acc.rho**2) * acc.s_eps * rng.standard_normal(len(a["eps"]))
            a["fresh"][:] = 0
            alive = a["rem"] > 0
            n_done += int((~alive).sum())
            st.keep(r, alive)
        M_bal = m.moe_time(hw, mr, float(Gs.mean()))
        T = hw.T0 + A.max() + M_bal * cfg.moe_lb + smp.max()
        T_bal = hw.T0 + A.mean() + M_bal + smp.mean()
        # 运行时反馈（与源码一致：每个 rank 看到的是全局 forward 时间）
        fwd = hw.T0 + A.max() + M_bal * cfg.moe_lb
        for r in range(R):
            ctxs[r]["fwd"].update(int(Gs[r]), (fwd - baseline[int(Gs[r])]) / Gs[r])
            ctxs[r]["smp"].update(views[r].S, smp.max())
        lam_prev = tokens / T

        # 闭环补充新请求：按路由策略分发
        snap = [(len(st.data[r]["L"]), float(d.pairs(st.data[r]["L"]).sum())) for r in range(R)]
        snap_hist.append(snap)
        view_snap = snap_hist[max(0, len(snap_hist) - 1 - cfg.stale_steps)]
        cnt = np.array([s[0] for s in view_snap], dtype=float)
        pairs_sum = np.array([s[1] for s in view_snap])
        if cfg.routing == "jsq_dep" and cfg.stale_steps > 0:
            age = len(snap_hist) - 1 - max(0, len(snap_hist) - 1 - cfg.stale_steps)
            keep = max(0.0, 1.0 - age / 225.0)  # 平均寿命约 225 步：快照里的请求按比例视为已结束
            # 快照之后由本路由器发出的请求，路由器自己知道（按每步平均补充量近似）
            assigned = sum(len(st.data[r]["L"]) for r in range(R)) / R - cnt.mean() * keep
            cnt = cnt * keep + max(assigned, 0.0)
            pairs_sum = pairs_sum * keep + pairs_sum.mean() * (1 - keep)
        K = cfg.routers if cfg.routing.endswith("_own") else 1
        cnt_k = [cnt.copy() for _ in range(K)]
        pairs_k = [pairs_sum.copy() for _ in range(K)]
        if cfg.routing.endswith("_own") and cfg.stale_steps > 0:
            for hist in assign_hist[-cfg.stale_steps:]:
                for k in range(K):
                    cnt_k[k] = cnt_k[k] + hist[k][0]
                    pairs_k[k] = pairs_k[k] + hist[k][1]
        step_assign = [(np.zeros(R), np.zeros(R)) for _ in range(K)]
        L_new, out_new = sample(n_done) if n_done else (np.zeros(0), np.zeros(0))
        for i in range(n_done):
            k_r = int(rng.integers(K))  # 这个请求由哪个路由器处理
            cnt, pairs_sum = cnt_k[k_r], pairs_k[k_r]
            if cfg.routing == "random":
                r = int(rng.integers(R))
            elif cfg.routing == "jsq_count":
                r = int(np.argmin(cnt + 1e-3 * rng.random(R)))
            elif cfg.routing in ("p2c", "p2c_own"):
                pw = d.pairs(np.array([L_new[i]]))[0]
                c2 = rng.choice(R, 2, replace=False)
                after = pred_attn(cnt[c2] + 1, pairs_sum[c2] + pw, pairs_sum[c2] + pw)
                r = int(c2[int(np.argmin(after))])
            else:  # jsq_work / jsq_dep / jsq_own
                pw = d.pairs(np.array([L_new[i]]))[0]
                after = pred_attn(cnt + 1, pairs_sum + pw, pairs_sum + pw)
                r = int(np.argmin(after + 1e-9 * rng.random(R)))
            cnt[r] += 1
            pw_i = d.pairs(np.array([L_new[i]]))[0]
            pairs_sum[r] += pw_i
            step_assign[k_r][0][r] += 1
            step_assign[k_r][1][r] += pw_i
            st.add(r, L_new[i], out_new[i], acc.m + acc.s_mu * rng.standard_normal(), acc.s_eps * rng.standard_normal())

        assign_hist.append(step_assign)
        if t >= cfg.warmup:
            tot["tok"] += tokens
            tot["T"] += T
            tot["T_bal"] += T_bal
            tot["T_att_bal"] += hw.T0 + A.mean() + M_bal * cfg.moe_lb + smp.max()
            tot["T_moe_bal"] += hw.T0 + A.max() + M_bal + smp.max()
            S_now = np.array([len(st.data[r]["L"]) for r in range(R)])
            S_cv.append(S_now.std() / S_now.mean())
    n = cfg.steps - cfg.warmup
    return {
        "throughput_per_gpu": tot["tok"] / tot["T"] * 1e3 / R,
        "user_tps": tot["tok"] / tot["T"] * 1e3 / (R * cfg.conc),
        "step_ms": tot["T"] / n,
        "loss_total_%": 100 * (1 - tot["T_bal"] / tot["T"]),
        "loss_attention_side_%": 100 * (1 - tot["T_att_bal"] / tot["T"]),
        "loss_moe_lb_%": 100 * (1 - tot["T_moe_bal"] / tot["T"]),
        "conc_cv_across_ranks": float(np.mean(S_cv)),
    }


if __name__ == "__main__":
    c = Cfg(ranks=int(sys.argv[1]) if len(sys.argv) > 1 else 16, steps=600, warmup=150)
    print(json.dumps(simulate(c), indent=1))
