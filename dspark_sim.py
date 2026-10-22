"""DSpark 变长验证调度的验证器。

被测对象：csrc/engine/mtp_varlen.hpp 的 MTPVarlen::schedule（逐行移植，见 schedule_dspark），
以及它的估时模型 ForwardRuntimeEstimator（slope = (forward - baseline(graph)) / graph，两层中位数）。

真值硬件模型（单个 DP rank，一步 decode）：
    forward = B(graph) + imb * G + sum_i ctx_cost(L_i, q_i) + phi * c_seq * S        （乘噪声）
    sampler = (1 - phi) * c_seq * S                                                 （draft 模型 + 采样，按请求数）
    B(G)    = T0 + a*G + b*G^2   —— 对应 profile_decode_perf：G 个 q=1、ctx=1 的 dummy 请求
    ctx_cost 来自 V4.1 真实结构（v41_config.json）：
        索引层 2/8/14 (ratio 2, dense)、20 (ratio 1, dense)、24/28/32/36 (候选块上限 2048*8)
        pairs(L) = 3 * L/2 + L + 4 * min(L, 16384)                  每个 query token 要打分的压缩 key 数
        每请求：读 index-K cache   68 B * pairs / BW                （同一请求的多个 q token 共享，DeepGEMM kv_shared_with_prev）
        每 token：logits 写+topk 读  6 B * pairs / BW
        每 Q-block(4 token)：MMA    4 * 8192 * pairs / FP4           （BLOCK_Q * 32 heads <= 128）

接受模型：logit(pi_ij) = mu_i + eps_i(t) - kappa * j + nu_ij，confidence head 输出 sigmoid(logit + eta)。
所有策略共享同一套随机数（请求生命周期、接受判定 U_ij、噪声），差异只来自调度决策。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

NEXTN = 5
CAND_CAP = 2048 * 8


def pairs(L: np.ndarray) -> np.ndarray:
    L = np.asarray(L, dtype=np.float64)
    return 3 * (L / 2) + L + 4 * np.minimum(L, CAND_CAP)


def build_graph_batch_sizes(max_size: int, interval: int = 8) -> np.ndarray:
    # 同 deepinfer/worker/cuda_graph_runner.py
    interval = min(interval, max_size)
    out, size = [], 1
    while size < interval:
        out.append(size)
        size *= 2
    out.extend(range(interval, max_size, interval))
    out.append(max_size)
    return np.array(out)


@dataclass
class HW:
    T0: float = 9.3  # ms
    a: float = 0.0080  # ms / token
    b: float = 6.0e-6  # ms / token^2
    c_seq: float = 0.020  # ms / 请求
    phi: float = 0.0  # 每请求开销中落在 forward 里的比例（会被 slope 误摊到 token 上）
    imb: float = 0.002  # ms / token：profile 里没有的逐 token 开销（真实稀疏注意力 gather、路由不均）
    bw: float = 6e12  # B/s，HBM 有效带宽
    fp4: float = 5e15  # FLOP/s，FP4 有效算力
    noise: float = 0.02
    max_graph: int = 128 * (NEXTN + 1)

    def __post_init__(self) -> None:
        self.graphs = build_graph_batch_sizes(self.max_graph)

    def B(self, G):
        G = np.asarray(G, dtype=np.float64)
        return self.T0 + self.a * G + self.b * G * G

    def graph_of(self, total_q: int) -> int:
        return int(self.graphs[np.searchsorted(self.graphs, total_q)])

    def ctx_cost(self, L: np.ndarray, q: np.ndarray) -> np.ndarray:
        """每请求的上下文相关开销 (ms)。"""
        p = pairs(L)
        kv = 68.0 * p / self.bw
        logits = q * 6.0 * p / self.bw
        mma = np.ceil(q / 4.0) * 4 * 8192.0 * p / self.fp4
        return (kv + logits + mma) * 1e3

    def marginal_token_cost(self, L: np.ndarray) -> np.ndarray:
        """光滑化的单 token 边际开销 (ms)，不含 B(graph) 的斜率。"""
        p = pairs(L)
        return self.imb + (6.0 * p / self.bw + 8192.0 * p / self.fp4) * 1e3


@dataclass
class Accept:
    m: float = 1.2
    s_mu: float = 1.0
    s_eps: float = 0.6
    rho: float = 0.8
    kappa: float = 0.35
    s_nu: float = 0.6
    s_eta: float = 0.6  # confidence head 误差（logit 空间）
    conf_scale: float = 1.0  # confidence head 的系统性偏差：conf_logit = conf_scale * logit + conf_bias + 噪声
    conf_bias: float = 0.0


@dataclass
class Workload:
    conc: int = 60
    ctx: str = "bench32k"  # bench32k | chat | agent | longmix
    mean_life_steps: float = 600.0
    steps: int = 3000
    warmup: int = 300
    seed: int = 0


def sample_ctx(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "bench32k":
        return np.full(n, 32768.0)
    if kind == "chat":
        return np.clip(rng.lognormal(math.log(6000), 1.0, n), 512, 131072)
    if kind == "agent":
        return np.clip(rng.lognormal(math.log(60000), 0.9, n), 4096, 1_000_000)
    if kind == "longmix":
        short = np.clip(rng.lognormal(math.log(8000), 1.0, n), 512, 131072)
        long = rng.uniform(262144, 1_000_000, n)
        return np.where(rng.random(n) < 0.3, long, short)
    raise ValueError(kind)


class World:
    """预先生成全部随机性（common random numbers）。"""

    def __init__(self, wl: Workload, acc: Accept) -> None:
        rng = np.random.default_rng(wl.seed)
        S, T = wl.conc, wl.steps
        self.S, self.T = S, T
        self.mu = np.empty((T, S))
        self.eps = np.empty((T, S))
        self.L = np.empty((T, S))
        self.fresh = np.zeros((T, S), dtype=bool)
        mu = acc.m + acc.s_mu * rng.standard_normal(S)
        eps = acc.s_eps * rng.standard_normal(S)
        L = sample_ctx(wl.ctx, S, rng)
        life = rng.geometric(1 / wl.mean_life_steps, S)
        for t in range(T):
            life -= 1
            dead = life <= 0
            if dead.any():
                k = int(dead.sum())
                mu[dead] = acc.m + acc.s_mu * rng.standard_normal(k)
                eps[dead] = acc.s_eps * rng.standard_normal(k)
                L[dead] = sample_ctx(wl.ctx, k, rng)
                life[dead] = rng.geometric(1 / wl.mean_life_steps, k)
                self.fresh[t, dead] = True
            self.mu[t], self.eps[t], self.L[t] = mu, eps, L
            eps = acc.rho * eps + math.sqrt(1 - acc.rho**2) * acc.s_eps * rng.standard_normal(S)
            L = L + 3.3  # 上下文按平均产出增长，与策略无关，保证 CRN
        j = np.arange(NEXTN)
        logit = self.mu[..., None] + self.eps[..., None] - acc.kappa * j + acc.s_nu * rng.standard_normal((T, S, NEXTN))
        self.pi = 1 / (1 + np.exp(-logit))
        self.conf = 1 / (1 + np.exp(-(acc.conf_scale * logit + acc.conf_bias + acc.s_eta * rng.standard_normal((T, S, NEXTN)))))
        self.U = rng.random((T, S, NEXTN))
        self.fwd_noise = rng.standard_normal(T)
        self.smp_noise = rng.standard_normal(T)


# ---------------------------------------------------------------- 估时器（移植 mtp_varlen.hpp）


class ForwardRuntimeEstimator:
    """两层中位数：先按 graph size 聚合取中位数，再对这些中位数取中位数。window=5000, min_samples=10。"""

    def __init__(self, window: int = 5000, min_samples: int = 10, refresh: int = 10) -> None:
        self.window, self.min_samples, self.refresh = window, min_samples, refresh
        self.buf: list[tuple[int, float]] = []
        self._cache, self._age = 0.0, refresh

    def update(self, graph: int, value: float) -> None:
        self.buf.append((graph, value))
        if len(self.buf) > self.window:
            self.buf.pop(0)
        self._age += 1

    def get(self) -> float:
        # 仿真提速：每 refresh 步重算一次（原实现每步增量维护，结果相同量级）
        if self._age >= self.refresh:
            self._cache, self._age = self._compute(), 0
        return self._cache

    def _compute(self) -> float:
        by: dict[int, list[float]] = {}
        for g, v in self.buf:
            by.setdefault(g, []).append(v)
        meds = [float(np.median(v)) for v in by.values() if len(v) >= self.min_samples]
        return float(np.median(meds)) if meds else 0.0


class SamplerEstimator:
    def __init__(self) -> None:
        self.by: dict[int, list[float]] = {}
        self.fallback: list[float] = []

    def update(self, n: int, v: float) -> None:
        for lst in (self.by.setdefault(n, []), self.fallback):
            lst.append(v)
            if len(lst) > 9:
                lst.pop(0)

    def get(self, n: int) -> float:
        lst = self.by.get(n, [])
        if len(lst) >= 5:
            return float(np.median(lst))
        return float(np.median(self.fallback)) if len(self.fallback) >= 5 else 0.0


class CtxAwareEstimator:
    """改进估时：forward - B(graph) ≈ θ_G * G + θ_1 * Σ q_i w_i + θ_2 * Σ w_i，w_i = pairs(L_i) / 1e6。

    滑窗岭回归（每 100 步重解一次 3x3 正规方程），θ 截断为非负。特征都是调度器手上已有的量
    （FlopsEstimator 已经按层结构算 pairs）。
    """

    def __init__(self, window: int = 3000, ridge: float = 1e-3) -> None:
        self.window, self.ridge = window, ridge
        self.X: list[np.ndarray] = []
        self.y: list[float] = []
        self.theta: np.ndarray | None = None
        self.n = 0

    def update(self, feats: np.ndarray, resid: float) -> None:
        self.X.append(feats)
        self.y.append(resid)
        if len(self.y) > self.window:
            self.X.pop(0)
            self.y.pop(0)
        self.n += 1
        if self.n % 100 == 0 and len(self.y) >= 200:
            X, y = np.array(self.X), np.array(self.y)
            scale = X.std(0) + 1e-9
            Xs = X / scale
            A = Xs.T @ Xs + self.ridge * len(y) * np.eye(3)
            th = np.linalg.solve(A, Xs.T @ y) / scale
            self.theta = np.maximum(th, 0.0)


class Recalibrator:
    """在线校准 confidence：用每个被验证 draft 的 accept_prob（DeepInfer 已在 finalize 里算出，用于 batch_confidence_error）
    做按位置分桶的保序回归。每个位置 20 个桶，桶内用 EMA 估 accept_prob 的均值，再做累积最大保证单调。"""

    def __init__(self, bins: int = 20, ema: float = 0.01, warm: int = 50) -> None:
        self.bins, self.ema, self.warm = bins, ema, warm
        self.mean = np.tile((np.arange(bins) + 0.5) / bins, (NEXTN, 1))
        self.count = np.zeros((NEXTN, bins))

    def _bin(self, conf: np.ndarray) -> np.ndarray:
        return np.minimum((conf * self.bins).astype(int), self.bins - 1)

    def update(self, conf: np.ndarray, accept_prob: np.ndarray, verified_mask: np.ndarray) -> None:
        b = self._bin(conf)
        for j in range(NEXTN):
            m = verified_mask[:, j]
            if not m.any():
                continue
            bj, pj = b[m, j], accept_prob[m, j]
            sums = np.bincount(bj, weights=pj, minlength=self.bins)
            cnts = np.bincount(bj, minlength=self.bins)
            nz = cnts > 0
            batch_mean = sums[nz] / cnts[nz]
            self.mean[j, nz] += self.ema * cnts[nz] / (1 + self.ema * cnts[nz]) * (batch_mean - self.mean[j, nz])
            self.count[j] += cnts

    def __call__(self, conf: np.ndarray) -> np.ndarray:
        if self.count.sum() < self.warm:
            return conf
        mono = np.maximum.accumulate(self.mean, axis=1)
        b = self._bin(conf)
        return np.take_along_axis(mono[None, :, :].repeat(conf.shape[0], 0), b[..., None], axis=2)[..., 0]


# ---------------------------------------------------------------- 调度策略


@dataclass
class StepView:
    S: int
    L: np.ndarray  # 已按 L 降序（同 scheduler.hpp 先按 get_len 降序排）
    conf_prev: np.ndarray  # [S, NEXTN] 上一轮的 confidence（定 total_q 用）
    conf_cur: np.ndarray  # [S, NEXTN] 当前 draft 的 confidence（GPU 上分配用）
    fresh: np.ndarray  # 新请求：GPU 分配时 confidence 视为 0


def _sorted_queue(key: np.ndarray) -> np.ndarray:
    """按 key 降序、同值按 (seq_idx, pos) 升序；返回扁平下标。"""
    S, N = key.shape
    flat = key.reshape(-1)
    idx = np.arange(S * N)
    # seq_idx 升序作为次关键字；同一请求内 cum 单调不增，pos 自然升序
    return np.lexsort((idx, -flat))


def schedule_fixed(v: StepView, hw: HW, ctx) -> np.ndarray:
    q = np.full(v.S, 1 + NEXTN)
    q[v.fresh] = 1
    return q


def _alloc(v: StepView, best_bsz: int, cost: np.ndarray | None) -> np.ndarray:
    """GPU kernel compute_seqlens_q_by_confidence_rates：在 total_q 固定下按当前 confidence 分配。"""
    cur = np.where(v.fresh[:, None], 0.0, v.conf_cur)
    cum = np.cumprod(cur, axis=1)
    key = cum if cost is None else cum / cost[:, None]
    order = _sorted_queue(key)
    q = np.ones(v.S, dtype=int)
    take = order[: best_bsz - v.S]
    np.add.at(q, take // NEXTN, 1)
    return q


def _best_bsz(v: StepView, hw: HW, baseline, time_fn, key_cost: np.ndarray | None) -> int:
    """移植 MTPVarlen::schedule 的 best_bsz 扫描。time_fn(graph, cur_bsz, extra) -> 估计耗时。"""
    prev = np.where(v.fresh[:, None], 1.0, v.conf_prev)  # 无 rate 且有 draft 时 pad 1.0
    cum = np.cumprod(prev, axis=1)
    key = cum if key_cost is None else cum / key_cost[:, None]
    order = _sorted_queue(key)
    gains = cum.reshape(-1)[order]
    seq_of = order // NEXTN
    w = pairs(v.L) / 1e6
    # 向量化等价于原实现的逐个扫描：候选 k=0..S*NEXTN，严格大于才更新 → 取第一个最大值
    cur_exp = v.S + np.concatenate([[0.0], np.cumsum(gains)])
    cur_bsz = v.S + np.arange(len(order) + 1)
    x1 = float(w.sum()) + np.concatenate([[0.0], np.cumsum(w[seq_of])])
    ok = cur_bsz <= hw.graphs[-1]
    cur_exp, cur_bsz, x1 = cur_exp[ok], cur_bsz[ok], x1[ok]
    g = hw.graphs[np.searchsorted(hw.graphs, cur_bsz)]
    t = time_fn(g, cur_bsz, x1)
    return int(cur_bsz[int(np.argmax(cur_exp / t))])


def schedule_dspark(v: StepView, hw: HW, ctx) -> np.ndarray:
    est_fwd, est_smp, baseline = ctx["fwd"], ctx["smp"], ctx["baseline"]
    slope = max(0.0, est_fwd.get())
    smp = est_smp.get(v.S)

    def time_fn(g, bsz, x1):
        return baseline[g] + slope * g + smp

    return _alloc(v, _best_bsz(v, hw, baseline, time_fn, None), None)


def _ca_time(ctx, v: StepView, hw: HW):
    est, baseline = ctx["ca"], ctx["baseline"]
    smp = ctx["smp"].get(v.S)
    if est.theta is None:
        return None
    thG, th1, th2 = est.theta
    x2 = float((pairs(v.L) / 1e6).sum())

    def time_fn(g, bsz, x1):
        return baseline[g] + thG * bsz + th1 * x1 + th2 * x2 + smp

    cost = thG + th1 * pairs(v.L) / 1e6
    # B(graph) 的局部斜率对所有 token 相同，加进去避免 cost 过小时排序被 ctx 项主导
    G_est = int(v.S * (1 + NEXTN * 0.7))
    gi = np.searchsorted(hw.graphs, G_est)
    lo, hi = hw.graphs[max(gi - 1, 0)], hw.graphs[min(gi, len(hw.graphs) - 1)]
    local_slope = (baseline[hi] - baseline[lo]) / max(hi - lo, 1)
    cost = cost + max(local_slope, 0.0)
    return time_fn, cost


def schedule_ca_time_only(v: StepView, hw: HW, ctx) -> np.ndarray:
    r = _ca_time(ctx, v, hw)
    if r is None:
        return schedule_dspark(v, hw, ctx)
    time_fn, _ = r
    return _alloc(v, _best_bsz(v, hw, ctx["baseline"], time_fn, None), None)


def schedule_ca(v: StepView, hw: HW, ctx) -> np.ndarray:
    r = _ca_time(ctx, v, hw)
    if r is None:
        return schedule_dspark(v, hw, ctx)
    time_fn, cost = r
    return _alloc(v, _best_bsz(v, hw, ctx["baseline"], time_fn, cost), cost)


def schedule_oracle(v: StepView, hw: HW, ctx) -> np.ndarray:
    """上界：真实 pi、真实边际开销、真实估时（同一步内两阶段都用当前真值）。"""
    pi = ctx["pi_t"]
    true_v = replace(v, conf_prev=pi, conf_cur=pi)
    cost = hw.marginal_token_cost(v.L)
    w = pairs(v.L) / 1e6

    def time_fn(g, bsz, x1):
        # 光滑近似：边际项按平均 q 估算 MMA
        per_tok = (6.0 / hw.bw + 8192.0 / hw.fp4) * 1e3 * 1e6
        return (hw.B(g) + hw.imb * bsz + per_tok * x1 + 68.0 / hw.bw * 1e9 * float(w.sum()) + hw.c_seq * v.S)

    gi = np.searchsorted(hw.graphs, int(v.S * 4))
    lo, hi = hw.graphs[max(gi - 1, 0)], hw.graphs[min(gi, len(hw.graphs) - 1)]
    cost = cost + max((hw.B(hi) - hw.B(lo)) / max(hi - lo, 1), 0.0)
    return _alloc(true_v, _best_bsz(true_v, hw, None, time_fn, cost), cost)


POLICIES = {
    "fixed5": schedule_fixed,
    "dspark": schedule_dspark,
    "ca_time": schedule_ca_time_only,
    "ca": schedule_ca,
    "oracle": schedule_oracle,
}


# ---------------------------------------------------------------- 仿真主循环


def simulate(world: World, hw: HW, policy: str, warmup: int) -> dict:
    rng = np.random.default_rng(12345)
    baseline = np.zeros(hw.graphs[-1] + 1)
    baseline[hw.graphs] = hw.B(hw.graphs) * (1 + 0.003 * rng.standard_normal(len(hw.graphs)))  # profile 值带小噪声
    ctx = {"fwd": ForwardRuntimeEstimator(), "smp": SamplerEstimator(), "ca": CtxAwareEstimator(), "baseline": baseline}
    recal = Recalibrator() if policy.endswith("_recal") else None
    fn = POLICIES[policy.removesuffix("_recal")]
    tot_tok = tot_time = tot_ver = 0.0
    n_steps = 0
    prev_conf = world.conf[0]
    for t in range(world.T):
        order = np.argsort(-world.L[t], kind="stable")  # scheduler.hpp：按长度降序
        L = world.L[t][order]
        fresh = world.fresh[t][order]
        cur_conf = world.conf[t][order]
        prev_c = prev_conf[order]
        if recal is not None:
            cur_conf, prev_c = recal(cur_conf), recal(prev_c)
        v = StepView(world.S, L, prev_c, cur_conf, fresh)
        ctx["pi_t"] = world.pi[t][order]
        q = fn(v, hw, ctx)
        G = int(q.sum())
        g = hw.graph_of(G)
        # 接受判定：前 q-1 个 draft 依次接受直到第一次拒绝
        acc_mask = world.U[t][order] < world.pi[t][order]
        ver = q - 1
        lead = np.cumprod(acc_mask, axis=1)
        accepted = (lead * (np.arange(NEXTN) < ver[:, None])).sum(1)
        tokens = float((1 + accepted).sum())
        fwd = (hw.B(g) + hw.imb * G + hw.ctx_cost(L, q).sum() + hw.phi * hw.c_seq * world.S) * (1 + hw.noise * world.fwd_noise[t])
        smp = (1 - hw.phi) * hw.c_seq * world.S * (1 + hw.noise * world.smp_noise[t])
        # 运行时反馈（同 llm_engine.post_process → update_mtp_runtime）
        ctx["fwd"].update(g, (fwd - baseline[g]) / g)
        ctx["smp"].update(world.S, smp)
        w = pairs(L) / 1e6
        ctx["ca"].update(np.array([G, float((q * w).sum()), float(w.sum())]), fwd - baseline[g])
        if recal is not None:
            ver_mask = np.arange(NEXTN) < ver[:, None]
            recal.update(world.conf[t][order], world.pi[t][order], ver_mask)
        prev_conf = world.conf[t]
        if t >= warmup:
            tot_tok += tokens
            tot_time += fwd + smp
            tot_ver += float(ver.sum())
            n_steps += 1
    return {
        "throughput": tot_tok / tot_time * 1e3,  # tokens/s/rank
        "tau": tot_tok / (n_steps * world.S),
        "verified": tot_ver / (n_steps * world.S),
        "step_ms": tot_time / n_steps,
    }


def run_all(wl: Workload, hw: HW, acc: Accept, policies=("fixed5", "dspark", "ca_time", "ca", "oracle")) -> dict:
    world = World(wl, acc)
    return {p: simulate(world, hw, p, wl.warmup) for p in policies}


if __name__ == "__main__":
    import json

    for conc in (16, 60, 96):
        res = run_all(Workload(conc=conc, steps=1500, warmup=300), HW(), Accept(), ("fixed5", "dspark"))
        print(conc, json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in res.items()}))
