"""Prefill 调度（操作系统进程调度思路）的闭环验证：Mooncake 真实到达时间与长度回放。

prefill 的工作量（未命中前缀的 token 数）开始前就精确已知 —— 等价于"已知 CPU 突发时间"，
SRPT 对平均响应时间最优（Schrage 1968），但长任务可能饥饿，OS 用老化(aging)补偿。

模型：M 个 prefill 引擎；chunked prefill，每处理完一个 chunk 可以重新选（即 chunk 粒度可抢占）；
已开始的请求留在原引擎（部分 KV 在本地）。工作量 = 未命中 token 数，代价按 DeepSeek 文档的
二次型 cost(q, c) = q + beta * q * (c + q / 2)（c 为已算前缀长度），按目标利用率缩放引擎速度。

策略：
    fifo      先来先服务（DeepInfer 开源调度器的引擎内行为）
    tier      DeepSeek PLB 风格：按总上下文长度分 4 档（<16K/16-64K/64-256K/>=256K），引擎按各档需求占比分配，档内 FIFO
    srpt      剩余工作最少优先（chunk 粒度抢占）
    srpt_age  SRPT + 老化：优先级 = 剩余工作 - w * 等待时间（w 让等待 1 秒抵消 w 秒的工作）
"""

import heapq
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BETA = 7e-7  # 每 token 对每个上下文 token 的二次项系数（索引器 ~20 KFLOP / 线性部分 ~30 GFLOP）
CHUNK = 8192


def load(name):
    with open(os.path.join(HERE, "data", f"{name}_trace.jsonl")) as f:
        reqs = [json.loads(line) for line in f]
    seen = set()
    jobs = []
    for r in reqs:
        h = r["hash_ids"]
        full = min(r["input_length"] // 512, len(h))
        k = 0
        while k < full and h[k] in seen:
            k += 1
        seen.update(h[:full])
        c = k * 512
        q = r["input_length"] - c
        work = q + BETA * q * (c + q / 2)
        jobs.append(dict(t=r["timestamp"] / 1000, work=work, ctx=r["input_length"]))
    return jobs


def tier_of(ctx):
    return 0 if ctx < 16384 else 1 if ctx < 65536 else 2 if ctx < 262144 else 3


def simulate(jobs, M, speed, policy, age_w=0.0):
    """堆实现。优先级键与时间无关：fifo=到达时间，srpt=剩余工作，
    srpt_age=剩余工作 + w*speed*到达时间（等价于 剩余工作 - w*speed*等待时间 再减去公共常数，类似 CFS 的 vruntime）。"""
    n = len(jobs)
    rem = np.array([j["work"] for j in jobs])
    arrive = np.array([j["t"] for j in jobs])
    done = np.full(n, np.nan)

    def key(i):
        if policy in ("fifo", "tier"):
            return arrive[i]
        if policy == "srpt":
            return rem[i]
        return rem[i] + age_w * speed * arrive[i]

    if policy == "tier":
        demand = np.bincount([tier_of(j["ctx"]) for j in jobs], weights=rem, minlength=4)
        share = demand / demand.sum()
        alloc = (np.maximum(1, np.round(share * M)) * (share > 0)).astype(int)
        while alloc.sum() > M:
            alloc[np.argmax(alloc)] -= 1
        server_tier = list(np.repeat(np.arange(4), alloc))
        server_tier += [int(np.argmax(share))] * (M - len(server_tier))
        n_queues = 4
        queue_of_job = [tier_of(j["ctx"]) for j in jobs]
    else:
        server_tier = [0] * M
        n_queues = 1
        queue_of_job = [0] * n
    waiting = [[] for _ in range(n_queues)]  # 未开始的请求（堆）
    owned = [[] for _ in range(M)]  # 已在该引擎开始、被抢占或待续的请求（堆）
    events, free = [], list(range(M))
    nxt = 0

    def pick(s):
        w = waiting[server_tier[s]]
        o = owned[s]
        if not w and not o:
            return None
        if o and (not w or o[0][0] <= w[0][0]):
            return heapq.heappop(o)[1]
        return heapq.heappop(w)[1]

    def dispatch(now):
        for s in list(free):
            i = pick(s)
            if i is None:
                continue
            work = min(rem[i], CHUNK)
            rem[i] -= work
            free.remove(s)
            heapq.heappush(events, (now + work / speed, s, i))

    while nxt < n or events:
        t_arr = arrive[nxt] if nxt < n else np.inf
        t_ev = events[0][0] if events else np.inf
        if t_arr <= t_ev:
            t = t_arr
            heapq.heappush(waiting[queue_of_job[nxt]], (key(nxt), nxt))
            nxt += 1
        else:
            t, s, i = heapq.heappop(events)
            if rem[i] <= 1e-9:
                done[i] = t
            else:
                heapq.heappush(owned[s], (key(i), i))
            free.append(s)
        dispatch(t)
    ttft = done - arrive
    works = np.array([j["work"] for j in jobs])
    big = works >= np.quantile(works, 0.99)
    return {"mean": float(np.mean(ttft)), "p50": float(np.percentile(ttft, 50)), "p90": float(np.percentile(ttft, 90)),
            "p99": float(np.percentile(ttft, 99)), "largest1%_mean": float(np.mean(ttft[big]))}


if __name__ == "__main__":
    out = {}
    M = 8
    for name in ("conversation", "toolagent"):
        jobs = load(name)
        span = jobs[-1]["t"]
        total = sum(j["work"] for j in jobs)
        out[name] = {}
        for util in (0.7, 0.85, 0.95):
            speed = total / span / (M * util)  # token/s 每引擎，使平均利用率 = util
            res = {p: simulate(jobs, M, speed, p) for p in ("fifo", "tier", "srpt")}
            res["srpt_age"] = simulate(jobs, M, speed, "srpt_age", age_w=0.5)
            base = res["fifo"]
            row = {p: {k: round(v / base[k], 3) for k, v in r.items()} for p, r in res.items()}
            row["fifo_abs_s"] = {k: round(v, 2) for k, v in base.items()}
            out[name][f"util{util}"] = row
            print(name, util, json.dumps(row), flush=True)
    json.dump(out, open(os.path.join(HERE, "results", "prefill_sched.json"), "w"), indent=1)
