"""KV 死块预测（CPU 末级缓存的 dead-block prediction / cache bypass 思路）用于 KV 存储写准入。

在 KV 块刚算出、准备写入存储时，用存储侧已知的特征预测它以后是否会被复用；预测为"死块"就不写。
- 训练：前半段 trace 写入的块（标签 = 之后在 trace 内是否被前缀复用，观察期 >= 30 分钟）。
- 测试：回放整条 trace，模型只用于后半段，只统计后半段请求的命中率与 SSD 写入。
- 前缀语义：某块不写，后续请求在它之后的部分都无法命中（回放自动体现）。
"""

import json
import os
from collections import Counter, OrderedDict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["hit_blocks", "parent_reuse_count", "new_blocks", "rel_pos", "is_last_full", "parent_idle_s", "input_len"]


def load(name):
    with open(os.path.join(HERE, "data", f"{name}_trace.jsonl")) as f:
        return [json.loads(line) for line in f]


def features_and_labels(reqs):
    """按"无限容量存储"的视角提特征（存储侧知道哈希链匹配到哪、父块被复用过几次、上次访问时间）。"""
    seen, reuse, last_access = set(), Counter(), {}
    rows = []  # (req_idx, block, feats)
    for i, r in enumerate(reqs):
        h, t = r["hash_ids"], r["timestamp"] / 1000
        full = min(r["input_length"] // 512, len(h))
        k = 0
        while k < full and h[k] in seen:
            reuse[h[k]] += 1
            k += 1
        parent = h[k - 1] if k > 0 else None
        parent_reuse = reuse[parent] if parent is not None else -1
        parent_idle = t - last_access.get(parent, t) if parent is not None else -1
        n_new = full - k
        for j in range(k, full):
            if h[j] not in seen:
                rows.append((i, h[j], [k, parent_reuse, n_new, (j - k) / max(n_new, 1), int(j == full - 1), parent_idle, r["input_length"]]))
        for j in range(full):
            seen.add(h[j])
            last_access[h[j]] = t
    labels = {b: int(reuse[b] > 0) for _, b, _ in rows}
    return rows, labels


def replay(reqs, admit, cap, start_eval):
    """LRU 存储回放；admit(i, b) 决定新块是否写入。只统计 start_eval 之后的请求。"""
    cache = OrderedDict()
    hit = tot = writes = 0
    for i, r in enumerate(reqs):
        h = r["hash_ids"]
        full = min(r["input_length"] // 512, len(h))
        k = 0
        while k < full and h[k] in cache:
            cache.move_to_end(h[k])
            k += 1
        if i >= start_eval:
            hit += k * 512
            tot += r["input_length"]
        for j in range(k, full):
            if h[j] in cache or not admit(i, h[j]):
                continue
            cache[h[j]] = True
            if i >= start_eval:
                writes += 1
            if len(cache) > cap:
                cache.popitem(last=False)
    return hit / tot, writes


if __name__ == "__main__":
    out = {}
    for name in ("conversation", "toolagent"):
        reqs = load(name)
        half = next(i for i, r in enumerate(reqs) if r["timestamp"] >= reqs[-1]["timestamp"] / 2)
        rows, labels = features_and_labels(reqs)
        X = np.array([f for _, _, f in rows], dtype=float)
        y = np.array([labels[b] for _, b, _ in rows])
        is_train = np.array([i < half for i, _, _ in rows])
        model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0).fit(X[is_train], y[is_train])
        p = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y[~is_train], p[~is_train])
        prob = {(i, b): p[n] for n, (i, b, _) in enumerate(rows)}
        U = len({b for _, b, _ in rows})
        res = {"test_auc": round(float(auc), 3), "test_base_reuse_rate": round(float(y[~is_train].mean()), 3), "rows": []}
        for cap_frac in (0.05, 0.2, 1.0):
            cap = int(cap_frac * U)
            base_hit, base_w = replay(reqs, lambda i, b: True, cap, half)
            for tau in (0.02, 0.05, 0.1, 0.2):
                hit, w = replay(reqs, lambda i, b: i < half or prob.get((i, b), 1.0) >= tau, cap, half)
                res["rows"].append({"cap_%": cap_frac * 100, "tau": tau, "hit_admit_all": round(base_hit, 4), "hit_bypass": round(hit, 4),
                                    "hit_delta_pts": round(100 * (hit - base_hit), 2), "ssd_write_reduction_%": round(100 * (1 - w / base_w), 1)})
                print(name, res["rows"][-1], flush=True)
        print(name, "test AUC", res["test_auc"], "base reuse rate", res["test_base_reuse_rate"])
        out[name] = res
    json.dump(out, open(os.path.join(HERE, "results", "deadblock.json"), "w"), indent=1)
