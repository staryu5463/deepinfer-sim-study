"""Mooncake 生产 trace 的 KV 复用特征（为 CPU/OS 缓存思路的闭环验证做准备）。

数据：kvcache-ai/Mooncake FAST25-release/traces（Kimi 线上 1 小时采样，512-token 前缀块哈希）。
前缀语义：一个块只有在它前面的所有块都命中时才能复用，所以统计"有效复用"时按前缀匹配计算。
"""

import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    with open(os.path.join(HERE, "data", f"{name}_trace.jsonl")) as f:
        return [json.loads(line) for line in f]


def stats(reqs):
    seen = set()
    first_write_req = {}
    reuse = Counter()  # 块被"前缀命中"复用的次数（无限容量下）
    hit_tok = tot_tok = 0
    by_pos = defaultdict(lambda: [0, 0])  # 块在请求中的位置 -> [写入数, 之后被复用的数]
    for i, r in enumerate(reqs):
        h = r["hash_ids"]
        k = 0
        while k < len(h) and h[k] in seen:
            reuse[h[k]] += 1
            k += 1
        hit_tok += min(k * 512, r["input_length"])
        tot_tok += r["input_length"]
        for j in range(k, len(h)):
            if h[j] not in seen:
                seen.add(h[j])
                first_write_req[h[j]] = (i, j)
    written = len(seen)
    never = sum(1 for b in seen if reuse[b] == 0)
    for b, (i, j) in first_write_req.items():
        pb = min(j, 40) // 4 * 4
        by_pos[pb][0] += 1
        by_pos[pb][1] += reuse[b] > 0
    counts = np.array([reuse[b] for b in seen])
    top10 = np.sort(counts)[::-1][: max(1, len(counts) // 10)].sum() / max(1, counts.sum())
    return {
        "requests": len(reqs),
        "blocks_written": written,
        "one_hit_wonder_fraction": round(never / written, 3),
        "ideal_token_hit_rate": round(hit_tok / tot_tok, 3),
        "top10pct_blocks_share_of_reuses": round(float(top10), 3),
        "reuse_prob_by_block_position": {f"{p}-{p + 3}": round(v[1] / v[0], 3) for p, v in sorted(by_pos.items())},
    }


if __name__ == "__main__":
    out = {n: stats(load(n)) for n in ("conversation", "toolagent")}
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(HERE, "results", "trace_stats.json"), "w"), indent=1)
