"""KV 前缀缓存的写准入/分代实验（CPU、CDN 缓存思路的闭环验证），用 Mooncake 生产 trace 回放。

现状（DeepInfer）：算出的 KV 块全部写进 3FS（SSD），存储侧 GC 用 LRU 回收。
借鉴：S3-FIFO（SOSP'23）/ 分代 GC 假说——新块先进小的 DRAM 试用队列（不写 SSD），
     被再次访问过才在离开试用队列时晋升到 SSD 主区；没被访问就丢弃，只留哈希作为"幽灵"，
     幽灵再次出现时直接写入主区。
公平性：LRU 基线拿到相同的总容量（SSD + DRAM），且两者都只存整块（尾部不满 512 token 的块不存）。
前缀语义：命中长度 = 从头开始连续命中的块数；一个块只有前面所有块都在时才有用。
"""

import json
import os
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    with open(os.path.join(HERE, "data", f"{name}_trace.jsonl")) as f:
        return [json.loads(line) for line in f]


class LRU:
    def __init__(self, cap):
        self.cap, self.d, self.ssd_writes = cap, OrderedDict(), 0

    def has(self, b):
        return b in self.d

    def touch(self, b):
        self.d.move_to_end(b)

    def insert(self, b):
        if b in self.d:
            return
        self.ssd_writes += 1
        self.d[b] = True
        if len(self.d) > self.cap:
            self.d.popitem(last=False)


class Generational:
    """DRAM 试用 FIFO（small）+ SSD 主区 LRU（main）+ 幽灵集合（只存哈希，容量 = main 容量）。"""

    def __init__(self, ssd_cap, dram_cap):
        self.main_cap, self.small_cap = ssd_cap, dram_cap
        self.main, self.small, self.ghost = OrderedDict(), OrderedDict(), OrderedDict()
        self.ssd_writes = 0

    def has(self, b):
        return b in self.main or b in self.small

    def touch(self, b):
        if b in self.main:
            self.main.move_to_end(b)
        else:
            self.small[b] = 1  # 记一次访问，离开试用队列时据此晋升

    def _to_main(self, b):
        self.ssd_writes += 1
        self.main[b] = True
        if len(self.main) > self.main_cap:
            self.main.popitem(last=False)

    def insert(self, b):
        if self.has(b):
            return
        if b in self.ghost:
            del self.ghost[b]
            self._to_main(b)
            return
        self.small[b] = 0
        while len(self.small) > self.small_cap:
            old, freq = self.small.popitem(last=False)
            if freq > 0:
                self._to_main(old)
            else:
                self.ghost[old] = True
                if len(self.ghost) > self.main_cap:
                    self.ghost.popitem(last=False)


def replay(reqs, cache):
    hit = tot = 0
    for r in reqs:
        h = r["hash_ids"]
        full = min(r["input_length"] // 512, len(h))
        k = 0
        while k < full and cache.has(h[k]):
            cache.touch(h[k])
            k += 1
        hit += k * 512
        tot += r["input_length"]
        for j in range(k, full):
            cache.insert(h[j])
    return hit / tot, cache.ssd_writes


def unique_full_blocks(reqs):
    s = set()
    for r in reqs:
        s.update(r["hash_ids"][: r["input_length"] // 512])
    return len(s)


if __name__ == "__main__":
    out = {}
    for name in ("conversation", "toolagent"):
        reqs = load(name)
        U = unique_full_blocks(reqs)
        rows = []
        for ssd_frac in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
            for dram_frac in (0.01, 0.03):
                C, D = int(ssd_frac * U), int(dram_frac * U)
                lru_hit, lru_w = replay(reqs, LRU(C + D))
                gen_hit, gen_w = replay(reqs, Generational(C, D))
                rows.append({
                    "ssd_cap_%": ssd_frac * 100, "dram_cap_%": dram_frac * 100,
                    "lru_hit": round(lru_hit, 4), "gen_hit": round(gen_hit, 4),
                    "hit_delta_pts": round(100 * (gen_hit - lru_hit), 2),
                    "ssd_write_reduction_%": round(100 * (1 - gen_w / lru_w), 1),
                })
                print(name, rows[-1], flush=True)
        out[name] = {"unique_full_blocks": U, "rows": rows}
    json.dump(out, open(os.path.join(HERE, "results", "kv_cache_sim.json"), "w"), indent=1)
