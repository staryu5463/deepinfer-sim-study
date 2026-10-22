"""decode 每步时间拆解：用 DeepSeek 自己的帕累托曲线（doc/images/decode_pareto.png、wide_ep_decode_pareto.png）
+ V4.1 配置算出的每步权重读取量，把"固定开销"拆成 权重读取 / 其余（胶水：kernel 边界、同步、通信延迟等）。

曲线条件：context 131,072，nextn=3 固定，τ = 3.25（每请求每步验证 4 个 token）。每步时间 = τ / 每用户速度。
交叉验证：EP8 与 EP64 的小 batch 时间差，应该等于两者每卡多读的路由专家权重 / HBM 带宽。
"""

import json
import os

import numpy as np

TAU = 3.25
# (每 DP rank 请求数, 每用户 tok/s)，从两张官方图上读出
EP64 = [(8, 325), (16, 315), (32, 297), (64, 246), (128, 178), (256, 103)]
EP8 = [(8, 238), (16, 230), (32, 220), (64, 197), (128, 150), (256, 98)]

# ---- V4.1 每步权重读取（v41_config.json）----
H, QR, HEADS, HD, OR, OG = 5120, 1280, 64, 512, 1024, 8
IDX_H, IDX_D, EXPERTS, INTER, LAYERS, VOCAB = 32, 128, 384, 2304, 40, 129280
attn_params = (H * QR + H * HD  # q_a + kv_swa
               + QR * (HEADS * HD + IDX_H * IDX_D)  # q_b（含 indexer q）
               + OG * OR * (HEADS // OG * HD)  # Wv
               + OG * OR * H)  # out_proj
shared_expert = 3 * H * INTER
gate_bytes = H * EXPERTS * 2  # bf16
per_layer_replicated = attn_params + shared_expert + gate_bytes  # FP8 按 1 字节/参数
expert_bytes = 3 * H * INTER * 0.5 * (1 + 1 / 32)  # FP4 + ue8m0 scale
lm_head = VOCAB * H * 2
draft = 0.6e9  # DFlash 3 层（粗估）


def weight_bytes(ep, extra_per_rank):
    local = (EXPERTS + extra_per_rank * ep + (-(EXPERTS + extra_per_rank * ep) % ep)) // ep
    return LAYERS * (per_layer_replicated + local * expert_bytes) + lm_head + draft, local


def steps(curve):
    n = np.array([c[0] for c in curve], dtype=float)
    t = TAU / np.array([c[1] for c in curve]) * 1e3  # ms
    return n, t


if __name__ == "__main__":
    out = {}
    w64, l64 = weight_bytes(64, 1)
    w8, l8 = weight_bytes(8, 4)  # 官方图注：EP8 为 384+32 个专家，每卡 4 个冗余
    n64, t64 = steps(EP64)
    n8, t8 = steps(EP8)
    gap_ms = float(t8[0] - t64[0])
    implied_bw = (w8 - w64) / (gap_ms * 1e-3)
    out["weights_per_step_GB"] = {"EP64": round(w64 / 1e9, 2), "EP8": round(w8 / 1e9, 2), "local_experts": [l64, l8]}
    out["cross_check"] = {"small_batch_gap_ms(EP8-EP64)": round(gap_ms, 2),
                          "implied_HBM_bandwidth_TBps": round(implied_bw / 1e12, 2),
                          "note": "B200 HBM3e 峰值约 8 TB/s：时间差几乎完全被多读的专家权重解释"}
    bw = min(implied_bw, 8e12)
    mem64 = w64 / bw * 1e3
    small = float(t64[0])
    out["EP64_small_batch_step_decomposition_ms"] = {"step(8 req/rank)": round(small, 2), "weight_reads": round(mem64, 2),
                                                     "glue(other)": round(small - mem64, 2),
                                                     "glue_share_%": round(100 * (small - mem64) / small, 1)}
    # 在线工作点：每卡约 61 请求
    t61 = float(np.interp(61, n64, t64))
    glue = small - mem64
    out["at_61_req_per_rank"] = {"step_ms": round(t61, 2), "glue_ms": round(glue, 2), "glue_share_%": round(100 * glue / t61, 1)}

    # ---- 架构方案的上界投影：在每用户速度 230 tok/s 时，每卡吞吐能到多少 ----
    var = t64 - small  # 随 batch 增长的部分（计算 + KV 读取），视为与胶水无关
    def throughput_at_speed(glue_cut=0.0, extra_mem_ms=0.0, overlap=False, target=230):
        ns = np.linspace(8, 256, 497)
        v = np.interp(ns, n64, var)
        if overlap:  # 两个 micro-batch：胶水与对方的计算/访存重叠，权重要多读一遍
            t = np.maximum(glue * (1 - glue_cut), 2 * mem64 + v) + 0.5
        else:
            t = glue * (1 - glue_cut) + mem64 + extra_mem_ms + v
        speed = TAU / t * 1e3
        ok = speed >= target
        if not ok.any():
            return None
        i = int(np.nonzero(ok)[0][-1])
        return {"req_per_rank": round(float(ns[i])), "tok_s_per_gpu": round(float(ns[i] * speed[i])), "step_ms": round(float(t[i]), 2)}

    base = throughput_at_speed()
    proj = {"baseline(today)": base}
    for cut in (0.25, 0.5, 0.75):
        proj[f"fusion: glue -{int(cut * 100)}%"] = throughput_at_speed(glue_cut=cut)
    proj["two-batch overlap (upper bound)"] = throughput_at_speed(overlap=True)
    proj["two-batch overlap + glue -50%"] = throughput_at_speed(glue_cut=0.5, overlap=True)
    for k, v in proj.items():
        if v and k != "baseline(today)":
            v["gain_vs_today_%"] = round(100 * (v["tok_s_per_gpu"] / base["tok_s_per_gpu"] - 1))
    out["projection_at_230_tok_s_per_user"] = proj
    print(json.dumps(out, indent=1, ensure_ascii=False))
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "decode_glue.json"), "w"), indent=1, ensure_ascii=False)
