"""痛点经济账：只用 DeepSeek 公开的线上数字（知乎《DeepSeek V4.1 Flash 大规模推理部署实践》、
DeepInfer doc/inference_deployment_qa.md），推出每个痛点值多少卡、多少钱。所有推导都在这里，可复核。
"""

import json
import os

# ---- DeepSeek 公开的线上实测（知乎博客"实际线上运行数据"表 + 正文） ----
PREFILL_TPS_ALL = 1.5e6  # 每卡吞吐（含缓存命中）
PREFILL_TPS_MISS = 60e3  # 每卡吞吐（不含缓存命中）
HIT = 0.96  # 平均输入缓存命中率
AVG_INPUT = 77e3  # 平均输入长度
DECODE_TPS = 14e3  # decode 每卡吞吐
USER_TPS = 230  # 每用户吐字速度
TOKENS_PER_DAY = 155e12  # 单一集群日吞吐
REV_PER_GPU_H = 38.0  # 美元 / 卡 / 小时
PD_RATIO = 1.0  # P:D 接近 1:1
LB_EFFICIENCY = 0.80  # "大约能达到完美均衡离线基准的 80%"
TTL_HIT_LOSS = (0.005, 0.01)  # 3 天 TTL 让 chatbot 命中率下降 0.5%-1%
KV_BYTES_PER_TOKEN = 112 * 1024 / 128  # 一个 128-token block 约 112 KiB

# ---- 公开 GPU 故障率（Meta Llama 3 论文：54 天 16,384 张 H100 上 419 次意外中断） ----
FAIL_PER_GPU_DAY = 419 / (16384 * 54)
RESTART_MIN = (3, 10)  # "错误重启在分钟量级"
DECODE_INSTANCE = 128  # EP64~128

out = {}
out["consistency_check: 1.5M x (1-96%) vs 60K"] = PREFILL_TPS_ALL * (1 - HIT)

n_prefill = TOKENS_PER_DAY / 86400 / PREFILL_TPS_ALL  # 若 155T/天 主要是输入 token
n_decode = n_prefill / PD_RATIO
n_total = n_prefill + n_decode
out_tokens_day = n_decode * DECODE_TPS * 86400
out["fleet_estimate"] = {
    "prefill_gpus": round(n_prefill), "decode_gpus": round(n_decode), "total_gpus": round(n_total),
    "input_output_ratio": round(TOKENS_PER_DAY / out_tokens_day), "requests_per_s": round(TOKENS_PER_DAY / 86400 / AVG_INPUT),
    "avg_output_tokens_per_request": round(out_tokens_day / (TOKENS_PER_DAY / AVG_INPUT)),
    "decode_concurrency_per_gpu": round(DECODE_TPS / USER_TPS),
    "revenue_per_day_usd": round(n_total * 24 * REV_PER_GPU_H),
}
day_rev = n_total * 24 * REV_PER_GPU_H

# 痛点 1：命中率杠杆。prefill 卡数 ∝ 未命中比例
miss = 1 - HIT
lever = {}
for dh in (0.001, 0.005, 0.01):
    saved_prefill = dh / miss
    lever[f"+{dh * 100:.1f}pt hit"] = {"prefill_compute_saved_%": round(100 * saved_prefill, 1),
                                       "fleet_saved_%": round(100 * saved_prefill * n_prefill / n_total, 1),
                                       "usd_per_day": round(saved_prefill * n_prefill * 24 * REV_PER_GPU_H)}
out["pain1_hit_rate_leverage"] = lever
out["pain1_ttl_loss_chatbot"] = {f"{x * 100:.1f}pt": f"chatbot prefill compute +{100 * x / miss:.0f}%" for x in TTL_HIT_LOSS}

# 痛点 2：负载均衡损失（官方自述 ~80%）
out["pain2_load_balance"] = {"throughput_lost_vs_balanced_%": round(100 * (1 - LB_EFFICIENCY)),
                             "gpus_equivalent_if_fully_recovered": round(n_total * (1 / LB_EFFICIENCY - 1)),
                             "usd_per_day_if_fully_recovered": round(day_rev * (1 / LB_EFFICIENCY - 1))}

# 痛点 3：大 EP 故障爆炸半径
fails = n_decode * FAIL_PER_GPU_DAY
lost = [fails * DECODE_INSTANCE * m for m in RESTART_MIN]
out["pain3_fault_blast_radius"] = {"decode_gpu_failures_per_day": round(fails, 2),
                                   "gpu_minutes_lost_per_day": [round(x) for x in lost],
                                   "share_of_decode_capacity_%": [round(100 * x / (n_decode * 1440), 3) for x in lost],
                                   "requests_interrupted_per_failure": round(DECODE_INSTANCE * DECODE_TPS / USER_TPS)}

# 痛点 4：结构性重算（每个请求最少 134 token 不能命中；128 对齐平均多算 64；decoder 有界回放 128 token 过后 20 层）
new_tokens = AVG_INPUT * miss
structural = 64 + 128 * 0.5  # 对齐损失 + decoder 回放（只过一半层，折半）；134 上限通常落在新内容里，不额外计
out["pain4_structural_recompute"] = {"new_tokens_per_request": round(new_tokens),
                                     "structural_token_equiv": structural,
                                     "share_of_prefill_compute_%": round(100 * structural / new_tokens, 1)}

# 痛点 5：存储
out["pain5_storage"] = {"kv_bytes_per_token": KV_BYTES_PER_TOKEN,
                        "kv_read_GBps_per_prefill_gpu": round(PREFILL_TPS_ALL * HIT * KV_BYTES_PER_TOKEN / 1e9, 2),
                        "kv_written_TB_per_day_from_misses_and_outputs": round((TOKENS_PER_DAY * miss + out_tokens_day) * KV_BYTES_PER_TOKEN / 1e12)}

print(json.dumps(out, indent=1, ensure_ascii=False))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "pain_econ.json"), "w"), indent=1, ensure_ascii=False)
