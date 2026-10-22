# deepinfer-sim-study

A simulation study of optimization opportunities in [DeepSeek DeepInfer](https://github.com/deepseek-ai/DeepInfer), with verifiers that check the simulators against DeepInfer's own code and published numbers.

> **Disclaimer.** All results come from simulators, not from real SM100 hardware. The simulators are calibrated against the numbers DeepSeek published in [`doc/inference_deployment_qa.md`](https://github.com/deepseek-ai/DeepInfer/blob/main/doc/inference_deployment_qa.md), and routing statistics are synthetic. Treat every number as a hypothesis to test on real hardware. Not affiliated with DeepSeek.

## Main result: speculative-decoding verification length as a cross-rank load balancer

The [DSpark paper](https://arxiv.org/abs/2607.05147) assumes, in a footnote, that engine throughput depends mainly on batch size because "decode load balancers keep both request counts and total context lengths roughly balanced across data-parallel (DP) ranks". This study finds that the assumption does not hold under wide expert parallelism (EP):

1. **Per-rank decisions under global synchronization.** Each DP rank runs its own DSpark scheduler, but EP synchronizes every MoE layer on the slowest rank. The runtime feedback (`forward_ms`) is the *global* step time including waiting, while the profiled baseline assumes all ranks run the same graph size. As a result, lightly loaded ranks see a large residual and verify *fewer* drafts, and the straggler verifies *more*. Even with perfectly balanced load, DSpark spreads per-rank token counts by 3.2% (vs 0.4% with fixed length).
2. **Fix: water-fill to a shared attention-time level.** Each step, ranks all-reduce 2 scalars and fill drafts up to a common level H, updated by the Dinkelbach condition dE/dH = λ·dT/dH. This reaches 85–100% of a centralized upper bound.
3. **Verification length becomes a free knob for absorbing DP-rank imbalance**, which request-level load balancing cannot fix within a step.

| Setting (simulated) | Throughput vs DSpark | TPOT vs DSpark |
|---|---|---|
| EP64, perfectly balanced | +0.8% to +1.2% | −0.8% |
| EP64, ±10% request-count imbalance | **+3.0%** (recovers ~92% of the imbalance loss) | **−2.9%** |
| EP64, long-context mix, ±10% | **+4.1%** | — |
| EP8, ±10% | +1.4% | −1.4% |
| model error: assumed attention share off by ±33% | +2.8% (vs +3.0% exact) | — |

Gains grow with EP size (straggler cost is a max over ranks), and latency improves together with throughput. We found no prior work coordinating speculative verification budgets across DP ranks under EP; SGLang sets draft parameters globally and statically.

## Other findings

| Finding | Status |
|---|---|
| Online recalibration of the DSpark confidence head using `accept_prob`, which DeepInfer already computes per verified draft (only logged as `batch_confidence_error`). DSpark's Sequential Temperature Scaling is offline. | Insurance, not a demonstrated gain: the published table shows no evidence of miscalibration (fitted bias ≈ 0). If live traffic drifts, it recovers up to 6 points of DSpark's gain at high concurrency, and costs nothing when calibrated. Related: [AdaFlash](https://arxiv.org/abs/2607.19223) updates a length head online. |
| EPLB `moe_lb` 1.09 ≈ 1.06 per-step sampling noise + 0.03 statistics lag; per-step replica routing plus more redundancy reaches 1.004 | **Direction already known**: DeepSeek's own LPLB, [SGLang Waterfill/LPLB](https://www.lmsys.org/blog/2026-06-26-waterfill-lplb/), [UltraEP](https://arxiv.org/html/2606.04101). Our contribution is the quantitative decomposition. |
| Weight migration costs ~12 ms per rearrangement, so a shorter interval is nearly free (`moe_lb` 1.087 → 1.071 at 2500 steps) | Config-only change |
| `csrc/engine/scheduler.hpp`: preemption comment says "shortest output", code preempts the longest | Code fact |

**Rejected by the verifier**: context-cost-aware draft ranking (upper bound only +0.3%), greedy dynamic routing (worse than baseline), local search on top of EPLB (better in-sample, no out-of-sample gain, i.e. overfitting), confidence miscalibration as the explanation for DSpark's −1.5% at concurrency 16 (not supported). DeepSeek's step-aware expert placement was independently validated.

## Reproduce

```bash
git clone https://github.com/deepseek-ai/DeepInfer      # the EPLB simulator imports its real policy.py
git clone https://github.com/staryu5463/deepinfer-sim-study
cd deepinfer-sim-study
pip install -r requirements.txt
export DEEPINFER_SRC=../DeepInfer                        # default is ../DeepInfer
python test_port.py                                      # L0: port matches DeepInfer source
python calibrate_dspark.py                               # L1: calibrate on published table, check holdout
python experiments_dspark.py                             # DSpark single-rank A/B
python mr_experiments.py                                 # multi-rank coordination A/B
python eplb_interval.py results/eplb_interval.json       # EPLB interval vs dynamic routing
python eplb_ls_redundancy.py                             # EPLB redundancy x routing
python mr_latency_scale.py                               # coordination: latency and EP size
python calibrate_dspark_v2.py                            # does confidence miscalibration explain the data?
```

A CUDA GPU is recommended for the EPLB simulator (token-level routing runs in torch); everything else runs on CPU. Results from our runs are in `results/`.

| File | Purpose |
|---|---|
| `dspark_sim.py` | Single-rank DSpark simulator with a line-by-line port of `MTPVarlen::schedule` |
| `mr_sim.py` | Multi-rank (EP-coupled) DSpark simulator and the coordinated policy |
| `eplb_sim.py` | Token-level MoE routing simulator driving DeepInfer's real `rebalance_experts_global` |
| `test_port.py` | Port-correctness tests against DeepInfer's reference implementation |
| `calibrate_dspark.py` | Calibration with fit/holdout split |

`test_port.py` contains a short excerpt of DeepInfer code (Apache-2.0); this repository is released under the same license.

---

# 中文报告

本机只有一张 RTX 4070（8GB），跑不了 DeepInfer 本体（它需要 SM100 多卡），所以所有结论都来自仿真器。仿真器的参数用 DeepSeek 公开的数据校准过（DeepInfer 仓库里的 `doc/inference_deployment_qa.md`），但仍然只是仿真结果。

## 结论总表（按价值排序）

| # | 发现 | 位置 | 仿真收益 | 状态 |
|---|---|---|---|---|
| 1 | DSpark 各 DP rank 各自决策，没有考虑 EP 每层都要等最慢的 rank；而且估时器用全局耗时去拟合本 rank 的斜率，反而放大了 rank 间的差距 | `csrc/engine/mtp_varlen.hpp`、`deepinfer/worker/model_runner.py:562` | **+1.2% ~ +4.1%**；rank 间不均衡造成的损失能追回 65% ~ 92% | 已验证，含鲁棒性测试 |
| 2 | EPLB 动态副本分流（按实时负载分配副本，坐标下降可以达到 LP 最优解）+ 每卡多放冗余 expert | DeepGEMM `sm100_mega_gate.cuh:205` | moe_lb 1.087 → 1.052（每卡 1 个冗余）→ **1.004**（每卡 3 个冗余） | 已验证，但通信开销和多读的权重没有计入 |
| 3 | EPLB 缩短重排间隔：搬运权重的代价几乎为零 | 环境变量 `DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL` | moe_lb 1.087 → 1.078（5000 步）/ 1.071（2500 步） | 已验证；只需改配置 |
| 4 | decode 抢占的注释写"抢 output 最短的"，代码实际抢的是 output 最长的（CE 上被抢占的请求会直接终止） | `csrc/engine/scheduler.hpp:557` vs `624-633` | — | 代码事实，影响用户体验 |
| 5 | DSpark 排队时同值的情况下，最长的请求优先拿到验证名额（先按长度降序排，再取小下标） | `scheduler.hpp:648`、`mtp_varlen.hpp:211` | 可以忽略 | 代码事实 |

## 被验证器否定的想法（同样重要）

| 想法 | 结果 |
|---|---|
| DSpark 按"每个 token 的代价"排序（长上下文的 token 更贵） | 理论上界只有 +0.1% ~ +0.3%，不值得做 |
| EPLB 单遍贪心动态分流 | **比原方案更差**（目光短浅，热门 rank 被反复塞满） |
| 在 EPLB 原 policy 的基础上再做局部搜索 | 在历史窗口上目标从 1.057 降到 1.050，但对未来的步数毫无改善，属于**过拟合**，原方案的贪心程度刚刚好 |
| 零通信的 round-robin 分流 | 只降低 0.3 个点，可以忽略 |
| 迁移最小化重标号（同一划分，换卡号） | 搬运量只从 98.7% 降到 79.9%，且搬运本身就便宜，意义不大 |

## 验证器分层

| 层 | 内容 | 结果 |
|---|---|---|
| L0 移植正确性（`test_port.py`） | 和仓库参考实现、C++ 逐行直译比对 | 860 个用例全部一致 |
| L1 仿真器可信度（`calibrate_dspark.py`） | 用公开表格中 3 行拟合，再预测另外 3 行 | 留出集收益误差约 1 个百分点；conc=16 的 −1.5% 没复现 |
| L2 A/B | 所有策略共享随机数，多 seed，给出区间，并和上帝视角上界对比 | 见上表 |
| L3 防自欺 | 改进策略只用真实拿得到的信息；故意给模型参数加误差 | 协调策略在模型估错 ±33% 时，收益从 2.99% 降到 2.76% ~ 2.84%，仍然稳定 |

## 发现 1 详解：跨 rank 协调验证（`mr_sim.py`）

- 事实一：`profile_decode_perf` 测基线时，是 64 个 rank 同时跑同样大小的 graph。
- 事实二：运行时每个 rank 喂给估时器的 `forward_ms`，是包含等待其他 rank 时间的全局耗时。
- 后果：token 少的 rank 算出来的残差大，以为自己的 token 贵，于是验证得更少；最慢的 rank 反而验证得更多。即使负载完全对称，DSpark 也会让各 rank 的 token 数差出 3.2%（固定验证 5 个时只差 0.4%）。
- 改法：每步只 all-reduce 两个标量（上一步的全局吞吐 λ，以及各 rank 在公共水位 H 处的边际价值），H 按 Dinkelbach 条件 dE/dH = λ·dT/dH 来调整，各 rank 把 token 填到 H 为止。实测达到中心化上界的 85% ~ 100%。

| 场景（64 rank） | 相对 DSpark 的收益 |
|---|---|
| 对称 32K benchmark | +1.24% |
| agent 上下文，各 rank 请求数 ±10% | +2.99% |
| 长上下文混合，±10% | +4.10% |
| attention 占比 0.3 / 0.6（敏感性） | +1.97% / +3.92% |

| 各 rank 请求数的不均衡程度 | DSpark | 协调版 |
|---|---|---|
| 0 | 基准 | +0.8% |
| ±10% | −3.1% | −0.2% |
| ±20% | −6.1% | −2.2% |

主要假设：EP 每层都要同步（step ≈ 最慢 rank 的 attention + MoE），attention 占 per-token 开销的 45%（敏感性测了 0.3 ~ 0.6）。

## 发现 2、3 详解：EPLB（`eplb_sim.py`，直接调用 DeepInfer 的 `policy.py`）

- 实测 moe_lb 1.09 可以拆成两部分：约 1.06 是每一步的随机噪声，任何静态摆放都消不掉；约 0.03 是统计滞后。
- DeepSeek 按历史每一步的峰值来摆放，比按平均热度摆放好：在请求相关性强时是 1.077 vs 1.090。这个设计是对的。
- 每次重排约 98.7% 的槽位都会换 expert，每卡约搬 4.9 GB，按 400 GB/s 算约 12 ms。每 150 s 才发生一次，可以忽略，所以缩短间隔几乎不花钱。
- 冗余数 × 分流方式（3 seed）：

| 每卡冗余 | 每卡 expert 数 | hash（现状） | 动态分流 |
|---|---|---|---|
| 1（现状） | 7 | 1.087 | 1.052 |
| 2 | 8 | 1.085 | 1.017 |
| 3 | 9 | 1.079 | **1.004** |

  只加冗余几乎没用，只做动态分流能改善一半，两者叠加才能把不均衡基本消掉。代价是每卡每步要多读 1/7 ~ 2/7 的 MoE 权重，每多一个冗余还要多占约 0.7 GB 显存（decode 显存不紧）。

## 需要真机验证的部分

1. 最便宜的实验：设置 `DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL=2500`，对比 moe_lb 和吞吐。
2. 跨 rank 协调：先在 64 卡上打出各 rank 的 attention 计时（`decode_core_attn` 等 timer 已经有了），确认确实存在拖尾等待，再在 `MTPVarlen::schedule` 里接入两个标量的 all-reduce。
3. 动态分流要改 `sm100_mega_gate.cuh` 的副本选择逻辑，并测出额外的计数交换开销。
4. 路由统计用的是合成数据，应该换成真实的路由 trace 重新校准。

## 第三轮：创新点与查新（2026-10-22）

| 想法 | 查新结果 | 结论 |
|---|---|---|
| 跨 rank 协调验证长度（把投机验证长度当成 rank 间负载均衡的旋钮） | 没找到前人工作。DSpark 论文脚注明确假设各 DP rank 已经由负载均衡器调平；SGLang 的 draft 参数是全局静态的 | **本研究的主要创新点** |
| 用现成的 accept_prob 在线校准 confidence head | DSpark 的 STS 是离线校准；AdaFlash 有在线更新长度头的做法 | 属于增量改进；DeepSeek 公开的数据里看不出有偏差，只当作"保险" |
| EPLB 每步动态分流 + 多放冗余 | DeepSeek 自己的 LPLB、SGLang Waterfill、UltraEP 都已在做 | 不新；我们的贡献只在定量拆解 |
| 语法感知的验证截断（JSON/工具调用中注定被拒的草稿不占验证名额） | 未查到针对 DSpark 的做法 | 想法成立（`sampler.py:366-376` 到验证时才检查语法），但 ZOS 下需要改 GPU 侧，也没有真实模型可以测非法草稿的比例，**未验证** |

补充检验（`mr_latency_scale.py`）：协调方案在提升吞吐的同时降低每用户延迟（TPOT −0.2% ~ −2.9%），没有任何设置变差；EP 越大收益越大（EP8 +0.2% ~ +1.4%，EP64 +0.8% ~ +3.0%）。

扩展校准（`calibrate_dspark_v2.py`）：加入 confidence 偏差参数后 loss 只从 25.9 降到 23.2，拟合出的偏差约为 0，仍然无法复现 conc=16 的 −1.5%。所以"head 没校准好"这个解释不被数据支持。

## 第四轮：借鉴 CPU / 操作系统的思路（2026-10-22）

### DeepInfer 已经借鉴的 CPU/OS 思路

| CPU / OS 思路 | DeepInfer 中的对应 | 位置 |
|---|---|---|
| 虚拟内存分页 | paged KV cache，block table 间接寻址 | `csrc/engine/block_manager.hpp` |
| fork() 写时复制 | `fork_seq`：引用计数 pin 共享块，只复制写了一半的最后一块 | `scheduler.hpp:1061`、`:602-617` |
| 分支预测 / 推测执行 | DSpark 草稿 + confidence head，验证失败即回滚 | `mtp_varlen.hpp` |
| 软件 TLB | DeepGEMM 分页 MQA 每个 lane 缓存一个页坐标 | `sm100_paged_mqa_logits.cuh` |
| 流水线 / 乱序执行 | 零开销调度：CPU 调度与 GPU 执行重叠；combine 期间算共享专家 | `worker.py`、`mlp.py` |
| 环形缓冲 | SWA 状态按 ring buffer 存放，128 token 边界合并写 | `cache_engine.py` |
| 存储层次 + LRU | HBM / 主机内存 / 3FS SSD 三级，3FS GC 使用 LRU | 部署文档 |
| 核隔离 / 资源划分 | `limit_num_sms` 给通信 kernel 预留 SM | `mlp.py` |

### 闭环验证（Kimi 公开生产 trace：Mooncake FAST25-release，1 小时，35,639 个请求）

数据下载（不随本仓库分发）：

```bash
mkdir -p data && for f in conversation_trace toolagent_trace; do curl -L -o data/$f.jsonl https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/$f.jsonl; done
```

**1. KV 复用特征（`trace_stats.py`）**：只算整块时，71%～73% 写进存储的 KV 块在之后 30 分钟以上的时间里从没被复用过；最热的 10% 块贡献了 75%～91% 的复用。

**2. 写准入（`kv_cache_sim.py`、`deadblock.py`）**

| 思路 | 结果 |
|---|---|
| 直接照搬 S3-FIFO / "第二次访问才写 SSD"（DRAM 试用队列） | SSD 写入减少 75%～92%，但中大容量下命中率掉 5～15 个点，**不划算**：KV 第一次复用往往晚于试用窗口，而未命中要重算 prefill，代价很高 |
| CPU 末级缓存的死块预测（写入时预测是否会被复用，预测"死"则绕过） | 只用 trace 能拿到的特征，预测 AUC 只有 0.62～0.64。容量紧张（5%）时**命中率 +1.5 个点，同时 SSD 写入减少 20%～37%**（双赢）；容量宽松时写入减少 15%～22%，代价是命中率下降 0.3～1.5 个点 |

**3. Prefill 调度（`prefill_sched.py`，真实到达时间回放，8 个引擎，相对开源版 FIFO）**

| 负载 | SRPT：平均 / P99 / 最大 1% 请求 | SRPT+老化：平均 / P99 / 最大 1% 请求 |
|---|---|---|
| 70% | −8%～−22% / +16%～+39% / ×1.5 | −10%～−23% / +12%～+22% / ×1.3 |
| 85% | −40%～−56% / −28%～+24% / ×2.0～2.4 | −33%～−46% / −17%～+14% / ×1.5 |
| 95% | −72%～−80% / −1%～−52% / ×2.6～3.5 | −36%～−45% / −2%～−19% / ×1.5～1.6 |

结论和操作系统教科书一致：SRPT 显著降低平均 TTFT，但会饿着长请求；老化用一部分平均收益换来尾延迟和公平。（DeepSeek 线上的 PLB 是动态分档且不开源，这里没有和它比较。）

### 查新

- Prefill 的 SRPT/SPF+老化：已有 Prophet、Nexus（arXiv 2507.06608）、arXiv 2606.09061、Chimera（arXiv 2603.22206）。
- KV SSD 层选择性准入：arXiv 2609.28870 明确主张"严格选择性准入"，arXiv 2609.07175 研究把写入和未命中解耦。用 CPU 死块预测这个名义做 KV 准入的没查到，但思路已很接近。

**本轮结论**：借鉴 CPU/OS 思路是对的方向，但能用的思路在 LLM serving 领域基本都已有人做过。本轮的价值在于用真实生产 trace 在 DeepInfer 的设定下做了定量验证，并测出了"照搬就会失败"的边界（S3-FIFO、纯 SRPT）。

## 第五轮：先找痛点，再谈优化（2026-10-22）

前几轮的问题：拿到代码就去调参数、刷指标，输入条件太少。这一轮先把 DeepSeek 自己公开的线上数据全部放进来（知乎《DeepSeek V4.1 Flash 大规模推理部署实践》的"实际线上运行数据"表与正文、DeepInfer 部署 Q&A），再加上外部故障率数据，算清每个痛点值多少钱（`pain_econ.py`，推导全部可复核）。

**线上数据（DeepSeek 公开）**：prefill 每卡 1.5M tok/s（含命中）/ 60K（不含命中），命中率 96%，平均输入 77K token，TTFT 平均 400 ms / P99 1000 ms；decode 每卡 14K tok/s，每用户 230 tok/s；P:D ≈ 1:1；每卡每小时收入 38 美元；单集群 155 万亿 token/天。自洽检验：1.5M × (1 − 96%) = 60K，完全吻合。

**推出的集群画像**：约 1,200 张 prefill 卡 + 1,200 张 decode 卡；输入 : 输出 ≈ 107 : 1（典型 agent 流量）；每个请求平均输出约 720 token，按 230 tok/s 算，**一个请求在 decode 上只活约 3 秒**；每张 decode 卡同时约 61 个请求；日收入约 218 万美元。

| # | 痛点 | 依据 | 量级 |
|---|---|---|---|
| 1 | **负载不均衡**：只能达到完美均衡离线基准的约 80% | 部署 Q&A 原话 | 若全部追回 ≈ 600 张卡 ≈ **54.5 万美元/天** |
| 2 | **命中率杠杆**：命中率已是 96%，prefill 却占半个集群；每 +0.1 个点命中率 = prefill 少 2.5% = 全集群少 1.2% | 线上数据表推导 | 每 0.1 个点 ≈ **2.7 万美元/天**；3 天 TTL 让 chatbot 掉 0.5～1 个点 = chatbot 的 prefill 多花 12%～25% |
| 3 | 存储：每天新写入约 6.9 PB KV；每卡逻辑容量 14.4 TB ≈ 5 天的写入量，所以 TTL 只能是 3～7 天 | 推导 + Q&A | 第四轮测得约 72% 写入的块从未被复用 → 准入过滤可以把有效容量放大 2～3 倍，从而延长 TTL |
| 4 | 结构性重算（128 对齐 + decoder 有界回放） | 博客的命中策略 | 约占 prefill 计算的 4%，**小** |
| 5 | 大 EP 故障爆炸半径：一卡坏整实例停 | 博客原话 + Meta Llama 3 故障率 | 只占 decode 容量的 0.01%～0.04%，**经济上很小**；但每次故障打断约 7,800 个请求（体验问题） |

**分诊结论**：钱主要在第 1、2、3 点。第 4、5 点经济上不值得优先做（第 5 点更多是体验问题）。

**第 1 点和我们第二、三轮发现的关系**：一个请求在 decode 上只活约 3 秒，每卡约 61 个并发，请求不停进出，所以各卡的并发数会持续随机波动；按泊松估计，标准差约 √61 ≈ 7.8，也就是 ±13%。EP 同步时，所有卡都要等最慢的那张。这正是"跨卡协调验证长度"要解决的问题。下一步应该用这组真实参数（每卡 61 个请求、3 秒寿命、77K 上下文、按最短队列路由）重新建模 decode 的不均衡，算出它在那 20% 里占多少、协调方案能追回多少。

## 第六轮：用线上真实参数重建 decode 负载不均衡（`dlb_sim.py`、`dlb_experiments.py`）

**模型**：EP64 一个实例，闭环（每卡固定 61 个并发，结束一个就路由进来一个）；输入、输出长度的分布形状取自 Kimi 生产 trace，缩放到 DeepSeek 的均值（77K / 约 720）；时间模型沿用已校准的 DSpark 与多 rank 耦合模型；EP 每层同步，step 时间 = 最慢 rank 的 attention + MoE × moe_lb（1.09）。

**独立自检**：模型算出每卡 13.5K tok/s、每用户 221 tok/s，DeepSeek 线上实测是 14K / 230。这组线上数字从未参与拟合，说明模型可信。

**decode 不均衡损失（相对完美均衡；路由信息延迟 20～70 步 ≈ 0.3～1.2 秒；agent 长度分布）**

| 路由架构 | 贪心路由（argmin）+ DSpark | 随机两选一（P2C）+ DSpark | P2C + 跨卡协调 |
|---|---|---|---|
| 路由信息实时 | 2.4%～2.9% | 2.6%～2.7% | 1.7% |
| 1 个统一路由器（记得自己的分配，只有"请求何时结束"是过期信息） | 2.9%～3.7% | 3.1%～3.8% | 1.8%～1.9% |
| 4 个独立路由器 | 4.7%～5.7% | 3.4%～4.2% | 1.9%～2.0% |
| 16 个独立路由器 | 11%～16% | 3.6%～4.7% | 1.9%～2.1% |
| 路由器连自己的分配都不记得 | 11%～29% | 3.6%～5.9% | 1.9%～2.5% |

其中约 1.2 个点来自 MoE 专家不均（moe_lb 1.09），只能靠 EPLB/LPLB 解决，协调方案对它无效。

**结论**

1. **机制**：一个请求在 decode 上只活约 3 秒，每卡每秒约 20 个请求进出。多个路由器拿着过期的负载信息做贪心分配时，会产生羊群效应：大家同时把请求塞给"看起来空"的卡（Mitzenmacher 2000，"How Useful Is Old Information?"）。
2. **它在那 20% 里占多少，取决于 DeepSeek 的路由架构**：如果是单一统一路由器，decode 不均衡只占约 3～4 个点，20% 的大头应该在 prefill、P/D 配比、扩缩容等其他环节；如果是多台 Entrypoint 并发路由，decode 不均衡可能占 10～16 个点甚至更多。公开资料不足以区分这两种情况。
3. **解法分两层，都几乎零成本**：
   - 路由层：把确定性的贪心分配换成随机两选一（P2C），无论有几个路由器，损失都压在约 3.4%～5.9%。这是经典做法，不新。
   - 每步层：跨卡协调验证长度（第三轮的创新点），在 P2C 的基础上再把损失压到约 2%。
4. **对第三轮结论的修正**：第三轮假设的"请求数 ±10%"对应的是多路由器的情形；在单一统一路由器下，协调方案的收益只有 +0.7～1.5%，没有之前说的 +3%。
