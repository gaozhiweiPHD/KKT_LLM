# KKT_LLM：利用可验证的数学反馈训练优化决策 Agent

[English](README.md) · [复现步骤](docs/REPRODUCING.md) · [完整数学说明](docs/METHOD_ZH.md) · [原始研究报告](reference/report/research_report.md)

本项目研究：**能否通过 KKT 最优性反馈，让语言模型学会更好的多轮活跃集修复决策？**

我们将任务拆成三个明确的部分：

1. **Qwen3-0.6B + LoRA 决策**：根据违反约束、活跃乘子和残差，输出 `ADD(i)`、`DROP(i)` 或 `STOP`。
2. **数值模块计算**：给定活跃约束集，求解等式约束子问题，得到候选解和乘子。
3. **独立 KKT 验证与奖励**：检查 stationarity、primal feasibility、dual feasibility、complementarity，反馈给模型；只有主动 STOP 且四项归一化残差都不超过 1e-5 才算成功。

训练先用 oracle 教师轨迹做 SFT，再从同一道题、同一个初始活跃集采样 4 条完整轨迹，通过 GRPO 更新真实 LoRA 参数。奖励分为 A（成功奖励）、B（A + 残差改善）、C（B − 动作和非法动作成本）。Oracle 只参与离线数据/教师/curriculum 构造，在线策略不读取最优解或目标活跃集。

## 如何复现

无需 GPU 即可检查数学模块和查看小例子：

```bash
git clone https://github.com/gaozhiweiPHD/KKT_LLM.git
cd KKT_LLM
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
bash scripts/reproduce.sh cpu
bash scripts/run.sh examples/active_set_demo.py
```

完整环境使用 Python 3.13.9：

```bash
python -m pip install -r requirements.lock.txt
python -m pip install -e . --no-deps
```

从头复现，在 GPU 机器或已分配的集群计算节点内运行：

```bash
bash scripts/reproduce.sh data
bash scripts/reproduce.sh train
bash scripts/reproduce.sh evaluate
```

也可下载 Release 中的数据、最终权重和原始轨迹，直接评估或审计，省去重新训练。具体下载、校验、安装、显存与恢复训练说明见 [复现指南](docs/REPRODUCING.md)。`reference/` 保留原始结果，新运行写入 `results/` 和 `report/`。

## 已有结果与解释边界

统一采用 cold start、20 步上限、ID 1,000 题 / OOD 500 题、推理种子 11/22/33；下表为平均成功率。

| 方法 | ID | OOD |
|---|---:|---:|
| Zero-shot | 0.00% | 0.00% |
| SFT | 78.57% | 81.07% |
| GRPO-A | 84.63% | 90.73% |
| GRPO-B | 85.10% | 87.40% |
| GRPO-C | 83.93% | 89.33% |
| Greedy repair | 90.20% | 96.20% |
| Classical，20 次迭代 | 90.00% | 86.20% |
| OSQP，独立求解预算 | 100.00% | 100.00% |

这次实验中，GRPO-B 相比 SFT 的 ID 成功率提高 **6.53 个百分点**。所有学习模型只有一个训练种子；三个推理种子不代表三次独立重训。数值基线仍然很强，不能据此声称语言模型优于经典 QP 求解器。模型学习的是集合修复决策，候选解的线性代数计算由数值模块完成。

OOD 同时改变维度、条件数和约束数，不能用混合成功率单独判断某个维度的泛化。归一化容差证书也不等于精确算术证明。完整消融、时耗、原始/归一化残差和失败案例均保存在 [reference](reference/README.md)。
