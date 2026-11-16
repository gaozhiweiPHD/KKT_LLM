# KKT-Guided Reinforcement Learning for LLM-Based Optimization Agents

本项目研究：数学最优性反馈能否通过真实 Qwen3-0.6B LoRA post-training，改善多轮 active-set repair 决策。它不是用语言模型替代经典 QP 求解器。实际实验范围、训练更新数和结果见 [研究报告](../reference/report/research_report.md)；未运行的配置不算实验结果。

## 数学问题与最优性证明

求解严格凸 QP：

\[
\min_x f(x)=\tfrac12 x^\top Qx+c^\top x,\qquad Ax\le b,\quad Q=Q^\top\succ0.
\]

KKT 条件为 \(Qx+c+A^\top\lambda=0,\ Ax\le b,\ \lambda\ge0,\ \lambda\odot(Ax-b)=0\)。

**充分性及唯一性。** 设 \((\hat x,\hat\lambda)\) 满足精确 KKT 条件。对任意可行 \(y\)，令 \(d=y-\hat x\)，则

\[
f(y)-f(\hat x)
=(Q\hat x+c)^\top d+\tfrac12 d^\top Qd
=-\hat\lambda^\top A(y-\hat x)+\tfrac12d^\top Qd
=\hat\lambda^\top(b-Ay)+\tfrac12d^\top Qd\ge0.
\]

第二个等号使用 stationarity，第三个使用 complementarity。第一项因原始和对偶可行性非负；若 \(y\ne\hat x\)，正定性使第二项严格为正。因此 \(\hat x\) 是唯一全局最优解。乘子是否唯一另需约束秩条件。本数据构造中的真实活跃行满行秩，严格正乘子满足这些条件。非空闭可行集与强制增长的正定二次目标保证最优解存在。数值容差证书是近似证书，不等同于精确算术证明或固定的绝对解误差界。

## 数据及 oracle 隔离

生成 Q 的正交特征向量和可控特征值；约束行归一化；选取满行秩活跃集；设置活跃 slack 为零、非活跃 slack 严格为正。由已知 x_star、严格正活跃乘子构造 c。每题检查 KKT，并由 OSQP 独立验证。问题保存为 split.jsonl；真实解和活跃集单独保存为 split.oracle.jsonl；难度标签保存为 split.metadata.jsonl；数值验证与 SHA256 保存为 split.audit.jsonl。

数据量为训练 5,000、验证 500、ID 测试 1,000、OOD 测试 500。问题种子使用互不交叉的 10M、20M、30M、40M 区间，矩阵哈希检测重复。训练条件数 10/100/1000 的比例约 60%/30%/10%。OOD 包含 n=12、条件数 10000，以及 n=16、m=30–40 或 45–50。四类难度为 standard、small_slack、nearly_active、correlated；活跃规模 3–10。OOD 的维度、约束数、问题族须分层报告，混合成功率不能单独归因于维度变化。

环境、在线训练及评估策略只读公开 QP 数据。offline teacher 和 curriculum preparation 可以读训练 oracle；初始修复状态是任务输入，标签和目标活跃集不会传给在线策略。最终测试全部 cold start，无人工简化。评估后的分层统计才合并 metadata。

## 数值环境与验证器

活跃子问题使用 Q 的 Cholesky 分解和缩放后的 Schur complement 求解；不求逆。检查尺寸、有限值、对称正定性、活跃行数、秩、条件数和等式残差。无法执行的 ADD 原子性拒绝，不偷偷删除其他约束。空活跃集合法。重复动作、越界和格式错误返回反馈并占用一步。

对满行秩活跃行矩阵 \(A_I\)，Schur complement 为 \(S=A_IQ^{-1}A_I^\top\)，且对非零 u 有 \(u^\top Su=\|Q^{-1/2}A_I^\top u\|_2^2>0\)。因此可用 Cholesky 解 \(S\lambda_I=-b_I-A_IQ^{-1}c\)，再恢复 \(x=-Q^{-1}(c+A_I^\top\lambda_I)\)；这些 \(Q^{-1}\) 表达式在代码中均由线性求解实现。空集直接解无约束子问题。候选子问题可解不等于原 QP 可行或最优。

精确子问题求解后 stationarity 和 complementarity 通常接近机器精度，因为非活跃乘子为零，活跃等式被满足。因此本环境的残差奖励主要由 primal/dual violation 提供信号；四项仍全部独立复核，不能把几乎为零的两项解释为模型学会了求解线性系统。

动作只有 ADD(i)、DROP(i)、STOP，索引从零开始。每次合法状态变化重算候选解及证书；非法动作保持状态并复核。最多 20 次动作，包含 STOP。只有主动 STOP 且验证通过才算 success。错误 STOP、max_steps、cycle、numerical_error 分开保存。重复四次同一活跃集终止为 cycle。未 STOP 的可验证候选记录为 kkt_candidate，不记作 Agent success。

这是一种 **启发式 Active-Set Repair Environment**，没有经典可行 active-set 算法的全局收敛保证。独立 classical baseline 使用 phase-I LP、可行搜索方向和阻塞约束线搜索。

令 \(X=\max(1,\|x\|_\infty)\)，\(a_i=\|A_i\|_1\)，
\(G=\|Q\|_\infty X+\|c\|_\infty\)，\(P_i=a_iX+|b_i|\)。
验证器使用：

\[
r_s=\|Qx+c+A^\top\lambda\|_\infty/G,\quad
r_p=\max_i (A_ix-b_i)_+/P_i,
\]
\[
r_d=\max_i(-\lambda_i)_+a_i/G,\quad
r_c=\max_i|\lambda_i(A_ix-b_i)|/(GX).
\]

所有四项均不超过 1e-5 才通过。另保存未经归一化的四项残差。正目标缩放 \(Q,c\mapsto t(Q,c)\)、正约束行缩放 \(A_i,b_i\mapsto d_i(A_i,b_i)\)、对应乘子 \(\lambda_i\mapsto t\lambda_i/d_i\) 不改变归一化残差（浮点极限除外）。这不声称对任意变量单位变换不变。阈值由独立 OSQP、扰动和尺度测试校准。

compact observation 包含全部显著违反约束、活跃乘子、残差、当前集和步数，是部分信息、无几何矩阵的响应式策略。full observation 在首轮提供公开矩阵，并保留会话。两种模式不是信息等价的；full 模式可评估但不能将未运行结果当作消融结论。

## SFT 与 GRPO

SFT 启用 activation checkpointing；GRPO 用 eval 模式关闭 dropout，但反传时保持 autograd，4 条转移的微批量可直接装入显存，不依赖 activation checkpointing。

SFT 使用 TRL SFTTrainer，对真实 Qwen3 LoRA 参数进行 completion-only 交叉熵训练。教师轨迹包含 cold、missing、extra、mixed、heuristic warm start，保存动作、执行结果和下一观测。当前 compact 策略逐轮读当前状态，环境跨轮持久化；它不是一次预测最终活跃集。Oracle 提供监督动作，但模型输入不含 oracle 答案。

主 GRPO 后端位于 kkt_agent/grpo.py。每组在**同一 QP、同一初始集**上用当前 Qwen 采样 4 条完整交互轨迹，全部生成结束后才更新。temperature=0.7、top_p=1、top_k=0；对生成时和更新时相同的温度缩放 token 概率计算比率。只优化模型生成的动作及 EOS token，环境观测不进入损失。每条轨迹的 terminal reward 产生 group-relative advantage：

\[
\hat A_g=(R_g-\bar R)/(\operatorname{std}_{pop}(R)+10^{-4}),
\]
\[
L=-\frac1G\sum_g\frac1{T_g}\sum_t
\min(\rho_{g,t}\hat A_g,\operatorname{clip}(\rho_{g,t},0.8,1.2)\hat A_g).
\]

这里使用原始逐轨迹 token 平均 GRPO 目标、一次 on-policy update、KL 系数 0。因为只做一次更新，old 与 current 参数在整组反传完成前相同，按官方 mu=1 形式使用当前 replay log-probability 的 stop-gradient 分母；初始 ratio 精确为 1。此时由于组内 advantage 的均值为零，标量 loss 常接近零，但 stop-gradient 分母使其导数不为零；真实学习须看梯度、参数变化及独立测试，不能要求该 loss 曲线单调下降。额外保存实际采样 log-probabilities 与未填充 replay log-probabilities，并单列 FP16/batch shape 的数值差异。保存 old log-probabilities，检查更新前 ratio≈1、loss/梯度有限、组方差、LoRA 参数变化。零奖励方差不会被伪造为学习信号。短动作后端便于检查多轮状态与 token mask，也避免每轮工具 JSON 的额外 token 开销；它明确是自行实现的 GRPO，而不是声称调用了 TRL GRPOTrainer。

另提供可选 scripts/train_grpo_trl.py 和 QPToolEnvironment，用当前 TRL environment_factory API 暴露唯一 apply_action 工具。该后端要求 tool-call dialect 的 SFT checkpoint；默认直接动作 SFT checkpoint 与工具格式不同，不能假定二者互换。此可选后端提供接口实现，尚未完成端到端验证，不计入主实验。

奖励：
- A：只有经主动 STOP 验证成功才给 1。
- B：A 加 \(0.5\operatorname{clip}(\log(1+V_0)-\log(1+V_T),-1,1)\)，\(V=r_s+r_p+r_d+r_c\)。
- C：B 减去 0.01×动作数和 0.05×非法动作数。

A/B 按消融定义不附加非法动作罚分；非法动作仍占用预算，C 明确计罚。残差改善不是最优性成功。无效 STOP 不获得 binary 奖励。记录提前停止和组奖励方差，分析潜在奖励偏好。

## 安装与运行

可移植安装、下载原始数据和权重、完整训练与评估命令见 [复现指南](REPRODUCING.md)。原始结果位于 [reference](../reference/README.md)。

## 日志、复现和限制

错误 STOP 率按 episode 计；非法动作率是每题 invalid/actions 的均值。repair 的 cycle 标志表示至少重复访问一次活跃集，重复四次才触发 cycle 终止；完整终止原因另列。CSV/JSONL 保存成功率、四类原始及归一化残差、动作数、非法率、错误 STOP、cycle、token 用量、推理时间、求解时间和完整墙钟时间。模型加载是每次评估一次的 setup 开销，须单列。GRPO checkpoint 包含 LoRA、配置、优化器、AMP scaler 和随机数状态；恢复使用 --resume。SFT 的 final 目录用于加载 adapter 做推理，继续 SFT 应使用 Trainer 保存的 checkpoint-100 或 checkpoint-200 目录，其中包含优化器与训练状态。不把 reward curve 当作独立测试成功率曲线。

数值方法具有本任务上的结构优势。compact observation 缺少约束间几何信息；teacher action accuracy 是某条教师路径的一致率，不等价于唯一正确决策率。近活跃约束可能在数值容差内产生不同但有效的活跃集。经典 baseline 的一次迭代与 LLM 一次动作不是同等成本。固定 20 步也可能截断经典求解器；可额外运行 classical200，但必须单列预算。

本项目不预设 RQ1–RQ4 的答案为肯定。训练预算、样本量、单训练种子、未运行工具格式/完整信息消融等限制需保留在结果报告中。

官方参考：[TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer)、[Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)。实际 API 兼容性存于 results/api_compatibility.json。
