"""Build evidence tables from actual saved runs, never from placeholder data."""
import json,re,sys
from pathlib import Path
import numpy as np
import pandas as pd

root=Path("."); lines=["# KKT-Guided Qwen3 Optimization Agent：实测研究报告","",
"本报告由已保存的训练审计、CSV 和 JSONL 自动汇总。数学证明、残差尺度与算法定义见 [README](../README.md)。未运行实验不会补入估计值。",""]
env=json.load(open("results/environment.json"))
lines += ["## 执行环境与可复现性", "",
          f"Python {env.get('python', sys.version.split()[0])}；GPU：{env['gpu']}；PyTorch {env['torch']} / CUDA {env['cuda_version']}；Transformers {env['transformers']}；TRL {env['trl']}；PEFT {env['peft']}。",
          "当前结果来自工作目录。原始实验记录单独保存在 reference/。训练超参数与实际更新数见下方 parameter_audit 汇总。",
          "模型 revision：" + Path("configs/model_revision.txt").read_text().strip() + "。原始依赖见 requirements.lock.txt；本次环境见 results/environment.json。", ""]
testlog=Path("results/pytest.log").read_text()
matches=re.findall(r"\d+ passed[^\n]*",testlog)
lines += [f"最后保存的测试结果：{matches[-1] if matches else '请查看 results/pytest.log'}。",""]
manifest=json.load(open("data/manifest.json"))
lines += ["## 最优性证书的数学含义","",
          r"对任意可行 y，若候选 (x, λ) 精确满足 KKT，则 $f(y)-f(x)=\lambda^\top(b-Ay)+\frac12(y-x)^\top Q(y-x)\ge0$。这里使用了 stationarity 和 complementarity；λ≥0 与 b−Ay≥0 使第一项非负，Q 正定使 y≠x 时第二项严格正。因此 x 是唯一全局最优解。",
          "数值实验采用四项尺度归一化残差各 ≤1e-5 的近似证书，不能把它表述为精确算术证明。尺度定义及正目标/约束行缩放不变性见 README。",""]
lines += ["## 数学与数据验证","","| Split | QP 数量 | OSQP 验证失败 | 最大归一化 KKT merit | 最大解误差 infinity norm |","|---|---:|---:|---:|---:|"]
for split,d in manifest["splits"].items():
    lines.append(f"| {split} | {d['count']} | {len(d['osqp_failures'])} | {d['max_osqp_merit']:.3g} | {d['max_x_error_inf']:.3g} |")
lines += ["","所有问题来自 primal-dual construction，Q 正定、非活跃 slack 严格为正、真实活跃行满行秩。训练条件数以 10/100 为主，包含少量 1000；OOD 分为条件数 10000 和 n=16 的维度/约束数泛化。",
          "oracle 解、乘子和活跃集独立存储；最终评估全部 cold start，在线策略无 oracle 读取。评价成功由四项归一化残差以及显式 STOP 决定。",""]
teacher=json.load(open("results/teacher_train.json"))
lines += ["## SFT 与 GRPO 的真实性证据","",
          "默认复现配置使用前 12,000 个监督状态、200 次 SFT 更新及每次梯度累积 4 个样本；三种 GRPO 默认各 200 次更新、每组 4 条轨迹。这些是默认协议，实际训练更新数与数据量应以下方审计和保存的配置为准。",f"Teacher：{teacher['trajectories']} 条轨迹，成功率 {teacher['success_rate']:.2%}；动作分布 {teacher['action_distribution']}。",
          "Teacher 成功率是离线监督轨迹质量，不是学习模型的测试性能。训练表秒数包含训练循环、周期验证与保存，不包含模型加载及初始数据预处理；评估则另存程序总墙钟。","",
          "| 训练方法 | 更新数 | LoRA 改变参数数 | 参数差 L2 | 奖励 std>1e-8 的组数 | 训练秒数 |","|---|---:|---:|---:|---:|---:|"]
audits={}
for name in ["sft","grpo_A","grpo_B","grpo_C"]:
    path=Path("checkpoints")/name/"parameter_audit.json"
    if path.exists():
        d=json.loads(path.read_text()); audits[name]=d
        lines.append(f"| {name} | {d['steps']} | {d['changed_parameters']} | {d['delta_l2']:.5g} | {d.get('groups_with_variance','—')} | {d['seconds']:.1f} |")
    else: lines.append(f"| {name} | 未完成 | — | — | — | — |")
lines += ["","SFT 使用 TRL SFTTrainer、completion-only loss。GRPO 主实验使用公开可审计的多轮 token-level 实现：4 条同题、同初始状态轨迹；在整组生成后，以 group-relative advantage 更新真实 Qwen LoRA。按轨迹 token 均值计算 clipped surrogate，KL=0，top_p=1 / top_k=0，采样与重算 log-probabilities 均使用 temperature=0.7。环境观测没有被当作生成 token 优化。",
          "保存了逐组 rewards、advantage、动作/环境轨迹、loss、grad_norm、更新前 ratio 误差、优化器和随机数状态。单次 on-policy 更新的 ratio=1 且组内 advantage 均值为零，标量 loss 接近零是预期现象；其导数仍可非零，真实更新由梯度和参数差证实。可选 TRL environment_factory 工具格式后端提供代码，但不计入下面的主实验结果。",""]
diagnostics=[]
for variant in ("A","B","C"):
    path=Path("checkpoints")/f"grpo_{variant}"/"training.jsonl"
    if path.exists():
        train=pd.read_json(path,lines=True)
        for level,g in train.groupby("level"):
            diagnostics.append({"reward":variant,"level":int(level),"updates":len(g),
                "reward_std_above_1e_minus8_fraction":float((g.reward_std>1e-8).mean()),
                "mean_reward_std":float(g.reward_std.mean()),"mean_group_success":float(g.group_success_rate.mean()),
                "mean_actions":float(g.mean_actions.mean()),"mean_grad_norm":float(g.grad_norm.mean())})
if diagnostics:
    pd.DataFrame(diagnostics).to_csv("results/training_diagnostics.csv",index=False)
    lines += ["| 奖励 | Curriculum 阶段 | 更新数 | std>1e-8 组比例 | 平均组内 std | 平均 rollout 成功率 | 平均动作 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for d in diagnostics:
        lines.append(f"| {d['reward']} | {d['level']} | {d['updates']} | {d['reward_std_above_1e_minus8_fraction']:.1%} | {d['mean_reward_std']:.4g} | {d['mean_group_success']:.1%} | {d['mean_actions']:.2f} |")
    lines += ["","这里是实际训练 rollout 统计，不能替代独立测试。阶段 0/1/2 分别为接近正确集、混合错误集、cold start。组方差为零时没有新的 group-relative policy-gradient 信号；增加非零组数也不保证测试性能提高。",""]
resume=Path("checkpoints/grpo_smoke/resume_audit.json")
if resume.exists():
    d=json.loads(resume.read_text())
    lines += [f"GRPO 恢复验证：从第 {d['resumed_from_step']} 步恢复至第 {d['steps']} 步，参数差 L2={d['delta_l2']:.5g}，finite_losses={d['finite_losses']}。",""]
action=Path("results/sft_action_accuracy.json")
if action.exists():
    d=json.loads(action.read_text()); nq=len({r["problem_id"] for r in d["results"]})
    lines += [f"独立验证集教师动作评估：{d['n']} 个状态、{nq} 个不同 QP；exact action accuracy={d['accuracy']:.2%}，可解析率={d['parse_rate']:.2%}。教师动作不是唯一可能成功的动作，因此此指标不替代终局 KKT 成功率。",""]
matched=Path("results/matched_evaluations.csv")
if matched.exists():
    df=pd.read_csv(matched)
    lines += ["## 统一预算的独立测试结果","",
              "下表只使用所有方法共有的 (problem_id, inference seed) 测试对。LLM 正式测试使用相同的批量设置（具体 batch_size 见各 setup.json）。所有修复 Agent 初始集为空，最多 20 次决策；OSQP 独立直接求解。classical 的 phase-I LP 成本包含在时间内。失败不调用 OSQP 补救。",
              "错误 STOP 率按 episode 计；非法动作率先计算每题 invalid/actions，再对题目取均值；动作数包含非法尝试和 STOP。均值 ± 标准差来自推理随机种子；所有学习模型只有一个训练种子 42。这不是多训练种子的统计证据。","",
              "| Split | 方法 | QP 数 | 推理种子数 | KKT success | 平均动作 | 非法动作率 | 错误 STOP | 平均单题延迟(s) | 分摊墙钟(s) |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (split,method),g in df.groupby(["split","method"]):
        by=g.groupby("seed").success.mean(); sd=by.std(ddof=1)
        sd_text=f"{sd:.2%}" if len(by)>1 else "未估计"
        amort=g.amortized_wall_seconds.mean() if "amortized_wall_seconds" in g else g.wall_seconds.mean()
        lines.append(f"| {split} | {method} | {g.problem_id.nunique()} | {len(by)} | {by.mean():.2%} ± {sd_text} | {g.actions.mean():.2f} | {g.invalid_action_rate.mean():.2%} | {g.incorrect_stop.mean():.2%} | {g.wall_seconds.mean():.4f} | {amort:.4f} |")
    lines += ["","分层结果见 stratified_metrics.csv，终止原因计数见 failure_reasons.csv。完整四项残差、cycle、token、模型推理和数值求解时间见 [比较 CSV](../results/comparison.csv) 与 [逐题 CSV](../results/matched_evaluations.csv)。加载模型的 setup 时间和真实评估总墙钟分别存于每个结果目录的 setup.json，不能将并行批内单题延迟相加当作总墙钟。",""]
    stats=df.groupby(["split","method"]).success.mean()
    def value(split,method):
        return stats.get((split,method),np.nan)
    lines += ["## RQ1–RQ4 的证据与边界",""]
    s=value("test_id","qwen_sft"); z=value("test_id","qwen_zero")
    a=value("test_id","qwen_grpo_A"); b=value("test_id","qwen_grpo_B"); c=value("test_id","qwen_grpo_C")
    greedy=value("test_id","greedy")
    lines += [f"**RQ1：KKT reward 是否能 post-train active-set repair？** 实际 LoRA 更新已由参数差和日志确认。ID 成功率：zero-shot={z:.2%}，SFT={s:.2%}，GRPO-A={a:.2%}，GRPO-B={b:.2%}，GRPO-C={c:.2%}。参数更新本身不能证明优化能力提升；必须用这些独立终局结果与 SFT 对照。",""]
    if np.isfinite([s,a,b,c]).all():
        gain=max(a,b,c)-s
        lines += [(f"本次单训练种子中，最佳 GRPO 相对 SFT 的 ID 成功率提高 {gain*100:.2f} 个百分点，说明该设置下存在实际测试增益；这不等同于多训练种子上的稳健结论。" if gain>0 else
                   f"本次测试未观察到 GRPO 相对 SFT 的 ID 成功率提高（最佳差值 {gain*100:.2f} 个百分点）。"),""]
    behavior=df[df.split=="test_id"].groupby("method")[["actions","cycle","invalid_action_rate","incorrect_stop"]].mean()
    if {"qwen_sft","qwen_grpo_A"}.issubset(behavior.index):
        bs=behavior.loc["qwen_sft"]; ba=behavior.loc["qwen_grpo_A"]
        lines += [f"ID 行为变化（SFT→Binary）：重复状态比例 {bs['cycle']:.2%}→{ba['cycle']:.2%}，平均动作 {bs['actions']:.2f}→{ba['actions']:.2f}，非法动作率 {bs['invalid_action_rate']:.2%}→{ba['invalid_action_rate']:.2%}。成功率变化伴随着不同的多轮轨迹；非法动作同时包含语法和状态合法性错误，仅凭汇总指标不能完全拆开两者的因果贡献，后面的语法错误细分提供补充证据。",""]
    lines += [f"**RQ2：Residual 是否优于 Binary？** 同题测试中 B−A 的 ID 成功率差为 {(b-a)*100:.2f} 个百分点。下列按问题重采样的 paired bootstrap 只描述当前训练种子与测试集的不确定性，不包含重训方差。",""]
    deltas={}
    rng=np.random.default_rng(12345)
    for split in sorted(df.split.unique()):
        part=df[df.split==split].groupby(["problem_id","method"]).success.mean().unstack()
        if {"qwen_grpo_A","qwen_grpo_B"}.issubset(part):
            dif=(part.qwen_grpo_B-part.qwen_grpo_A).dropna().to_numpy()
            draws=np.mean(rng.choice(dif,size=(2000,len(dif)),replace=True),axis=1)
            lo,hi=np.quantile(draws,[.025,.975]); deltas[split]={"B_minus_A":float(dif.mean()),"ci95":[float(lo),float(hi)]}
            lines.append(f"- {split}：差值 {dif.mean():.2%}，问题级 bootstrap 95% 区间 [{lo:.2%}, {hi:.2%}]。")
            interpretation=("区间跨零，不能据此确认 Residual 优于 Binary。" if lo<=0<=hi else
                            "在当前固定训练种子的测试重采样中，Residual 的差值为正。" if lo>0 else
                            "在当前固定训练种子的测试重采样中，Residual 的差值为负。")
            lines.append("  "+interpretation)
    Path("results/paired_bootstrap.json").write_text(json.dumps(deltas,indent=2))
    cost_stats=df[df.split=="test_id"].groupby("method")[["actions","incorrect_stop","success","amortized_wall_seconds"]].mean()
    if {"qwen_grpo_B","qwen_grpo_C"}.issubset(cost_stats.index):
        rb=cost_stats.loc["qwen_grpo_B"]; rc=cost_stats.loc["qwen_grpo_C"]
        lines += ["",f"ID 成本惩罚检查：B→C 的平均动作数为 {rb['actions']:.2f}→{rc['actions']:.2f}，错误 STOP 率为 {rb['incorrect_stop']:.2%}→{rc['incorrect_stop']:.2%}，KKT 成功率为 {rb['success']:.2%}→{rc['success']:.2%}，分摊墙钟为 {rb['amortized_wall_seconds']:.4f}→{rc['amortized_wall_seconds']:.4f} 秒/题。缩短轨迹本身不作为成功，必须连同后两项判断是否存在提前停止偏好。",""]
    if {"qwen_grpo_B","qwen_grpo_C"}.issubset(cost_stats.index):
        if rc["actions"]>=rb["actions"]:
            lines += ["本次成本系数没有减少 ID 测试集的平均动作数；实际计算成本须另看保存的墙钟和 token 指标。",""]
        if rc["incorrect_stop"]>rb["incorrect_stop"]:
            lines += ["C 的 ID 错误 STOP 率较 B 上升，提示需要检查提前停止偏好。该判断来自实际错误 STOP 记录；不能把较短的失败路径计为优化成功，也不能从一个训练种子直接推出普遍因果结论。",""]
    lines += ["",f"**RQ3：是否泛化？** OOD 上 SFT={value('test_ood','qwen_sft'):.2%}，GRPO-B={value('test_ood','qwen_grpo_B'):.2%}。OOD 包含维度、约束数、条件数及问题族差异，需按下表分层；混合 OOD 成功率不构成单一维度效应的因果结论。","",
              "| 方法 | n | cond(Q) | QP 数 | KKT success |","|---|---:|---:|---:|---:|"]
    for (method,n,cond),g in df[df.split=="test_ood"].groupby(["method","n","condition"]):
        lines.append(f"| {method} | {n} | {cond:g} | {g.problem_id.nunique()} | {g.success.mean():.2%} |")
    lines += ["",f"**RQ4：是否超过简单启发式，代价如何？** ID greedy={greedy:.2%}；最佳学习模型={max(s,a,b,c):.2%}。"+
              ("本次学习模型未超过 greedy，不能声称取代确定性启发式。" if max(s,a,b,c)<=greedy else "当前测试均值超过 greedy，但仍需更多独立训练种子确认。"),
              "LLM 的 GPU 推理、token 和模型加载成本必须与毫秒级数值算法一起考虑。训练成本是额外成本，不从单题推理时间中隐藏。",""]
else:
    lines += ["## 当前实验状态","","尚无完整匹配测试 CSV，RQ1–RQ4 暂不作性能结论。",""]
if matched.exists():
    lines += ["## 失败类型的实测分解","",
              "| Split | 模型 | 失败数 | 最常见终止原因（失败内占比） | 重复状态比例 |",
              "|---|---|---:|---|---:|"]
    for (split,method),g in df[df.method.isin(["qwen_sft","qwen_grpo_A","qwen_grpo_B","qwen_grpo_C"])].groupby(["split","method"]):
        failures=g[~g.success]; counts=failures.reason.value_counts()
        dominant=f"{counts.index[0]} ({counts.iloc[0]/len(failures):.1%})" if len(counts) else "无失败"
        lines.append(f"| {split} | {method} | {len(failures)} | {dominant} | {g.cycle.mean():.2%} |")
    lines += ["","终止原因描述观察到的失败形式，不单独证明因果机制。重复状态比例包括尚未达到四次重复终止门槛的回访。必须结合非法动作、错误 STOP、逐轮残差及具体轨迹，区分格式、预算与集合修正问题。",""]
audit_path=Path("results/certificate_audit.json")
if audit_path.exists():
    ca=json.loads(audit_path.read_text())
    lines += ["## 保存结果的独立复核与容差敏感性","",
              f"从公开 QP 与保存的最终 x、λ 重算了 {ca['checked_records']} 条记录，残差/成功标记不一致数为 {ca['mismatches']}；另检查 repair 成功均来自显式 STOP、没有超出 20 步、非法动作未改变活跃集。复核不读取 oracle。","",
              "下表固定已生成的轨迹和最终候选，仅收紧验证阈值；不是在新阈值下重新运行策略。","",
              "| Split | 方法 | τ=1e-5 | τ=1e-7 | τ=1e-9 |","|---|---|---:|---:|---:|"]
    for g in ca["groups"]:
        rates=[g["successful_stop_counts"][str(t)]/g["count"] for t in ca["tolerances"]]
        lines.append(f"| {g['split']} | {g['method']} | {rates[0]:.2%} | {rates[1]:.2%} | {rates[2]:.2%} |")
    lines.append("")
    lines += ["| Split | 模型 | 语法非法/全部动作 | 其他非法/全部动作 |","|---|---|---:|---:|"]
    for g in ca["groups"]:
        if g["method"].startswith("qwen"):
            syntax=g["format_invalid_actions"]/max(1,g["actions"])
            other=(g["invalid_actions"]-g["format_invalid_actions"])/max(1,g["actions"])
            lines.append(f"| {g['split']} | {g['method']} | {syntax:.3%} | {other:.3%} |")
    lines += ["","这里按全部动作数汇总，区别于主表按题平均的非法率。语法非法由严格 parser 重算；其他非法包括重复/越界/不适用动作或数值拒绝。该细分来自实际轨迹，不调用 oracle。",""]
    for name,td in ca["training_groups"].items():
        lines.append(f"GRPO-{name}：独立复核 {td['updates']} 个更新、{td['rollouts']} 条 rollout，同组任务/初始状态、reward 重算、group advantage 和 token 数均一致；最大采样/replay log-probability 数值差 {td['max_sampling_replay_logprob_difference']:.4g}。")
    lines.append("三种消融的训练问题与初始状态序列一致。FP16 生成与不同 batch shape 的 replay 存在上述数值差，按单次 on-policy 更新采用同一 replay 前向的 stop-gradient 分母；这项数值局限不会被 ratio=1 的记录掩盖。")
    lines.append("")
lines += ["## 学习曲线、轨迹与失败分析","",
          "图表分层：活跃集规模图仅用 ID；维度图限制到共同的 m=30–40、条件数≤1000 范围；约束数图按 n 分面；条件数图固定 n=12。实测图位于 results/figures：GRPO reward、独立 validation success、迭代 merit、约束数量/条件数/活跃集规模/维度泛化、成本与成功率，以及 SFT loss。",
          "GRPO curriculum 在第 40/100 步改变起始状态和最大轮数，训练 reward 曲线跨阶段不可直接解释为同分布进步。独立验证为固定 cold-start 任务；8 题曲线分辨率较低，最终测试才是主指标。",
          "残差逐轮图使用仍被观测到的轨迹中位数，存在提前终止的样本选择效应；原始轨迹保留供复核。",
          "至少三个真实成功与三个失败轨迹见 [case studies](case_studies.md)，对应机器可读记录见 case_studies.json。若没有足够 LLM 成功样例，文件会明确使用其他方法的真实轨迹，不能伪造成功。","",
          "## 局限与剩余工作","",
          "- 只有一个模型训练种子；多个推理种子的标准差不涵盖训练随机性。需要独立重训才能作稳健的 reward 排名。",
          "- zero-shot 到训练模型的增益可能同时来自输出格式学习与优化决策学习；非法动作率和 SFT 对照提供部分区分，但仍需要额外的格式约束 zero-shot 消融才能完全拆开这两种作用。",
          "- 每次动作后的等式子问题由数值模块求解，stationarity/complementarity 通常已接近机器精度；Residual Reward 的有效信号主要是 primal/dual violation。模型学习的是集合修正决策，而非线性代数求解。",
          "- compact 模式缺少约束几何，可能导致不同 QP 的相似观察具有不同最佳动作。Oracle teacher 路径也未必能由 compact 观察唯一恢复。",
          "- A 的奖励可能稀疏；B 的小残差改善未必足以产生有用方差；C 可能偏好短轨迹。逐组日志和错误 STOP 指标用于检查这些问题，不能靠泄漏 oracle 解决。",
          "- n=16 与更大条件数上的失败可来自观察、索引决策、训练覆盖和 20 步截断，不能只归因于模型容量。",
          "- 构造时乘子量级保持不变，提高 Q 条件数不保证 repair 任务更难；同时归一化 dual residual 的尺度也随 Q 改变。较高 OOD 成功率不能单独证明对病态性的鲁棒性，原始残差与容差敏感性应一起考察。",
          "- 全信息与 thinking 模式、MLP/GNN、parametric warm start、更多训练种子未计入本次已完成实验。",
          "- 原生 TRL 工具调用后端与主要直接文本动作后端的格式不同，提供接口实现但尚未完成端到端验证，不作为性能证据。",
          "- classical20 与 repair Agent 使用相同上限，但每步含义不同；OSQP 没有人为 20 次迭代限制。数值 solver 对照是参考性能，不是同构策略。",
          "", "所有结论以保存的实际结果为准，没有将示意数据、oracle 轨迹或其他求解器的补救结果记为模型成功。"]
Path("report").mkdir(exist_ok=True)
Path("report/research_report.md").write_text("\n".join(lines)+"\n")
print("Wrote report/research_report.md")
