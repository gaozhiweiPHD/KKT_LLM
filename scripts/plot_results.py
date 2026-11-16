"""Plot only persisted measurements; missing experiments stay missing."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def save_figure(fig,path,dpi=160):
    fig.savefig(path,dpi=dpi,bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"),bbox_inches="tight")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--results",default="results"); p.add_argument("--checkpoints",default="checkpoints")
    a=p.parse_args(); root=Path(a.results); out=root/"figures"; out.mkdir(exist_ok=True)
    benchmark_dirs=["baselines","qwen_zero","qwen_sft","qwen_grpo_A","qwen_grpo_B","qwen_grpo_C"]
    paths=[root/name/f"{split}.csv" for name in benchmark_dirs for split in ("test_id","test_ood")
           if (root/name/f"{split}.csv").exists()]
    if not paths: raise RuntimeError("no measured evaluation files")
    df=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    df.to_csv(root/"all_evaluations.csv",index=False)
    matched=[]
    for split,part in df.groupby("split"):
        sets=[set(zip(g.problem_id,g.seed)) for _,g in part.groupby("method")]
        common=set.intersection(*sets)
        if not common: raise RuntimeError(f"No common problem/seed pairs in {split}")
        mask=[(pid,seed) in common for pid,seed in zip(part.problem_id,part.seed)]
        matched.append(part.loc[mask])
    df=pd.concat(matched,ignore_index=True)
    df.to_csv(root/"matched_evaluations.csv",index=False)
    metrics=["success","actions","invalid_action_rate","incorrect_stop","cycle","inference_seconds","solver_seconds",
             "wall_seconds","amortized_wall_seconds","episode_latency_seconds","input_tokens","output_tokens","stationarity","primal","dual","complementarity"]
    per_seed=df.groupby(["method","split","seed"])[metrics].mean()
    per_seed.groupby(level=["method","split"]).agg(["mean","std","count"]).to_csv(root/"comparison.csv")
    df.groupby(["split","method","n","m","condition","active_size","family","seed"])[metrics].mean().to_csv(root/"stratified_metrics.csv")
    df.groupby(["split","method","reason"]).size().rename("count").to_csv(root/"failure_reasons.csv")
    plt.rcParams.update({"font.size":10,"figure.figsize":(8,5),"pdf.fonttype":42})
    for key,label,filename in [("m","Number of constraints","success_vs_constraints"),
                              ("condition","Condition number of Q (n=12)","success_vs_condition"),
                              ("active_size","True active-set size (ID; analysis only)","success_vs_active_size"),
                              ("n","Dimension (m=30–40, condition ≤1000)","success_vs_dimension")]:
        if key=="m":
            fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
            panels=[(axes[0],df[df.n==12],"n=12"),(axes[1],df[df.n==16],"n=16")]
        else:
            fig,ax=plt.subplots()
            source=(df[df.n==12] if key=="condition" else
                    df[(df.m<=40)&(df.condition<=1000)] if key=="n" else
                    df[df.split=="test_id"])
            panels=[(ax,source,"")]
        for ax,source,title in panels:
            for method,g in source.groupby("method"):
                agg=g.groupby(["seed",key]).success.mean().groupby(level=key).agg(["mean","std"])
                ax.errorbar(agg.index,agg["mean"],yerr=agg["std"].fillna(0),marker="o",capsize=2,label=method)
            if key=="condition": ax.set_xscale("log")
            if key=="n": ax.set_xticks([12,16])
            ax.set(xlabel=label,ylabel="KKT success rate",ylim=(-.03,1.03),title=title)
        panels[-1][0].legend(fontsize=7)
        fig.tight_layout(); save_figure(fig,out/f"{filename}.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots()
    for j,(method,g) in enumerate(df.groupby("method")):
        ax.scatter(g.amortized_wall_seconds.mean(),g.success.mean(),s=65,
                   marker=["o","s","^","D","v","P","X","*","h"][j%9],label=method)
    ax.legend(loc="upper left",bbox_to_anchor=(1.01,1),fontsize=8)
    ax.set(xscale="log",xlabel="Mean amortized wall seconds / problem",ylabel="KKT success rate",ylim=(-.03,1.03))
    fig.tight_layout(); save_figure(fig,out/"cost_vs_success.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(); trace_rows=[]
    selected_keys=set(zip(df.method,df.problem_id,df.seed))
    for path in [root/name/"test_id_trajectories.jsonl" for name in benchmark_dirs
                 if (root/name/"test_id_trajectories.jsonl").exists()]:
        for line in path.open():
            record=json.loads(line)
            r=record["record"]
            if (r["method"],r["problem_id"],r["seed"]) not in selected_keys: continue
            for h in record["history"]:
                trace_rows.append({"method":record["record"]["method"],"iteration":h["iteration"],"merit":h["residuals"]["merit"]})
    if trace_rows:
        td=pd.DataFrame(trace_rows); td.to_csv(root/"iteration_residuals.csv",index=False)
        for method,g in td.groupby("method"):
            curve=g.groupby("iteration").merit.median(); ax.plot(curve.index,np.maximum(curve.values,1e-16),label=method)
        ax.set(yscale="log",xlabel="Agent iteration",ylabel="Median KKT merit (display floor: 1e-16)")
        ax.legend(fontsize=7); fig.tight_layout(); save_figure(fig,out/"violation_vs_iteration.png",dpi=160)
    plt.close(fig)
    logs=[p for p in Path(a.checkpoints).glob("grpo_*/training.jsonl") if p.parent.name in ("grpo_A","grpo_B","grpo_C")]
    if logs:
        fig,ax=plt.subplots()
        for path in logs:
            d=pd.read_json(path,lines=True); ax.plot(d.step,d.reward_mean.rolling(10,min_periods=1).mean(),label=path.parent.name)
        ax.set(xlabel="GRPO update",ylabel="Mean training reward (10-update average)")
        ax.legend(); fig.tight_layout(); save_figure(fig,out/"grpo_reward.png",dpi=160); plt.close(fig)
        fig,ax=plt.subplots()
        for path in [p for p in Path(a.checkpoints).glob("grpo_*/validation.jsonl") if p.parent.name in ("grpo_A","grpo_B","grpo_C")]:
            d=pd.read_json(path,lines=True)
            baseline=root/"sft_fixed_validation/validation_summary.json"
            if baseline.exists():
                initial=json.loads(baseline.read_text())["qwen_sft"]["mean"]["success"]
                d=pd.concat([pd.DataFrame([{"step":0,"success_rate":initial,"n":8}]),d],ignore_index=True)
            ax.plot(d.step,d.success_rate,marker="o",label=f"{path.parent.name} (n={d.n.iloc[0]})")
        ax.set(xlabel="GRPO update",ylabel="Independent validation KKT success",ylim=(-.03,1.03))
        if ax.lines: ax.legend(); fig.tight_layout(); save_figure(fig,out/"validation_success.png",dpi=160)
        plt.close(fig)
    sft=Path(a.checkpoints)/"sft"/"training_log.json"
    if sft.exists():
        data=[r for r in json.loads(sft.read_text()) if "loss" in r]
        fig,ax=plt.subplots(); ax.plot([r["step"] for r in data],[r["loss"] for r in data])
        ax.set(xlabel="SFT update",ylabel="Completion-token cross entropy")
        fig.tight_layout(); save_figure(fig,out/"sft_loss.png",dpi=160); plt.close(fig)
    print("Wrote",out)
if __name__=="__main__": main()
