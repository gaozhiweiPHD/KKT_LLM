"""Identical cold-start tasks/20 decisions for every repair agent; no fallback solver."""
import argparse,json,csv
from pathlib import Path
from time import perf_counter
PROGRAM_START=perf_counter()
import numpy as np
from kkt_agent.io import load_qps,read_jsonl,dump_json
from kkt_agent.active_set_env import ActiveSetEnv
from kkt_agent.baselines import greedy,random_valid,classical_active_set
from kkt_agent.qp_solver import solve_osqp
from kkt_agent.kkt_verifier import verify
from kkt_agent.evaluation_contract import validate_model_label

def main():
    p=argparse.ArgumentParser(); p.add_argument("--split",default="test_id"); p.add_argument("--limit",type=int)
    p.add_argument("--methods",nargs="+",default=["random","greedy","classical","osqp"])
    p.add_argument("--seeds",nargs="+",type=int,default=[11,22,33]); p.add_argument("--adapter")
    p.add_argument("--batch-size",type=int,default=8); p.add_argument("--model",default="Qwen/Qwen3-0.6B"); p.add_argument("--output",default="results/baselines")
    p.add_argument("--observation",choices=["compact","full"],default="compact")
    a=p.parse_args()
    try: validate_model_label(a.methods,a.adapter)
    except ValueError as exc: p.error(str(exc))
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    qps=load_qps(f"data/{a.split}.jsonl",a.limit)
    # Metadata is joined only after a policy has returned its completed record.
    metadata={r["problem_id"]:r for r in read_jsonl(f"data/{a.split}.metadata.jsonl")}
    policy=None
    if any(m.startswith("qwen") for m in a.methods):
        from kkt_agent.llm import QwenPolicy
        policy=QwenPolicy(a.model,a.adapter,mode=a.observation)
    records=[]; evaluation_start=perf_counter()
    with (out/f"{a.split}_trajectories.jsonl").open("w") as traces:
        for method in a.methods:
            for seed in a.seeds:
                rng=np.random.default_rng(seed)
                if policy: policy.seed(seed)
                if method.startswith("qwen"):
                    from kkt_agent.batch_eval import evaluate_batches
                    llm_results=iter(evaluate_batches(policy,qps,a.batch_size))
                for j,qp in enumerate(qps):
                    if method=="osqp":
                        start=perf_counter(); s=solve_osqp(qp); c=verify(qp,s.x,s.lam)
                        rec={"problem_id":qp.problem_id,"success":c.success,"reason":"kkt_success" if c.success else "verification_failed",
                             "actions":0,"invalid_actions":0,"invalid_action_rate":0.,"incorrect_stop":False,"cycle":False,
                             "inference_seconds":0.,"solver_seconds":s.seconds,"wall_seconds":perf_counter()-start,
                             "input_tokens":0,"output_tokens":0,**c.to_dict(),"kkt_candidate":c.success}
                        history=[{"iteration":0,"active":[],"action":"OSQP","residuals":c.to_dict(),
                                  "candidate":{"x":s.x.tolist(),"lambda":s.lam.tolist()}}]
                    elif method.startswith("qwen"):
                        rec,history=next(llm_results)
                    elif method in ("classical","classical200"):
                        rec,history=classical_active_set(qp,200 if method=="classical200" else 20)
                    else:
                        env=ActiveSetEnv(qp)
                        if policy and method.startswith("qwen"): policy.reset(env)
                        while not env.done:
                            if method=="greedy": action=greedy(env)
                            elif method=="random": action=random_valid(env,rng)
                            elif method.startswith("qwen"): action=policy.act(env)
                            else: raise ValueError(f"unknown method {method}")
                            env.step(action)
                        rec,history=env.record(),env.history
                        history[-1]["candidate"]={"x":env.solution.x.tolist(),"lambda":env.solution.lam.tolist()}
                    rec.setdefault("episode_latency_seconds",rec["wall_seconds"])
                    rec.setdefault("amortized_wall_seconds",rec["wall_seconds"])
                    rec.setdefault("inference_batch_size",1)
                    rec.update(method=method,seed=seed,split=a.split,observation=a.observation)
                    rec.update({k:v for k,v in metadata[qp.problem_id].items() if k not in ("seed","problem_id")})
                    records.append(rec)
                    traces.write(json.dumps({"record":rec,"history":history})+"\n"); traces.flush()
                    if (j+1)%100==0: print(method,seed,j+1,flush=True)
                group=[r for r in records if r["method"]==method and r["seed"]==seed]
                print(method,seed,"success",np.mean([r["success"] for r in group]),flush=True)
    with (out/f"{a.split}.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    numeric=["success","actions","invalid_action_rate","incorrect_stop","cycle","inference_seconds","solver_seconds",
             "wall_seconds","amortized_wall_seconds","episode_latency_seconds","input_tokens","output_tokens","stationarity","primal","dual","complementarity"]
    summary={}
    for method in a.methods:
        runs=[]
        for seed in a.seeds:
            g=[r for r in records if r["method"]==method and r["seed"]==seed]
            runs.append({k:float(np.mean([r[k] for r in g])) for k in numeric})
        summary[method]={"n_problems":len(qps),"seeds":a.seeds,"n_seeds":len(a.seeds),
                         "mean":{k:float(np.mean([r[k] for r in runs])) for k in numeric},
                         "std_across_seeds":{k:float(np.std([r[k] for r in runs],ddof=1)) if len(runs)>1 else None for k in numeric}}
    dump_json(out/f"{a.split}_summary.json",summary)
    peak_cuda=None
    if policy:
        import torch
        if torch.cuda.is_available(): peak_cuda=int(torch.cuda.max_memory_allocated())
    dump_json(out/f"{a.split}_setup.json",{"peak_cuda_allocated_bytes":peak_cuda,"model_load_seconds":policy.load_seconds if policy else 0.,
              "model":a.model if policy else None,"adapter":a.adapter,"seeds":a.seeds,
              "n_problems":len(qps),"start":"cold","max_actions":20,"temperature":.7 if policy else None,"batch_size":a.batch_size if policy else 1,"evaluation_wall_seconds":perf_counter()-evaluation_start,"program_wall_seconds":perf_counter()-PROGRAM_START})
if __name__=="__main__": main()
