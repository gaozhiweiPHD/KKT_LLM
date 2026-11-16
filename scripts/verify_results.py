"""Recompute saved final certificates from public QPs; never read oracle data."""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
from kkt_agent.io import load_qps,read_jsonl,dump_json
from kkt_agent.kkt_verifier import verify
from kkt_agent.residual_audit import compare_residuals
from kkt_agent.active_set_env import parse_action


def audit_training():
    training={}
    reference=None
    for version in ("A","B","C"):
        root=Path("checkpoints")/f"grpo_{version}"
        logs=list(read_jsonl(root/"training.jsonl"))
        if [r["step"] for r in logs]!=list(range(1,201)):
            raise AssertionError(f"incomplete formal training: {version}")
        schedule=[(r["problem_id"],r["initial_active"],r["level"],r["horizon"]) for r in logs]
        if reference is None: reference=schedule
        elif schedule!=reference: raise AssertionError("reward ablations used different training tasks")
        counts=defaultdict(int)
        for trace in read_jsonl(root/"rollouts.jsonl"):
            step=trace["step"]; member=trace["group_member"]; record=trace["record"]
            row=logs[step-1]; history=trace["history"]; turns=trace["token_turns"]
            if record["problem_id"]!=row["problem_id"] or history[0]["active"]!=sorted(row["initial_active"]):
                raise AssertionError("group problem/initial state mismatch")
            if not 0<=member<4: raise AssertionError("bad group member")
            counts[(step,member)]+=1
            if len(turns)!=record["actions"] or len(history)!=len(turns)+1:
                raise AssertionError("missing action tokens")
            for turn in turns:
                lengths=[len(turn[k]) for k in ("completion_ids","old_replay_logprobs","sampling_logprobs")]
                if len(set(lengths))!=1 or lengths[0]==0: raise AssertionError("bad token mask")
                if not np.isfinite(turn["old_replay_logprobs"]+turn["sampling_logprobs"]).all():
                    raise AssertionError("nonfinite logged log probabilities")
            value=float(record["success"])
            if version!="A":
                value+=.5*np.clip(np.log1p(history[0]["residuals"]["merit"])-np.log1p(record["merit"]),-1,1)
            if version=="C": value-=.01*record["actions"]+.05*record["invalid_actions"]
            if not np.isclose(value,row["rewards"][member],atol=1e-12):
                raise AssertionError("reward mismatch")
        if counts!={(step,member):1 for step in range(1,201) for member in range(4)}:
            raise AssertionError("missing or duplicate group rollout")
        for row in logs:
            rewards=np.asarray(row["rewards"])
            advantage=(rewards-rewards.mean())/(rewards.std()+1e-4)
            if not np.allclose(advantage,row["advantages"]): raise AssertionError("incorrect group advantages")
            if not np.isfinite([row["loss"],row["grad_norm"]]).all(): raise AssertionError("nonfinite update")
        training[version]={"updates":200,"rollouts":800,
            "nonzero_variance_groups":sum(row["reward_std"]>1e-8 for row in logs),
            "max_sampling_replay_logprob_difference":max(row["sampling_logprob_roundoff_max"] for row in logs)}
    return training

def main():
    metrics=("stationarity","primal","dual","complementarity",
             "raw_stationarity","raw_primal","raw_dual","raw_complementarity")
    tolerances=(1e-5,1e-7,1e-9)
    groups=defaultdict(lambda:{"count":0,"reported_successes":0,"actions":0,"invalid_actions":0,"format_invalid_actions":0,
                              "certificate_counts":{str(t):0 for t in tolerances},
                              "successful_stop_counts":{str(t):0 for t in tolerances}})
    total=0
    roundoff_count=0; roundoff_examples=[]
    for split in ("test_id","test_ood"):
        qps={q.problem_id:q for q in load_qps(f"data/{split}.jsonl")}
        paths=[Path("results/baselines")/f"{split}_trajectories.jsonl"]
        paths += [Path("results")/f"qwen_{m}"/f"{split}_trajectories.jsonl"
                  for m in ("zero","sft","grpo_A","grpo_B","grpo_C")]
        seen=set()
        for path in paths:
            if not path.exists(): raise FileNotFoundError(path)
            setup=json.loads((path.parent/f"{split}_setup.json").read_text())
            if setup["n_problems"]!=len(qps) or setup["seeds"]!=[11,22,33]:
                raise AssertionError("incomplete test coverage or inconsistent seeds")
            if path.parent.name.startswith("qwen_"):
                label=path.parent.name.removeprefix("qwen_")
                expected=None if label=="zero" else f"checkpoints/{label}/final"
                if setup["adapter"]!=expected: raise AssertionError("mislabeled evaluation adapter")
            for row in read_jsonl(path):
                r,h=row["record"],row["history"]
                key=(r["method"],r["problem_id"],r["seed"])
                if key in seen: raise AssertionError(f"duplicate evaluation {key}")
                seen.add(key)
                candidate=h[-1]["candidate"]; qp=qps[r["problem_id"]]
                cert=verify(qp,candidate["x"],candidate["lambda"])
                if cert.success!=r["kkt_candidate"]: raise AssertionError(f"certificate mismatch {key}")
                reconciled=compare_residuals(cert,r,qp,candidate["x"],candidate["lambda"])
                roundoff_count+=len(reconciled)
                for item in reconciled:
                    if len(roundoff_examples)<10:
                        roundoff_examples.append({"key":key,**item})
                if r["success"] and not cert.success: raise AssertionError(f"false certificate {key}")
                # Numerical baselines have their own stopping rule.
                repair=r["method"] in ("random","greedy") or r["method"].startswith("qwen")
                stopped=False; format_invalid=0
                if repair:
                    try: stopped=parse_action(h[-1]["action"])[0]=="STOP"
                    except ValueError: pass
                    if r["success"] != (stopped and cert.success):
                        raise AssertionError(f"STOP/success mismatch {key}")
                    if r["actions"]>20: raise AssertionError(f"decision budget exceeded {key}")
                    for before,after in zip(h,h[1:]):
                        if after.get("invalid"):
                            if before["active"]!=after["active"]:
                                raise AssertionError(f"rejected action changed state {key}")
                            try: parse_action(after["action"])
                            except ValueError: format_invalid+=1
                else: stopped=r["success"]
                group=groups[(split,r["method"])]
                group["count"]+=1; group["reported_successes"]+=int(r["success"]); total+=1
                group["actions"]+=r["actions"]; group["invalid_actions"]+=r["invalid_actions"]
                group["format_invalid_actions"]+=format_invalid
                for tolerance in tolerances:
                    ok=verify(qp,candidate["x"],candidate["lambda"],tolerance).success
                    group["certificate_counts"][str(tolerance)]+=int(ok)
                    group["successful_stop_counts"][str(tolerance)]+=int(stopped and ok)
    expected_methods={"random","greedy","classical","osqp","qwen_zero","qwen_sft",
                      "qwen_grpo_A","qwen_grpo_B","qwen_grpo_C"}
    for split,expected_count in (("test_id",3000),("test_ood",1500)):
        if {method for sp,method in groups if sp==split}!=expected_methods:
            raise AssertionError("missing or unexpected baseline")
        if any(g["count"]!=expected_count for (sp,method),g in groups.items() if sp==split):
            raise AssertionError("incomplete evaluation records")
    rows=[{"split":s,"method":m,**g} for (s,m),g in sorted(groups.items())]
    dump_json("results/certificate_audit.json",{"checked_records":total,"mismatches":0,
              "raw_roundoff_reconciliations":roundoff_count,"raw_roundoff_examples":roundoff_examples,
              "tolerances":tolerances,"oracle_access":False,"groups":rows,"training_groups":audit_training(),
              "note":"Tolerance sweep re-verifies fixed final candidates; policies were not rerun at stricter tolerances."})
    print(f"Independently verified {total} final candidates; zero certificate/success mismatches; {roundoff_count} raw residual differences within arithmetic roundoff bounds.")

if __name__=="__main__": main()
