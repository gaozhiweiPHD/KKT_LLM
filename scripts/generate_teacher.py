import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np
from kkt_agent.io import load_qps,read_jsonl,dump_json
from kkt_agent.expert_policy import trajectory

def main():
    p=argparse.ArgumentParser(); p.add_argument("--split",default="train"); p.add_argument("--limit",type=int)
    p.add_argument("--seed",type=int,default=42); a=p.parse_args()
    qps=load_qps(f"data/{a.split}.jsonl",a.limit)
    oracles={r["problem_id"]:r for r in read_jsonl(f"data/{a.split}.oracle.jsonl")}
    rng=np.random.default_rng(a.seed); counts=Counter(); successes=0; total=0; rows_count=0
    modes=["cold","missing","extra","mixed","warm"]
    with open(f"data/sft_{a.split}.jsonl","w") as f,open(f"data/teacher_{a.split}_trajectories.jsonl","w") as t:
        for j,qp in enumerate(qps):
            # Two different initializations per training problem.
            for mode in [modes[j%5],modes[(j+2)%5]]:
                env,rows=trajectory(qp,oracles[qp.problem_id],mode,rng)
                assert env.success and env.invalid==0
                total+=1; successes+=env.success
                for row in rows:
                    f.write(json.dumps(row)+"\n"); counts[row["action"].split("(")[0]]+=1; rows_count+=1
                t.write(json.dumps({"record":env.record(),"history":env.history,"mode":mode})+"\n")
    dump_json(f"results/teacher_{a.split}.json",{"trajectories":total,"success_rate":successes/total,
              "action_distribution":dict(counts),"examples":rows_count,"seed":a.seed})
    print(total,successes,dict(counts),flush=True)
if __name__=="__main__": main()
