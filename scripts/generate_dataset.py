import argparse,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from kkt_agent.qp_generator import generate_qp
from kkt_agent.kkt_verifier import verify
from kkt_agent.qp_solver import solve_osqp
from kkt_agent.io import dump_json

def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",default="data"); p.add_argument("--scale",type=float,default=1.)
    args=p.parse_args(); root=Path(args.output); root.mkdir(parents=True,exist_ok=True)
    all_hashes=set(); summary={}
    for split,count,offset in [("train",5000,10000000),("validation",500,20000000),("test_id",1000,30000000),("test_ood",500,40000000)]:
        count=max(1,int(count*args.scale)); counts=Counter(); audit=[]; maximum=0.; errors=[]
        with (root/f"{split}.jsonl").open("w") as f,(root/f"{split}.oracle.jsonl").open("w") as o,(root/f"{split}.metadata.jsonl").open("w") as meta:
            for j in range(count):
                seed=offset+j; rng=np.random.default_rng(seed)
                n=16 if split=="test_ood" and j%2 else 12
                m=int(rng.integers(45,51) if n==16 and rng.random()<.5 else rng.integers(30,41))
                condition=float(rng.choice([10,100,1000],p=[.6,.3,.1]))
                if split=="test_ood" and n==12: condition=10000.
                family=["standard","small_slack","nearly_active","correlated"][(j//2 if split=="test_ood" else j)%4]
                q,oracle,md=generate_qp(seed,n,m,condition,family=family,problem_id=f"{split}-{j:05d}")
                cert=verify(q,oracle["x_star"],oracle["lambda_star"]); assert cert.success
                sol=solve_osqp(q); independent=verify(q,sol.x,sol.lam)
                if not independent.success: errors.append(q.problem_id)
                # Independent numerical solver must certify every persisted problem.
                if not independent.success: raise RuntimeError(f"OSQP verification failed: {q.problem_id}")
                maximum=max(maximum,independent.merit)
                payload=json.dumps(q.to_dict()); fingerprint=hashlib.sha256(q.Q.tobytes()+q.A.tobytes()+q.c.tobytes()+q.b.tobytes()).hexdigest()
                assert fingerprint not in all_hashes; all_hashes.add(fingerprint)
                f.write(payload+"\n"); o.write(json.dumps(oracle)+"\n"); meta.write(json.dumps(md)+"\n")
                counts[f"active_{md['active_size']}"]+=1; counts[f"condition_{condition:g}"]+=1; counts[family]+=1
                audit.append({"problem_id":q.problem_id,"osqp_success":independent.success,
                              "osqp_merit":independent.merit,"x_error_inf":float(np.max(np.abs(sol.x-oracle["x_star"]))),
                              "osqp_seconds":sol.seconds,"sha256":fingerprint})
        with (root/f"{split}.audit.jsonl").open("w") as a:
            for row in audit: a.write(json.dumps(row)+"\n")
        summary[split]={"count":count,"counts":dict(counts),"max_osqp_merit":maximum,"osqp_failures":errors,
                        "max_x_error_inf":max(r["x_error_inf"] for r in audit)}
        print(split,summary[split],flush=True)
    dump_json(root/"manifest.json",{"seed_scheme":"10M/20M/30M/40M + index","unique_qp_hashes":len(all_hashes),"splits":summary})
if __name__=="__main__": main()
