"""Offline teacher preparation; the online trainer reads no oracle file."""
import json
import numpy as np
from kkt_agent.io import load_qps,read_jsonl,dump_json
from kkt_agent.expert_policy import initial_set
qps=load_qps("data/train.jsonl")
oracles={r["problem_id"]:r for r in read_jsonl("data/train.oracle.jsonl")}
rng=np.random.default_rng(42); tasks=[]
for q in qps:
    for level,mode,difficulty in [(0,"missing",1),(1,"mixed",3),(2,"cold",5)]:
        tasks.append({"problem_id":q.problem_id,"initial_active":initial_set(q,oracles[q.problem_id],mode,rng,difficulty),"level":level})
with open("data/curriculum.jsonl","w") as f:
    for t in tasks: f.write(json.dumps(t)+"\n")
dump_json("results/curriculum_manifest.json",{"tasks":len(tasks),"seed":42,
          "oracle_access":"offline only; initial active sets are task inputs, never target active sets"})
