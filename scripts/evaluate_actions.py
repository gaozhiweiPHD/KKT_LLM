import argparse,json
from collections import Counter
import torch
import numpy as np
from kkt_agent.llm import load_model,set_seed
from kkt_agent.io import read_jsonl,dump_json
from kkt_agent.active_set_env import parse_action

p=argparse.ArgumentParser(); p.add_argument("--adapter"); p.add_argument("--limit",type=int,default=100)
p.add_argument("--output",default="results/action_accuracy.json"); a=p.parse_args()
set_seed(42); model,tok=load_model(adapter=a.adapter); model.eval(); results=[]; counts=Counter()
all_rows=list(read_jsonl("data/sft_validation.jsonl"))
selected=np.random.default_rng(42).choice(len(all_rows),size=min(a.limit,len(all_rows)),replace=False)
for row in (all_rows[int(i)] for i in selected):
    msgs=[row["prompt"][0],row["prompt"][-1]]
    msgs[-1]={**msgs[-1],"content":msgs[-1]["content"].removeprefix("apply_action result:\n")}
    text=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    ins=tok(text,return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out=model.generate(**ins,max_new_tokens=12,do_sample=False,pad_token_id=tok.pad_token_id)
    pred=tok.decode(out[0,ins.input_ids.shape[1]:],skip_special_tokens=True).strip()
    try: parse_action(pred); valid=True
    except ValueError: valid=False
    counts[row["action"].split("(")[0]]+=1
    results.append({"problem_id":row["problem_id"],"expected":row["action"],"predicted":pred,"exact":pred==row["action"],"parseable":valid})
    if len(results)>=a.limit: break
dump_json(a.output,{"n":len(results),"accuracy":sum(r["exact"] for r in results)/len(results),
                  "parse_rate":sum(r["parseable"] for r in results)/len(results),"label_counts":dict(counts),"results":results})
print(a.output)
