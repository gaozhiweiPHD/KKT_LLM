"""Optional native TRL environment_factory engine (tool-call dialect).
Use tool-format SFT before this engine; the primary measured engine uses the
short text-action dialect and is scripts/train_grpo.py.
"""
import argparse,inspect
from functools import partial
from datasets import Dataset
from trl import GRPOTrainer,GRPOConfig
from kkt_agent.llm import load_model
from kkt_agent.io import load_qps
from kkt_agent.active_set_env import ActiveSetEnv
from kkt_agent.observation import observation
from kkt_agent.trl_environment import QPToolEnvironment

p=argparse.ArgumentParser(); p.add_argument("--adapter",required=True); p.add_argument("--steps",type=int,default=2)
p.add_argument("--reward",choices=["A","B","C"],default="B"); a=p.parse_args()
if "environment_factory" not in inspect.signature(GRPOTrainer.__init__).parameters:
    raise RuntimeError("Install the pinned TRL version; no single-turn fallback.")
corpus={q.problem_id:q for q in load_qps("data/train.jsonl",100)}
rows=[{"problem_id":q.problem_id,"initial_active":[],"prompt":[{"role":"user","content":
       "Use apply_action for each ADD, DROP, or STOP. Exactly one call per turn. STOP requires verified=true.\n"+observation(ActiveSetEnv(q))}]} for q in corpus.values()]
model,tok=load_model(adapter=a.adapter,trainable=True)
args=GRPOConfig(output_dir="checkpoints/grpo_trl",max_steps=a.steps,num_generations=4,
                per_device_train_batch_size=1,gradient_accumulation_steps=4,max_completion_length=768,
                max_tool_calling_iterations=20,learning_rate=1e-5,fp16=True,bf16=False,
                chat_template_kwargs={"enable_thinking":False},temperature=.7,beta=0.,
                logging_steps=1,report_to="none",save_steps=a.steps)
trainer=GRPOTrainer(model=model,args=args,processing_class=tok,train_dataset=Dataset.from_list(rows),
                    environment_factory=partial(QPToolEnvironment,corpus,a.reward))
trainer.train(); trainer.save_model("checkpoints/grpo_trl/final")
