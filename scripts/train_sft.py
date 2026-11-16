"""TRL SFTTrainer with completion-only loss on real Qwen LoRA parameters."""
import argparse,json,time
from pathlib import Path
import yaml,torch
from datasets import Dataset
from trl import SFTTrainer,SFTConfig
from kkt_agent.llm import load_model,set_seed,trainable_snapshot,parameter_audit
from kkt_agent.io import read_jsonl,dump_json

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/sft.yaml")
    p.add_argument("--dialect",choices=["text","tool"],default="text"); p.add_argument("--steps",type=int); p.add_argument("--limit",type=int,default=12000)
    p.add_argument("--output",default="checkpoints/sft"); p.add_argument("--resume")
    a=p.parse_args(); cfg=yaml.safe_load(open(a.config))
    if a.steps: cfg["max_steps"]=a.steps
    cfg["dialect"]=a.dialect
    set_seed(cfg["seed"]); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    model,tok=load_model(cfg["model"],trainable=True,rank=cfg["rank"],alpha=cfg["alpha"])
    before=trainable_snapshot(model); assert all("lora_" in n for n in before); rows=[]
    for row in read_jsonl("data/sft_train.jsonl"):
        # Match compact online policy exactly; earlier actions affect the current
        # observation through the persistent environment, not through oracle text.
        msgs=[row["prompt"][0],row["prompt"][-1]]
        msgs[-1]={**msgs[-1],"content":msgs[-1]["content"].removeprefix("apply_action result:\n")}
        if a.dialect=="tool":
            from kkt_agent.active_set_env import parse_action
            from kkt_agent.tool_schema import TOOL_SCHEMA
            msgs[0]={"role":"system","content":"Repair the QP using exactly one apply_action call per turn. STOP only if verified=true. /no_think"}
            prompt=tok.apply_chat_template(msgs,tools=[TOOL_SCHEMA],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            act,idx=parse_action(row["action"])
            completion="<tool_call>"+json.dumps({"name":"apply_action","arguments":{"action":act,"index":-1 if idx is None else idx}})+"</tool_call>"+tok.eos_token
        else:
            prompt=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            completion=row["action"]+tok.eos_token
        rows.append({"prompt":prompt,"completion":completion})
        if len(rows)>=a.limit: break
    args=SFTConfig(output_dir=str(out),max_steps=cfg["max_steps"],learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["batch_size"],gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        max_length=cfg["max_length"],completion_only_loss=True,packing=False,
        gradient_checkpointing=True,gradient_checkpointing_kwargs={"use_reentrant":False},
        fp16=True,bf16=False,logging_steps=1,save_steps=max(2,cfg["max_steps"]//2),save_total_limit=2,
        report_to="none",seed=cfg["seed"],dataloader_num_workers=0,warmup_steps=max(1,int(cfg["max_steps"]*.05)),
        lr_scheduler_type="cosine",optim="adamw_torch",max_grad_norm=1.,dataset_num_proc=1)
    trainer=SFTTrainer(model=model,args=args,train_dataset=Dataset.from_list(rows),processing_class=tok)
    if trainer.accelerator.scaler is not None:
        trainer.accelerator.scaler=torch.amp.GradScaler("cuda",init_scale=128.)
    start=time.perf_counter(); trainer.train(resume_from_checkpoint=a.resume)
    trainer.save_model(str(out/"final")); tok.save_pretrained(out/"final")
    audit=parameter_audit(model,before); audit.update(seconds=time.perf_counter()-start,examples=len(rows),steps=trainer.state.global_step,config=cfg,base_revision=getattr(model.config,"kkt_base_revision",None))
    assert audit["changed_parameters"]>0 and audit["delta_l2"]>0
    dump_json(out/"parameter_audit.json",audit); dump_json(out/"training_log.json",trainer.state.log_history)
    print(json.dumps(audit,indent=2))
if __name__=="__main__": main()
